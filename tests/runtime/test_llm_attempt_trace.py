"""Provider diagnostics survive failed JSON parsing and bounded transport retries."""

import json
from io import BytesIO
from httpx import HTTPStatusError, Request, Response, TransportError

import pytest

from resagent2_runtime import ComposedContext, ModelProfile, OpenAICompatibleClient
from resagent2_runtime.models import AgentAction


def _response(content, *, finish_reason="stop", reasoning=None, tokens=10):
    return BytesIO(json.dumps({
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content, "reasoning_content": reasoning},
        }],
        "usage": {"completion_tokens": tokens},
    }).encode())


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "test-secret-not-for-trace")
    monkeypatch.setattr("resagent2_runtime.llm.time.sleep", lambda _: None)
    return OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1",
        api_key_env="TEST_LLM_KEY", trace_dir=tmp_path / "traces",
        trace_level="full",
    )


def _invoke(client):
    return client.next_action(ComposedContext(
        text="Return one JSON action.", included_sections=["system"],
        omitted_sections=[], estimated_tokens=6,
    ), AgentAction)


def _record(client):
    path = client.trace_dir / "llm_traces.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1  # HTTP retries are details, not extra logical calls.
    assert "test-secret-not-for-trace" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    return records[0]


def test_empty_length_limited_response_reaches_caller_with_diagnostics(client, monkeypatch):
    client.model_profile = ModelProfile(context_window=65536)
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.content))
        return _response(
            "", finish_reason="length", tokens=4096,
            reasoning=f"reasoning from attempt {len(requests)}",
        )

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    with pytest.raises(json.JSONDecodeError):
        _invoke(client)

    record = _record(client)
    assert [request["max_tokens"] for request in requests] == [4096]
    assert record["request_max_tokens"] == 4096
    assert client.last_attempts == record["retry_number"] + 1 == 1
    assert [a["retry_number"] for a in record["attempts"]] == [0]
    assert [a["raw_reasoning_text"] for a in record["attempts"]] == [
        "reasoning from attempt 1"
    ]
    for attempt in [record, *record["attempts"]]:
        assert attempt["finish_reason"] == "length"
        assert attempt["usage"] == {"completion_tokens": 4096}
        assert attempt["raw_response_text"] == ""
        assert attempt["action_valid"] is False
        assert attempt["validation_error"]
    # Trace data does not change the next request or increase its output budget.
    assert len(requests) == 1
    assert "reasoning from attempt" not in record["request_text"]


def test_success_after_bad_json_is_a_new_logical_call(client, monkeypatch):
    responses = iter([
        _response("broken JSON", finish_reason="length", reasoning="earlier", tokens=40),
        _response('{"tool":"finish"}', reasoning="final", tokens=20),
    ])
    monkeypatch.setattr("resagent2_runtime.llm.send_request", lambda *a, **k: next(responses))
    with pytest.raises(json.JSONDecodeError):
        _invoke(client)
    first = _record(client)
    assert first["raw_response_text"] == "broken JSON"
    assert first["usage"] == {"completion_tokens": 40}
    assert first["validation_error"]
    assert _invoke(client) == {"tool": "finish"}
    rows = [json.loads(line) for line in
            (client.trace_dir / "llm_traces.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    record = rows[1]
    assert first["call_id"] != record["call_id"]
    assert record["request_max_tokens"] is None  # Provider default, not unlimited.
    assert record["finish_reason"] == "stop"
    assert record["usage"] == {"completion_tokens": 20}
    assert record["raw_reasoning_text"] == "final"
    assert record["validation_error"] is None
    assert record["action_valid"] is True
    assert record["retry_number"] == 0
    assert len(record["attempts"]) == client.last_attempts == 1
    client.record_validation("caller rejected schema")
    rows = [json.loads(line) for line in
            (client.trace_dir / "llm_traces.jsonl").read_text().splitlines()]
    assert rows[2]["call_id"] == record["call_id"]
    assert rows[2]["schema_validation_error"] == "caller rejected schema"


@pytest.mark.parametrize("failure", ["network", "timeout", "http500", "http400"])
def test_later_transport_failure_cannot_reuse_earlier_response(
    client, monkeypatch, failure,
):
    client.set_attempt_limit(2)
    if failure.startswith("http"):
        error = HTTPStatusError(
            "failed", request=Request("POST", "https://example.invalid"),
            response=Response(int(failure[4:]), content=b"request failed"),
        )
    elif failure == "timeout":
        error = TimeoutError("timed out")
    else:
        error = TransportError("unavailable")
    responses = iter([_response("", finish_reason="length", reasoning="old", tokens=40), error])

    def respond(*args, **kwargs):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    with pytest.raises(json.JSONDecodeError):
        _invoke(client)
    first = _record(client)
    assert first["usage"] == {"completion_tokens": 40}
    assert first["raw_reasoning_text"] == "old"
    client.set_attempt_limit(1)
    with pytest.raises(RuntimeError):
        _invoke(client)
    records = [json.loads(line) for line in
               (client.trace_dir / "llm_traces.jsonl").read_text().splitlines()]
    assert len(records) == 2
    record = records[-1]
    assert record["call_id"] != first["call_id"]
    assert client.last_attempts == len(record["attempts"]) == 1
    for attempt in (record, record["attempts"][0]):
        assert attempt["raw_response_text"] is None
        assert attempt["raw_reasoning_text"] is None
        assert attempt["usage"] is None
        assert attempt["finish_reason"] is None
        assert attempt["validation_error"]
        assert attempt["action_valid"] is False


@pytest.mark.parametrize("content, invalid", [
    ("PRIVATE_BROKEN_JSON", True),
    ('{"tool":"finish","arguments":{"secret":"PRIVATE_ACTION"}}', False),
])
def test_metadata_does_not_leak_content_through_attempts(client, monkeypatch, content, invalid):
    client.trace_level = "metadata"
    monkeypatch.setattr("resagent2_runtime.llm.send_request", lambda *a, **k:
                        _response(content, reasoning="PRIVATE_REASONING", tokens=40))
    if invalid:
        with pytest.raises(json.JSONDecodeError):
            _invoke(client)
    else:
        _invoke(client)
    record = _record(client)
    assert "PRIVATE_" not in json.dumps(record)
    for attempt in [record, *record["attempts"]]:
        assert "raw_response_text" not in attempt
        assert "raw_reasoning_text" not in attempt
        assert "parsed_action" not in attempt
        assert attempt["response_sha256"]
        assert attempt["usage"]
        assert attempt["finish_reason"] == "stop"


def test_missing_optional_provider_metadata_is_not_invented(client, monkeypatch):
    response = BytesIO(b'{"choices":[{"message":{"content":"{\\"tool\\":\\"finish\\"}"}}]}')
    monkeypatch.setattr("resagent2_runtime.llm.send_request", lambda *a, **k: response)
    assert _invoke(client) == {"tool": "finish"}
    record = _record(client)
    assert len(record["attempts"]) == 1
    assert record["request_max_tokens"] is None
    assert record["finish_reason"] is None
    assert record["usage"] is None


def test_trace_off_still_keeps_failed_output_accounting_without_files(client, monkeypatch):
    client.trace_level = "off"
    monkeypatch.setattr("resagent2_runtime.llm.send_request", lambda *a, **k: _response(""))
    with pytest.raises(json.JSONDecodeError):
        _invoke(client)
    assert client.last_attempts == 1
    assert not client.trace_dir.exists()


@pytest.mark.parametrize("payload", [[], {"choices": [None]}, {"choices": [{"message": []}]}])
def test_invalid_response_envelope_remains_a_recorded_failure(client, monkeypatch, payload):
    client.set_attempt_limit(1)
    monkeypatch.setattr("resagent2_runtime.llm.send_request", lambda *a, **k: BytesIO(json.dumps(payload).encode()))
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        _invoke(client)
    assert _record(client)["validation_error"]
    assert client.last_attempts == 1


def test_envelope_json_error_still_retries_as_provider_failure(client, monkeypatch):
    responses = iter([BytesIO(b'broken HTTP response'), _response('{"tool":"finish"}')])
    monkeypatch.setattr("resagent2_runtime.llm.send_request", lambda *a, **k: next(responses))
    assert _invoke(client) == {"tool": "finish"}
    record = _record(client)
    assert client.last_attempts == len(record["attempts"]) == 2
    assert record["attempts"][0]["raw_response_text"] is None
    assert record["attempts"][0]["validation_error"]
    assert record["attempts"][1]["action_valid"] is True
