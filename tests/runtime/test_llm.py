"""OpenAI-compatible client retries and trace behavior."""

import json
from io import BytesIO
from unittest import mock
from httpx import HTTPStatusError, Request, Response, TransportError

import pytest

from resagent2_runtime import ComposedContext, ModelProfile, OpenAICompatibleClient
from resagent2_runtime.models import AgentAction


class _FakeResponse:
    """Context-manager stand-in for a urlopen response."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _client(monkeypatch) -> OpenAICompatibleClient:
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    return OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
    )


def _context() -> ComposedContext:
    return ComposedContext(
        text="do something",
        included_sections=[],
        omitted_sections=[],
        estimated_tokens=0,
    )


@pytest.mark.parametrize("level,directory,message", [
    ("verbose", "traces", "trace_level must be"),
    ("", None, "trace_level must be"),
    ("metadata", None, "trace_dir is required"),
    ("full", "", "trace_dir is required"),
])
def test_invalid_trace_configuration_fails_during_construction(level, directory, message):
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleClient(
            model="test", api_base="https://example.com/v1", api_key_env="UNUSED_KEY",
            trace_level=level, trace_dir=directory,
        )


def test_trace_directory_is_resolved_before_working_directory_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.com/v1", api_key_env="UNUSED_KEY",
        trace_level="full", trace_dir="traces",
    )
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    client._write_trace({"call_id": "test"})
    assert (tmp_path / "traces/llm_traces.jsonl").is_file()
    assert not (other / "traces").exists()


def test_model_profile_combines_model_capacity_and_component_limit() -> None:
    profile = ModelProfile(
        context_window=10_000,
        reserved_output_tokens=2_000,
        safety_margin_tokens=1_000,
    )

    assert profile.input_budget(schema_tokens=500, component_limit=8_000) == 6_500
    assert profile.input_budget(schema_tokens=500, component_limit=4_000) == 4_000


def test_model_profile_rejects_impossible_limits() -> None:
    with pytest.raises(ValueError, match="no room for input"):
        ModelProfile(
            context_window=4_000,
            reserved_output_tokens=3_000,
            safety_margin_tokens=1_000,
        )


def test_client_reserves_schema_and_output_tokens(monkeypatch) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    profile = ModelProfile(
        context_window=4_000,
        reserved_output_tokens=1_000,
        safety_margin_tokens=500,
    )
    client = OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
        model_profile=profile,
    )
    budget = client.context_budget(AgentAction, component_limit=3_000)
    assert 0 < budget < 2_500

    ok = _FakeResponse(
        {"choices": [{"message": {"content": json.dumps({"tool": "finish"})}}]}
    )
    with mock.patch("resagent2_runtime.llm.send_request", return_value=ok) as urlopen_mock:
        client.next_action(_context(), AgentAction)
    request = urlopen_mock.call_args.args[0]
    assert json.loads(request.content)["max_tokens"] == 1_000


def test_transient_llm_failure_is_retried(monkeypatch) -> None:
    client = _client(monkeypatch)
    ok = _FakeResponse(
        {"choices": [{"message": {"content": json.dumps({"tool": "finish"})}}]}
    )
    with (
        mock.patch("resagent2_runtime.llm.time.sleep"),
        mock.patch(
            "resagent2_runtime.llm.send_request", side_effect=[TransportError("boom"), ok]
        ) as urlopen_mock,
    ):
        result = client.next_action(_context(), AgentAction)

    assert result == {"tool": "finish"}
    assert urlopen_mock.call_count == 2


def test_trace_writes_jsonl_when_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    client = OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
        trace_dir=tmp_path / "traces",
        trace_level="full",
    )
    client.set_trace_context(
        run_id="run_x", session_id="session_s", agent="test", step=1
    )
    ok = _FakeResponse(
        {"choices": [{"message": {"content": json.dumps({"tool": "finish"})}}]}
    )
    with mock.patch("resagent2_runtime.llm.send_request", return_value=ok):
        result = client.next_action(_context(), AgentAction)

    assert result == {"tool": "finish"}
    trace_file = tmp_path / "traces" / "llm_traces.jsonl"
    records = [
        json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["run_id"] == "run_x"
    assert record["session_id"] == "session_s"
    assert record["agent"] == "test"
    assert record["step"] == 1
    assert record["model"] == "test-model"
    assert record["request_text"]
    assert record["raw_response_text"] == json.dumps({"tool": "finish"})
    assert record["parsed_action"] == {"tool": "finish"}

    assert record["raw_reasoning_text"] is None

def test_trace_metadata_level_omits_full_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    client = OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
        trace_dir=tmp_path / "traces",
        trace_level="metadata",
    )
    ok = _FakeResponse(
        {"choices": [{"message": {"content": json.dumps({"tool": "finish"}), "reasoning_content": "visible rationale"}}]}
    )
    with mock.patch("resagent2_runtime.llm.send_request", return_value=ok):
        client.next_action(_context(), AgentAction)

    trace_file = tmp_path / "traces" / "llm_traces.jsonl"
    record = json.loads(trace_file.read_text(encoding="utf-8").splitlines()[0])
    assert "request_text" not in record
    assert "raw_response_text" not in record
    assert "parsed_action" not in record
    assert "raw_reasoning_text" not in record
    assert record["request_sha256"]
    assert record["response_sha256"]
    assert record["action_sha256"]
    assert record["tool"] == "finish"
    assert record["action_valid"] is True


def test_full_trace_preserves_provider_reasoning(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    client = OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
        trace_dir=tmp_path / "traces",
        trace_level="full",
    )
    response = _FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"tool": "finish"}),
                        "reasoning_content": "provider-visible reasoning",
                    }
                }
            ]
        }
    )
    with mock.patch("resagent2_runtime.llm.send_request", return_value=response):
        client.next_action(_context(), AgentAction)

    trace_file = tmp_path / "traces" / "llm_traces.jsonl"
    record = json.loads(trace_file.read_text(encoding="utf-8").splitlines()[0])
    assert record["raw_reasoning_text"] == "provider-visible reasoning"

def test_trace_preserves_bad_json_response(monkeypatch, tmp_path) -> None:

    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    client = OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
        trace_dir=tmp_path / "traces",
        trace_level="full",
    )
    bad = _FakeResponse(
        {"choices": [{"message": {"content": "not valid json"}}]}
    )
    with (
        mock.patch("resagent2_runtime.llm.time.sleep"),
        mock.patch("resagent2_runtime.llm.send_request", return_value=bad),
    ):
        with pytest.raises(json.JSONDecodeError, match="Expecting value"):
            client.next_action(_context(), AgentAction)

    trace_file = tmp_path / "traces" / "llm_traces.jsonl"
    record = json.loads(trace_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["action_valid"] is False
    assert record["raw_response_text"] == "not valid json"
    assert record["validation_error"]


def test_transient_failure_exhausts_retries(monkeypatch) -> None:
    client = _client(monkeypatch)
    with (
        mock.patch("resagent2_runtime.llm.time.sleep"),
        mock.patch("resagent2_runtime.llm.send_request", side_effect=TransportError("boom")),
    ):
        with pytest.raises(RuntimeError, match="3 attempts"):
            client.next_action(_context(), AgentAction)


def test_attempt_limit_caps_transient_retries(monkeypatch) -> None:
    client = _client(monkeypatch)
    client.set_attempt_limit(1)
    with mock.patch(
        "resagent2_runtime.llm.send_request", side_effect=TransportError("boom")
    ) as urlopen_mock:
        with pytest.raises(RuntimeError, match="after 1 attempts"):
            client.next_action(_context(), AgentAction)
    assert urlopen_mock.call_count == 1
    assert client.last_attempts == 1


def test_client_error_counts_one_attempt_after_prior_retry(monkeypatch) -> None:
    client = _client(monkeypatch)
    ok = _FakeResponse(
        {"choices": [{"message": {"content": json.dumps({"tool": "finish"})}}]}
    )
    with (
        mock.patch("resagent2_runtime.llm.time.sleep"),
        mock.patch("resagent2_runtime.llm.send_request", side_effect=[TransportError("boom"), ok]),
    ):
        client.next_action(_context(), AgentAction)
    assert client.last_attempts == 2

    error = HTTPStatusError(
        "bad request",
        request=Request("POST", "https://example.com"),
        response=Response(400, content=b"invalid"),
    )
    with mock.patch("resagent2_runtime.llm.send_request", side_effect=error):
        with pytest.raises(RuntimeError, match="LLM HTTP 400"):
            client.next_action(_context(), AgentAction)
    assert client.last_attempts == 1


def test_malformed_json_returns_to_caller_without_identical_retry(monkeypatch) -> None:
    client = _client(monkeypatch)
    malformed = _FakeResponse(
        {"choices": [{"message": {"content": "not valid json"}}]}
    )
    ok = _FakeResponse(
        {"choices": [{"message": {"content": json.dumps({"tool": "finish"})}}]}
    )
    with (
        mock.patch("resagent2_runtime.llm.time.sleep"),
        mock.patch(
            "resagent2_runtime.llm.send_request", side_effect=[malformed, ok]
        ) as urlopen_mock,
    ):
        with pytest.raises(json.JSONDecodeError):
            client.next_action(_context(), AgentAction)

    assert client.last_attempts == urlopen_mock.call_count == 1


@pytest.mark.parametrize("content", [None, ["a", "b"], {"tool": "finish"}, 42])
def test_non_string_content_is_a_controlled_failure(monkeypatch, content) -> None:
    client = _client(monkeypatch)
    bad = _FakeResponse({"choices": [{"message": {"content": content}}]})
    with (
        mock.patch("resagent2_runtime.llm.time.sleep"),
        mock.patch("resagent2_runtime.llm.send_request", return_value=bad),
    ):
        with pytest.raises(RuntimeError, match="3 attempts"):
            client.next_action(_context(), AgentAction)

    assert client.last_attempts == 3


def test_full_trace_preserves_non_string_content(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    client = OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
        trace_dir=tmp_path / "traces",
        trace_level="full",
    )
    bad = _FakeResponse({"choices": [{"message": {"content": ["a", "b"]}}]})
    with (
        mock.patch("resagent2_runtime.llm.time.sleep"),
        mock.patch("resagent2_runtime.llm.send_request", return_value=bad),
    ):
        with pytest.raises(RuntimeError, match="3 attempts"):
            client.next_action(_context(), AgentAction)

    trace_file = tmp_path / "traces" / "llm_traces.jsonl"
    record = json.loads(trace_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["action_valid"] is False
    assert record["raw_response_text"] == ["a", "b"]
    assert record["validation_error"]


def test_metadata_trace_handles_non_string_content(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    client = OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
        trace_dir=tmp_path / "traces",
        trace_level="metadata",
    )
    bad = _FakeResponse({"choices": [{"message": {"content": {"tool": "finish"}}}]})
    with (
        mock.patch("resagent2_runtime.llm.time.sleep"),
        mock.patch("resagent2_runtime.llm.send_request", return_value=bad),
    ):
        with pytest.raises(RuntimeError, match="3 attempts"):
            client.next_action(_context(), AgentAction)

    trace_file = tmp_path / "traces" / "llm_traces.jsonl"
    record = json.loads(trace_file.read_text(encoding="utf-8").splitlines()[-1])
    assert "raw_response_text" not in record
    assert record["response_sha256"]
    assert record["action_valid"] is False


@pytest.mark.parametrize("failure", [None, "http400", "json"])
def test_trace_storage_failure_preserves_model_outcome(
    monkeypatch, tmp_path, caplog, failure,
):
    from resagent2_runtime.budget import MemoryUsage, execution_budget

    client = _client(monkeypatch)
    client.trace_level = "full"
    client.trace_dir = tmp_path / "PRIVATE_TRACE_PATH"
    client.trace_dir.mkdir()
    # A directory at the JSONL path exercises a real storage failure.
    (client.trace_dir / "llm_traces.jsonl").mkdir()
    content = "PRIVATE_INVALID_JSON" if failure == "json" else json.dumps({"tool": "finish"})
    response = _FakeResponse({"choices": [{"message": {"content": content}}]})
    if failure == "http400":
        response = HTTPStatusError(
            "original provider error", request=Request("POST", "https://example.com"),
            response=Response(400, content=b"original rejection"),
        )
    usage = MemoryUsage()
    with (
        execution_budget(max_llm_calls=3, timeout_seconds=30, usage=usage),
        mock.patch("resagent2_runtime.llm.send_request", side_effect=[response]) as provider,
    ):
        if failure == "http400":
            with pytest.raises(RuntimeError, match="LLM HTTP 400: original rejection"):
                client.next_action(_context(), AgentAction)
        elif failure == "json":
            with pytest.raises(json.JSONDecodeError):
                client.next_action(_context(), AgentAction)
        else:
            assert client.next_action(_context(), AgentAction) == {"tool": "finish"}

    assert provider.call_count == client.last_attempts == usage.used == 1
    assert list(usage.requests.values()) == ["succeeded" if failure is None else "failed"]
    assert "Could not write optional LLM trace (IsADirectoryError)." in caplog.text
    assert "PRIVATE_" not in caplog.text
    assert str(client.trace_dir) not in caplog.text


def test_trace_serialization_failure_preserves_model_response(monkeypatch, tmp_path, caplog):
    client = _client(monkeypatch)
    client.trace_level = "metadata"
    client.trace_dir = tmp_path / "traces"
    circular = {}
    circular["private"] = circular
    client.set_trace_context(diagnostic=circular)
    response = _FakeResponse({"choices": [{"message": {"content": json.dumps({"tool": "finish"})}}]})
    with mock.patch("resagent2_runtime.llm.send_request", return_value=response):
        assert client.next_action(_context(), AgentAction) == {"tool": "finish"}
    assert "Could not write optional LLM trace (ValueError)." in caplog.text
