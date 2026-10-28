"""External JSON services obey time/cancellation without claiming LLM usage."""
import asyncio
import json
import stat
import traceback
from unittest import mock

import httpx
import pytest

from resagent2_runtime import ServiceRequestClient, ServiceRequestError, ServiceRequestHTTPError
from resagent2_runtime import http
from resagent2_runtime.budget import DeadlineExceededError, MemoryUsage, execution_budget
from resagent2_runtime.cancellation import CancellationToken, OperationCancelled, bind_cancellation

KEY = "PRIVATE_SERVICE_SECRET"
BODY = {"query": "PUBLIC_QUERY", "count": 10, "need_content": False}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_SERVICE_KEY", KEY)
    return ServiceRequestClient(endpoint="https://service.example.test/search",
        api_key_env="TEST_SERVICE_KEY", trace_dir=tmp_path / "trace", trace_level="full")


def transport(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(http.httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))


def records(client):
    return [json.loads(line) for line in
        (client.trace_dir / "service_requests.jsonl").read_text().splitlines()]


def test_service_can_complete_with_zero_model_allowance_and_never_changes_wallet(client, monkeypatch):
    usage, requests = MemoryUsage(), []
    def respond(request):
        requests.append(request)
        assert usage.used == 0
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == BODY
        return httpx.Response(200, json={"results": []})
    transport(monkeypatch, respond)
    client.set_trace_context(run_id="run_service", task_id="task_root", session_id="session_service",
        provider="deepseek", operation="web_search", tool="web_search", step=3)
    with execution_budget(max_llm_calls=0, timeout_seconds=30, usage=usage):
        assert client.request(BODY) == {"results": []}
    assert usage.requests == {} and len(requests) == client.last_attempts == 1
    assert not (client.trace_dir / "llm_traces.jsonl").exists()
    record, = records(client)
    assert record["request_kind"] == "external_service"
    assert record["protocol"] == "json-service-request/v1"
    assert record["provider"] == "deepseek" and record["step"] == 3
    assert record["outcome"] == "succeeded" and record["response_valid"] is True
    assert "model" not in record and "usage" not in record
    assert stat.S_IMODE(client.trace_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((client.trace_dir / "service_requests.jsonl").stat().st_mode) == 0o600


@pytest.mark.parametrize("failure,outcome,error_type", [
    ("http429", "failed", "http_error"), ("network", "unknown", "network_error"),
    ("timeout", "unknown", "timeout"), ("json", "failed", "invalid_response"),
    ("list", "failed", "invalid_response"), ("utf8", "failed", "invalid_response"),
    ("gzip", "failed", "invalid_response"), ("large", "failed", "response_too_large"),
])
def test_failures_are_safe_not_retried_and_not_model_calls(client, monkeypatch, failure, outcome, error_type):
    calls, usage = [], MemoryUsage()
    def respond(request):
        calls.append(request)
        if failure == "http429":
            return httpx.Response(429, text=f"PRIVATE_ERROR {KEY}", headers={"Retry-After": "17"})
        if failure == "network":
            raise httpx.ConnectError(f"PRIVATE_ERROR {KEY}", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout(f"PRIVATE_ERROR {KEY}", request=request)
        if failure == "gzip":
            return httpx.Response(200, headers={"content-encoding": "gzip"},
                stream=httpx.ByteStream(f"PRIVATE_ERROR {KEY}".encode()))
        content = {"json": b"PRIVATE_INVALID_JSON", "list": b"[]", "utf8": b"\xff", "large": b"x" * 100}
        return httpx.Response(200, content=content[failure])
    client.max_response_bytes = 50
    transport(monkeypatch, respond)
    with execution_budget(max_llm_calls=0, timeout_seconds=30, usage=usage):
        with pytest.raises(ServiceRequestError) as raised:
            client.request(BODY)
    assert raised.value.error_type == error_type
    assert usage.requests == {} and len(calls) == client.last_attempts == 1
    error = "".join(traceback.format_exception(raised.value))
    assert KEY not in error and "PRIVATE_ERROR" not in error
    record, = records(client)
    assert record["outcome"] == outcome and record["error_type"] == error_type
    assert record["retry_number"] == 0 and record["response_valid"] is False
    assert KEY not in json.dumps(record) and "PRIVATE_ERROR" not in json.dumps(record)
    if failure == "http429":
        assert isinstance(raised.value, ServiceRequestHTTPError)
        assert raised.value.status_code == 429 and raised.value.retry_after == "17"
        assert not hasattr(raised.value, "response")


@pytest.mark.parametrize("trace_level", ["off", "metadata", "full"])
def test_service_trace_levels_redact_keys_and_preserve_only_requested_content(client, monkeypatch, trace_level):
    client.trace_level = trace_level
    client.set_trace_context(private_echo=KEY)
    transport(monkeypatch, lambda request: httpx.Response(200, json={"echo": KEY, "text": "PRIVATE_RESPONSE"}))
    assert client.request({"query": "PRIVATE_QUERY", "echo": KEY})["echo"] == KEY
    if trace_level == "off":
        assert not client.trace_dir.exists()
        return
    text = (client.trace_dir / "service_requests.jsonl").read_text()
    record, = records(client)
    assert KEY not in text and record["private_echo"] == "[REDACTED]"
    assert "Authorization" not in text
    if trace_level == "metadata":
        assert "PRIVATE_QUERY" not in text and "PRIVATE_RESPONSE" not in text
        assert record["request_sha256"] and record["response_sha256"]
        assert "request_text" not in record and "raw_response_text" not in record
    else:
        assert json.loads(record["request_text"])["echo"] == "[REDACTED]"
        assert json.loads(record["raw_response_text"])["text"] == "PRIVATE_RESPONSE"


def test_expired_run_does_not_dispatch_or_write_attempt(client, monkeypatch):
    transport(monkeypatch, lambda request: pytest.fail("expired service dispatched"))
    with execution_budget(max_llm_calls=0, timeout_seconds=0):
        with pytest.raises(DeadlineExceededError):
            client.request(BODY)
    assert client.last_attempts == 0 and not client.trace_dir.exists()


@pytest.mark.parametrize("run_limited", [False, True])
def test_service_time_is_clipped_to_run_and_local_timeout(client, monkeypatch, run_limited):
    async def respond(request):
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={})
    transport(monkeypatch, respond)
    client.timeout_seconds = 20 if run_limited else 0.01
    with execution_budget(max_llm_calls=0, timeout_seconds=0.01 if run_limited else 30) as budget:
        with pytest.raises(DeadlineExceededError if run_limited else ServiceRequestError):
            client.request(BODY)
        assert budget.usage.used == 0
    assert records(client)[0]["error_type"] == ("deadline_exceeded" if run_limited else "timeout")


def test_cancelled_service_never_dispatches(client, monkeypatch):
    transport(monkeypatch, lambda request: pytest.fail("cancelled service dispatched"))
    cancellation = CancellationToken()
    cancellation.cancel("User stopped")
    with bind_cancellation(cancellation):
        with pytest.raises(OperationCancelled):
            client.request(BODY)
    assert client.last_attempts == 0 and not client.trace_dir.exists()


def test_inflight_cancellation_uses_shared_http_and_never_becomes_model_usage(client, monkeypatch):
    cancellation = CancellationToken()
    async def respond(request):
        cancellation.cancel("User stopped")
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={})
    transport(monkeypatch, respond)
    with execution_budget(max_llm_calls=0, timeout_seconds=30) as budget, bind_cancellation(cancellation):
        with pytest.raises(OperationCancelled):
            client.request(BODY)
    assert budget.usage.used == 0
    assert records(client)[0]["error_type"] == "interrupted"
    assert records(client)[0]["outcome"] == "unknown"


def test_missing_credentials_does_not_dispatch(client, monkeypatch):
    monkeypatch.delenv("TEST_SERVICE_KEY")
    transport(monkeypatch, lambda request: pytest.fail("unauthenticated service dispatched"))
    with pytest.raises(ServiceRequestError) as raised:
        client.request(BODY)
    assert raised.value.error_type == "missing_credentials"
    assert client.last_attempts == 0 and not client.trace_dir.exists()


@pytest.mark.parametrize("http_error", [False, True])
def test_service_diagnostic_failure_preserves_real_result(client, monkeypatch, caplog, http_error):
    transport(monkeypatch, lambda request: httpx.Response(429 if http_error else 200,
        json={"results": []}, headers={"Retry-After": "17"}))
    with mock.patch("resagent2_runtime._trace.os.open", side_effect=OSError(f"PRIVATE_PATH {KEY}")):
        if http_error:
            with pytest.raises(ServiceRequestHTTPError) as raised:
                client.request(BODY)
            assert raised.value.status_code == 429
        else:
            assert client.request(BODY) == {"results": []}
    assert "Could not write optional service trace (OSError)." in caplog.text
    assert KEY not in caplog.text and "PRIVATE_PATH" not in caplog.text


@pytest.mark.parametrize("changes", [
    {"endpoint": "file:///tmp/service"}, {"endpoint": "https://user:secret@service.test/request"},
    {"api_key_env": ""}, {"timeout_seconds": 0}, {"timeout_seconds": float("nan")},
    {"max_response_bytes": 0}, {"max_response_bytes": 1.5},
    {"trace_level": "verbose"}, {"trace_level": "full"},
])
def test_service_rejects_invalid_configuration(changes):
    with pytest.raises(ValueError):
        ServiceRequestClient(**{"endpoint": "https://service.test/request", "api_key_env": "TEST_SERVICE_KEY", **changes})

@pytest.mark.parametrize("value,expected", [
    ("Fri, 31 Dec 1999 23:59:59 GMT", "Fri, 31 Dec 1999 23:59:59 GMT"),
    (f"Fri, 31 Dec 1999 23:59:59 GMT {KEY}", None),
])
def test_service_retry_after_preserves_dates_but_not_provider_text(client, monkeypatch, value, expected):
    transport(monkeypatch, lambda request: httpx.Response(429, headers={"Retry-After": value}))
    with pytest.raises(ServiceRequestHTTPError) as raised:
        client.request(BODY)
    assert raised.value.retry_after == expected
    record, = records(client)
    assert record["retry_after"] == expected
    assert KEY not in json.dumps(record)
