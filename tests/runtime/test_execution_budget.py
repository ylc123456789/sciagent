from e2e.native_fixtures import tool_turns
"""One wallet and one deadline cover nested invocations and transport retries."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from threading import Event, Thread
import time

import httpx
import pytest

from sciagent_runtime import ComposedContext, OpenAICompatibleClient, ScriptedLLMClient
from sciagent_runtime.budget import (
    BudgetExhaustedError, DeadlineExceededError, MemoryUsage, current_budget,
    execution_budget, invoke_model, remaining_timeout,
)
from sciagent_runtime.llm import send_request
from sciagent_runtime.models import AgentAction


def _context():
    return ComposedContext(
        text="finish", included_sections=[], omitted_sections=[], estimated_tokens=1,
    )


def _response():
    return BytesIO(json.dumps({
        "choices": [{"message": {"content": '{"tool":"finish"}'}}],
    }).encode())


def _client(monkeypatch):
    monkeypatch.setenv("TEST_BUDGET_KEY", "test")
    return OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1",
        api_key_env="TEST_BUDGET_KEY",
    )


def test_nested_scopes_share_usage_and_cannot_extend_the_deadline():
    usage = MemoryUsage()
    with execution_budget(max_llm_calls=2, timeout_seconds=10, usage=usage) as parent:
        parent.charge("first", 0)
        with execution_budget(max_llm_calls=200, timeout_seconds=200) as child:
            assert child.usage is usage
            assert child.remaining_calls == 1
            assert child.deadline == parent.deadline
            child.charge("second", 0)
        assert current_budget() is parent
        with pytest.raises(BudgetExhaustedError):
            parent.charge("third", 0)
    assert current_budget() is None
    assert usage.used == 2


def test_nested_scope_cannot_replace_usage_or_reset_expired_time():
    with execution_budget(max_llm_calls=2, timeout_seconds=0):
        with pytest.raises(ValueError, match="cannot replace"):
            with execution_budget(max_llm_calls=2, timeout_seconds=30, usage=MemoryUsage()):
                pass
        with execution_budget(max_llm_calls=2, timeout_seconds=30) as child:
            with pytest.raises(DeadlineExceededError):
                child.charge("late", 0)
            with pytest.raises(DeadlineExceededError):
                remaining_timeout(30)


def test_minimal_clients_use_the_same_wallet_once_per_invocation():
    client = ScriptedLLMClient(tool_turns([{"tool": "finish"}] * 2))
    with execution_budget(max_llm_calls=1, timeout_seconds=30) as budget:
        assert invoke_model(client, "next_tool_call", _context(), [], [], max_input_tokens=512).tool_calls[0].name == "finish"
        with pytest.raises(BudgetExhaustedError):
            invoke_model(client, "next_tool_call", _context(), [], [], max_input_tokens=512)
        assert list(budget.usage.requests.values()) == ["succeeded"]
        assert len(client.contexts) == 1


def test_failed_persistence_prevents_network_dispatch(monkeypatch):
    class UnavailableUsage(MemoryUsage):
        def charge(self, call_id, retry_index):
            raise OSError("cannot persist")

    client = _client(monkeypatch)
    monkeypatch.setattr("sciagent_runtime.llm.send_request", lambda *a, **k:
                        pytest.fail("HTTP must not start without durable charge"))
    with execution_budget(max_llm_calls=4, timeout_seconds=30, usage=UnavailableUsage()):
        with pytest.raises(OSError, match="cannot persist"):
            client.next_action(_context(), AgentAction)
    assert client.last_attempts == 0


def test_http_attempts_are_precharged_and_retries_have_distinct_entries(monkeypatch):
    client = _client(monkeypatch)
    usage = MemoryUsage()
    monkeypatch.setattr("sciagent_runtime.llm.time.sleep", lambda _: None)

    def respond(*args, **kwargs):
        assert usage.used in (1, 2)
        assert list(usage.requests.values())[-1] == "unknown"
        if usage.used == 1:
            raise httpx.TransportError("disconnected")
        return _response()

    monkeypatch.setattr("sciagent_runtime.llm.send_request", respond)
    with execution_budget(max_llm_calls=4, timeout_seconds=30, usage=usage):
        assert invoke_model(client, "next_action", _context(), AgentAction) == {"tool": "finish"}
    assert client.last_attempts == usage.used == 2
    entries = list(usage.requests.items())
    assert entries[0][0][0] == entries[1][0][0]
    assert [key[1] for key, _ in entries] == [0, 1]
    assert [outcome for _, outcome in entries] == ["unknown", "succeeded"]


def test_slow_durable_charge_cannot_extend_http_deadline(monkeypatch):
    class SlowUsage(MemoryUsage):
        def charge(self, call_id, retry_index):
            super().charge(call_id, retry_index)
            time.sleep(0.03)

    client = _client(monkeypatch)
    usage = SlowUsage()
    monkeypatch.setattr("sciagent_runtime.llm.send_request", lambda *a, **k:
                        pytest.fail("HTTP started after its deadline"))
    with execution_budget(max_llm_calls=3, timeout_seconds=0.01, usage=usage):
        with pytest.raises(DeadlineExceededError):
            client.next_action(_context(), AgentAction)
    assert list(usage.requests.values()) == ["unknown"]
    assert client.last_attempts == 1


def test_no_retry_can_spend_beyond_run_balance(monkeypatch):
    client = _client(monkeypatch)
    sent = []
    monkeypatch.setattr("sciagent_runtime.llm.time.sleep", lambda _: None)

    def fail(*args, **kwargs):
        sent.append(True)
        raise httpx.TransportError("lost connection")

    monkeypatch.setattr("sciagent_runtime.llm.send_request", fail)
    with execution_budget(max_llm_calls=1, timeout_seconds=30) as budget:
        with pytest.raises(BudgetExhaustedError):
            client.next_action(_context(), AgentAction)
        assert budget.usage.used == len(sent) == client.last_attempts == 1


def test_retry_backoff_is_clipped_to_shared_deadline(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr("sciagent_runtime.llm.send_request", lambda *a, **k:
                        (_ for _ in ()).throw(httpx.TransportError("failed")))
    started = time.monotonic()
    with execution_budget(max_llm_calls=3, timeout_seconds=0.04) as budget:
        with pytest.raises(DeadlineExceededError):
            client.next_action(_context(), AgentAction)
        assert budget.usage.used == client.last_attempts == 1
    assert time.monotonic() - started < 0.3


def test_total_timeout_returns_without_waiting_for_cancelled_dns(monkeypatch):
    import socket

    resolving = Event()
    release = Event()
    finished = Event()
    original = socket.getaddrinfo

    def slow_lookup(*args, **kwargs):
        resolving.set()
        try:
            release.wait(1.5)
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setattr(socket, "getaddrinfo", slow_lookup)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            send_request(httpx.Request("GET", "http://localhost:9/"), timeout=0.2, max_response_bytes=None)
        assert resolving.is_set()
        assert not finished.is_set()
        assert time.monotonic() - started < 0.8
    finally:
        # The OS resolver cannot be interrupted; let this test's lookup finish.
        # Its cancelled future must never resume the HTTP request.
        release.set()
        assert finished.wait(1)


def test_total_timeout_cancels_slow_drip_http_body_and_closes_socket():
    disconnected = Event()
    started_body = Event()

    class SlowResponse(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            started_body.set()
            try:
                for _ in range(100):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowResponse)
    server.daemon_threads = True
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            send_request(httpx.Request(
                "POST", f"http://127.0.0.1:{server.server_port}/", content=b"{}",
            ), timeout=0.2, max_response_bytes=None)
        assert started_body.is_set()
        assert time.monotonic() - started < 0.8
        assert disconnected.wait(1), "cancelled HTTP request left its socket open"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)
