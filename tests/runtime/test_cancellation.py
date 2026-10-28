"""Cancellation is scoped, race-safe and closes actual asynchronous resources."""
import asyncio
from threading import Event, Thread
import time

import httpx
import pytest

from sciagent_runtime import http
from sciagent_runtime.cancellation import (
    CancellationToken, OperationCancelled, bind_cancellation, check_cancelled,
    current_cancellation,
)


def test_scope_and_cancel_before_callback_registration():
    outer, inner = CancellationToken(), CancellationToken()
    called = []
    assert current_cancellation() is None
    with bind_cancellation(outer):
        with bind_cancellation(inner):
            inner.cancel("new upstream input")
            unregister = inner.on_cancel(lambda: called.append(True))
            unregister()
            inner.cancel("second cancellation")
            with pytest.raises(OperationCancelled, match="new upstream input"):
                check_cancelled()
        assert current_cancellation() is outer
        check_cancelled()
    assert current_cancellation() is None
    assert called == [True]


@pytest.mark.parametrize("stage", ["dns", "body"])
def test_interrupt_http_stops_wait_and_closes_response(monkeypatch, stage):
    token = CancellationToken()
    started = Event()
    closed = []

    async def resolve(loop, *args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.append("dns")

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"never"

        async def aclose(self):
            closed.append("body")

    client = httpx.AsyncClient
    monkeypatch.setattr(http.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Body())),
        **kwargs,
    ))
    if stage == "dns":
        monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolve)

    def interrupt():
        assert started.wait(2)
        token.cancel("pause requested")

    worker = Thread(target=interrupt)
    worker.start()
    begin = time.monotonic()
    try:
        with bind_cancellation(token), pytest.raises(OperationCancelled) as caught:
            http.send_request(
                httpx.Request("GET", "https://example.test/"), timeout=30,
                max_response_bytes=100, public_only=stage == "dns",
            )
        assert caught.value.stopped
        assert time.monotonic() - begin < 3
        assert closed == [stage]
    finally:
        worker.join(3)


def test_cancelled_http_never_dispatches(monkeypatch):
    token = CancellationToken()
    token.cancel()
    monkeypatch.setattr(http.httpx, "AsyncClient", lambda **kwargs: pytest.fail("dispatched"))
    with bind_cancellation(token), pytest.raises(OperationCancelled):
        http.send_request(httpx.Request("GET", "https://example.test/"), timeout=1, max_response_bytes=None)


def test_native_model_cancel_keeps_charge_and_explicit_trace(monkeypatch, tmp_path):
    import json
    from sciagent_runtime import ComposedContext, OpenAICompatibleClient, llm
    from sciagent_runtime.budget import execution_budget

    token = CancellationToken()
    monkeypatch.setenv("CANCELLATION_TEST_KEY", "dummy")

    def interrupt(*args, **kwargs):
        token.cancel("pause model request")
        check_cancelled()

    monkeypatch.setattr(llm, "send_request", interrupt)
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.test/v1", api_key_env="CANCELLATION_TEST_KEY",
        trace_dir=tmp_path, trace_level="full",
    )
    with execution_budget(max_llm_calls=3, timeout_seconds=10) as budget:
        with bind_cancellation(token), pytest.raises(OperationCancelled):
            client.next_tool_call(
                ComposedContext(text="test", included_sections=[], estimated_tokens=0),
                [], [], max_input_tokens=10_000,
            )
        assert budget.usage.used == client.last_attempts == 1
        assert list(budget.usage.requests.values()) == ["unknown"]
    records = [json.loads(line) for path in tmp_path.rglob("*.jsonl")
               for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["validation_error"] == "pause model request"
    assert records[0]["retry_number"] == 0
    assert len(records[0]["attempts"]) == 1
    assert records[0]["action_valid"] is False


def test_retry_wait_is_interrupted_without_waiting_for_its_timeout():
    token = CancellationToken()
    worker = Thread(target=lambda: (time.sleep(0.05), token.cancel("interrupt retry wait")))
    worker.start()
    begin = time.monotonic()
    with pytest.raises(OperationCancelled, match="interrupt retry wait"):
        token.wait(30)
    worker.join(2)
    assert time.monotonic() - begin < 1


@pytest.mark.parametrize("waiting_on", ["source_lock", "request_interval"])
def test_literature_local_wait_is_interruptible(waiting_on):
    from sciagent_components.literature._http import LiteratureHTTP

    policy = LiteratureHTTP("test", interval_seconds=30)
    token = CancellationToken()
    if waiting_on == "source_lock":
        policy._lock.acquire()
    else:
        policy._next_request_at = time.monotonic() + 30
    worker = Thread(target=lambda: (time.sleep(0.05), token.cancel("interrupt literature wait")))
    worker.start()
    begin = time.monotonic()
    try:
        with bind_cancellation(token), pytest.raises(OperationCancelled):
            policy.fetch(lambda: pytest.fail("request started"), max_attempts=1)
        assert time.monotonic() - begin < 1
    finally:
        if waiting_on == "source_lock":
            policy._lock.release()
        worker.join(2)
