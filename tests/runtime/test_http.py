"""Bounded HTTP bodies retain the shared deadline and default transport behavior."""

import asyncio
import gzip
import socket

import httpx
import pytest

from sciagent_runtime import http
from sciagent_runtime.budget import DeadlineExceededError, execution_budget


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks, *, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.read_count = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.read_count += 1
            yield chunk

    async def aclose(self):
        self.closed = True


def transport(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        http.httpx, "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_bounded_body_matches_explicit_unlimited_response(monkeypatch):
    transport(monkeypatch, lambda request: httpx.Response(
        200, stream=Body([b"a", b"bc"]), headers={"x-source": "test"},
    ))
    request = httpx.Request("GET", "https://example.test/paper.pdf")
    bounded = http.send_request(request, timeout=1, max_response_bytes=3)
    unlimited = http.send_request(request, timeout=1, max_response_bytes=None)
    assert bounded.content == unlimited.content == b"abc"
    assert bounded.headers["x-source"] == "test"
    assert bounded.request.url == request.url


def test_byte_limit_stops_reading_and_closes_response(monkeypatch):
    body = Body([b"a" * 65536] * 4)
    transport(monkeypatch, lambda request: httpx.Response(200, stream=body))
    with pytest.raises(http.ResponseTooLargeError, match="65536 decoded bytes"):
        http.send_request(
            httpx.Request("GET", "https://example.test/large"),
            timeout=1, max_response_bytes=65536,
        )
    assert body.read_count == 2
    assert body.closed


def test_limit_counts_decoded_gzip_bytes(monkeypatch):
    content = b"x" * 100000
    compressed = gzip.compress(content)
    assert len(compressed) < 1000
    transport(monkeypatch, lambda request: httpx.Response(
        200, stream=Body([compressed]),
        headers={"content-encoding": "gzip", "content-length": str(len(compressed))},
    ))
    request = httpx.Request("GET", "https://example.test/compressed")
    with pytest.raises(http.ResponseTooLargeError, match="1000 decoded bytes"):
        http.send_request(request, timeout=1, max_response_bytes=1000)
    response = http.send_request(request, timeout=1, max_response_bytes=len(content))
    assert response.content == content
    assert response.headers["content-length"] == str(len(content))
    assert "content-encoding" not in response.headers


def test_redirects_are_opt_in_and_bounded(monkeypatch):
    calls = []

    def redirect(request):
        calls.append(request.url)
        return httpx.Response(302, headers={"location": "/again"}, stream=Body([]))

    transport(monkeypatch, redirect)
    request = httpx.Request("GET", "https://example.test/start")
    with pytest.raises(httpx.HTTPStatusError):
        http.send_request(request, timeout=1, max_response_bytes=10)
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(httpx.TooManyRedirects):
        http.send_request(request, timeout=1, max_response_bytes=10, follow_redirects=True)
    assert len(calls) == 6


def test_bounded_body_honors_total_timeout_and_closes(monkeypatch):
    body = Body([b"x"], delay=0.1)
    transport(monkeypatch, lambda request: httpx.Response(200, stream=body))
    with pytest.raises(TimeoutError):
        http.send_request(
            httpx.Request("GET", "https://example.test/slow"),
            timeout=0.01, max_response_bytes=10,
        )
    assert body.closed


def test_redirect_body_is_discarded_without_unbounded_read(monkeypatch):
    redirect_body = Body([b"x" * 100000], delay=10)

    def respond(request):
        if request.url.path == "/first":
            return httpx.Response(302, headers={"location": "/paper"}, stream=redirect_body)
        return httpx.Response(200, stream=Body([b"pdf"]))

    transport(monkeypatch, respond)
    response = http.send_request(
        httpx.Request("GET", "https://example.test/first"),
        timeout=1, max_response_bytes=3, follow_redirects=True,
    )
    assert response.content == b"pdf"
    assert response.url.path == "/paper"
    assert len(response.history) == 1
    assert redirect_body.closed
    assert redirect_body.read_count == 0


def test_bounded_body_cannot_start_after_run_deadline(monkeypatch):
    transport(monkeypatch, lambda request: pytest.fail("HTTP should not start"))
    with execution_budget(max_llm_calls=1, timeout_seconds=0):
        with pytest.raises(DeadlineExceededError):
            http.send_request(
                httpx.Request("GET", "https://example.test/late"),
                timeout=100, max_response_bytes=10,
            )


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "10"])
def test_invalid_byte_limit_does_not_start_http(monkeypatch, limit):
    transport(monkeypatch, lambda request: pytest.fail("HTTP should not start"))
    with pytest.raises(ValueError, match="must be positive"):
        http.send_request(
            httpx.Request("GET", "https://example.test/"),
            timeout=1, max_response_bytes=limit,
        )


def test_response_byte_policy_must_be_explicit(monkeypatch):
    transport(monkeypatch, lambda request: pytest.fail("HTTP should not start"))
    with pytest.raises(TypeError, match="max_response_bytes"):
        http.send_request(httpx.Request("GET", "https://example.test/"), timeout=1)


def public_dns(monkeypatch, addresses):
    calls = []

    async def resolve(loop, host, port, **kwargs):
        calls.append((host, port))
        return [
            (socket.AF_INET6 if ":" in address else socket.AF_INET,
             socket.SOCK_STREAM, 6, "", (address, port))
            for address in addresses
        ]

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolve)
    return calls


@pytest.mark.parametrize("address", ["93.184.216.34", "2606:4700:4700::1111"])
def test_public_request_pins_ip_and_preserves_host_sni_and_logical_url(monkeypatch, address):
    dns_calls = public_dns(monkeypatch, [address])
    outgoing = []
    client_options = []
    client = httpx.AsyncClient

    def respond(request):
        outgoing.append((request.url.host, request.url.port, request.headers["Host"],
                         request.extensions["sni_hostname"]))
        return httpx.Response(200, stream=Body([b"body"]))

    def make_client(**kwargs):
        client_options.append(kwargs)
        return client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(http.httpx, "AsyncClient", make_client)
    request = httpx.Request("GET", "https://example.test:8443/page")
    response = http.send_request(request, timeout=1, max_response_bytes=4, public_only=True)

    assert outgoing == [(address, 8443, "example.test:8443", "example.test")]
    assert dns_calls == [("example.test", 8443)]
    assert response.url == request.url == httpx.URL("https://example.test:8443/page")
    assert response.content == b"body"
    assert client_options[0]["trust_env"] is False
    assert client_options[0]["limits"].max_keepalive_connections == 0


@pytest.mark.parametrize("host", [
    "127.0.0.1", "10.0.0.8", "169.254.169.254", "[::1]", "[fe80::1]",
    "[::ffff:127.0.0.1]", "localhost", "foo.localhost", "localhost.",
])
def test_public_request_rejects_nonpublic_hosts_without_sending(monkeypatch, host):
    dns_calls = public_dns(monkeypatch, ["93.184.216.34"])
    transport(monkeypatch, lambda request: pytest.fail("nonpublic HTTP must not start"))
    with pytest.raises(http.NonPublicAddressError):
        http.send_request(httpx.Request("GET", f"http://{host}/"), timeout=1, public_only=True, max_response_bytes=None)
    assert dns_calls == []


@pytest.mark.parametrize("addresses", [
    [], ["93.184.216.34", "10.0.0.8"], ["2606:4700:4700::1111", "::1"],
])
def test_public_request_rejects_empty_or_mixed_dns_before_sending(monkeypatch, addresses):
    public_dns(monkeypatch, addresses)
    transport(monkeypatch, lambda request: pytest.fail("nonpublic HTTP must not start"))
    with pytest.raises(http.NonPublicAddressError):
        http.send_request(httpx.Request("GET", "https://example.test/"), timeout=1, public_only=True, max_response_bytes=None)


def test_public_request_normalizes_dns_failure(monkeypatch):
    async def resolve(loop, *args, **kwargs):
        raise socket.gaierror("unavailable")

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolve)
    transport(monkeypatch, lambda request: pytest.fail("unresolved HTTP must not start"))
    with pytest.raises(http.NonPublicAddressError, match="could not be resolved"):
        http.send_request(httpx.Request("GET", "https://example.test/"), timeout=1, public_only=True, max_response_bytes=None)


@pytest.mark.parametrize("run_limited", [False, True])
def test_public_dns_is_cancelled_at_effective_deadline(monkeypatch, run_limited):
    cancelled = []

    async def resolve(loop, *args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolve)
    transport(monkeypatch, lambda request: pytest.fail("HTTP must not start before DNS finishes"))
    with execution_budget(max_llm_calls=1, timeout_seconds=0.01 if run_limited else 10):
        with pytest.raises(TimeoutError):
            http.send_request(
                httpx.Request("GET", "https://example.test/"),
                timeout=10 if run_limited else 0.01,
                max_response_bytes=None, public_only=True,
            )
    assert cancelled == [True]


def test_public_redirects_keep_logical_origin_and_recheck_destinations(monkeypatch):
    dns_calls = public_dns(monkeypatch, ["93.184.216.34"])
    calls = []

    def respond(request):
        calls.append((request.url.host, request.url.path, request.headers["Host"]))
        target = "/second" if request.url.path == "/first" else "http://127.0.0.1/private"
        return httpx.Response(302, headers={"location": target}, stream=Body([]))

    transport(monkeypatch, respond)
    with pytest.raises(http.NonPublicAddressError):
        http.send_request(
            httpx.Request("GET", "https://example.test/first"),
            timeout=1, max_response_bytes=10, follow_redirects=True, public_only=True,
        )
    assert calls == [
        ("93.184.216.34", "/first", "example.test"),
        ("93.184.216.34", "/second", "example.test"),
    ]
    assert dns_calls == [("example.test", 443), ("example.test", 443)]


def test_public_request_retains_byte_limit(monkeypatch):
    public_dns(monkeypatch, ["93.184.216.34"])
    body = Body([b"too large"])
    transport(monkeypatch, lambda request: httpx.Response(200, stream=body))
    with pytest.raises(http.ResponseTooLargeError):
        http.send_request(
            httpx.Request("GET", "https://example.test/"),
            timeout=1, max_response_bytes=3, public_only=True,
        )
    assert body.closed


def test_public_request_restores_logical_url_on_transport_error(monkeypatch):
    public_dns(monkeypatch, ["93.184.216.34"])

    def fail(request):
        raise httpx.ConnectError("failed", request=request)

    transport(monkeypatch, fail)
    request = httpx.Request("GET", "https://example.test/")
    with pytest.raises(httpx.ConnectError) as raised:
        http.send_request(request, timeout=1, public_only=True, max_response_bytes=None)
    assert raised.value.request.url == request.url == httpx.URL("https://example.test/")
