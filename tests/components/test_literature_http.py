"""HTTP policy tests use virtual time, never live sleeps or requests."""

from datetime import UTC, datetime, timedelta
import json
import gzip
import logging
from email.utils import format_datetime
import httpx

import pytest

from resagent2_components.literature import _http as http
from resagent2_components.literature import backends as literature
from resagent2_runtime import http as transport_http


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(http.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(http.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


def http_response(body=b"ok", *, status=200, headers=None, authenticated=False):
    request = httpx.Request(
        "GET", "https://example.test/?q=private-query&api_key=private-key",
        headers={"Authorization": "Bearer private-key"} if authenticated else {},
    )
    return httpx.Response(status, content=body, headers=headers, request=request)


def http_error(status, retry_after=None):
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    response = httpx.Response(status, headers=headers, request=httpx.Request("GET", "https://example.test/?q=x"))
    return httpx.HTTPStatusError("private response", request=response.request, response=response)


def test_arxiv_instances_share_spacing(clock, monkeypatch):
    monkeypatch.setattr(literature, "_ARXIV_HTTP", http.LiteratureHTTP("arXiv", interval_seconds=3))
    times = []
    def request(self, url):
        times.append(clock[0])
        return http_response(b'<feed xmlns="http://www.w3.org/2005/Atom"/>')
    monkeypatch.setattr(literature.ArxivLiteratureBackend, "_request", request)
    for _ in range(3):
        assert literature.ArxivLiteratureBackend().search("x", max_results=1).papers == []
    assert times == [0, 3, 6]


@pytest.mark.parametrize("retry_after,cooldown", [(None, 60), ("120", 120), ("bad", 60), ("-1", 60)])
def test_429_is_not_retried_and_cooldown_skips_network(clock, retry_after, cooldown):
    policy = http.LiteratureHTTP("test", interval_seconds=3)
    calls = []
    def request():
        calls.append(clock[0])
        if len(calls) == 1:
            raise http_error(429, retry_after)
        return http_response()
    with pytest.raises(http.LiteratureUnavailableError, match="429"):
        policy.fetch(request, max_attempts=3)
    assert calls == [0]
    assert clock[0] == 0  # No long blocking wait.
    clock[0] = cooldown - 1
    with pytest.raises(http.LiteratureUnavailableError, match="cooling down"):
        policy.fetch(request, max_attempts=3)
    assert calls == [0]
    clock[0] = cooldown
    assert policy.fetch(request, max_attempts=3) == b"ok"
    assert calls == [0, cooldown]


def test_406_switches_sources_without_retry_or_cooldown(clock):
    policy = http.LiteratureHTTP("test", interval_seconds=3)
    calls = []
    def request():
        calls.append(clock[0])
        raise http_error(406)
    with pytest.raises(http.LiteratureUnavailableError, match="HTTP 406") as caught:
        policy.fetch(request, max_attempts=3)
    assert calls == [0]
    assert clock[0] == 0
    assert policy._cooldown_until == 0
    assert "private" not in str(caught.value)
    assert "example.test" not in str(caught.value)


def test_retry_after_supports_http_date():
    future = datetime.now(UTC) + timedelta(seconds=180)
    assert 178 <= http._retry_after(format_datetime(future, usegmt=True)) <= 180
    assert http._retry_after("yesterday") == 0


def test_503_with_retry_after_defers_instead_of_waiting(clock):
    policy = http.LiteratureHTTP("test", interval_seconds=3)
    def request():
        raise http_error(503, "3600")
    with pytest.raises(http.LiteratureUnavailableError, match="Retry-After") as caught:
        policy.fetch(request, max_attempts=3)
    assert caught.value.error_type == "service_error"
    assert caught.value.retry_after == 3600
    assert clock[0] == 0
    assert policy._cooldown_until == 3600


@pytest.mark.parametrize("failure,error_type", [(TimeoutError(), "timeout"), (httpx.ConnectError("offline"), "network_error"), (ConnectionResetError(), "network_error"), (http_error(503), "service_error"), (http_error(408), "timeout")])
def test_transient_failures_have_finite_spaced_attempts(clock, failure, error_type):
    policy = http.LiteratureHTTP("test", interval_seconds=3)
    times = []
    def request():
        times.append(clock[0])
        raise failure
    with pytest.raises(http.LiteratureUnavailableError, match="after 3 attempts") as caught:
        policy.fetch(request, max_attempts=3)
    assert caught.value.error_type == error_type
    assert caught.value.retry_after == 60
    assert times == [0, 3, 9]


def test_transient_failure_can_recover(clock):
    policy = http.LiteratureHTTP("test", interval_seconds=3)
    calls = []
    def request():
        calls.append(clock[0])
        if len(calls) < 2:
            raise TimeoutError()
        return http_response()
    assert policy.fetch(request, max_attempts=3) == b"ok"
    assert calls == [0, 3]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_request_errors_are_not_retried_or_classified_as_unavailable(clock, status):
    policy = http.LiteratureHTTP("test", interval_seconds=3)
    calls = []
    def request():
        calls.append(clock[0])
        raise http_error(status)
    with pytest.raises(http.LiteratureSearchError, match=f"HTTP {status}") as caught:
        policy.fetch(request, max_attempts=3)
    assert not isinstance(caught.value, http.LiteratureUnavailableError)
    assert "private" not in str(caught.value)
    assert calls == [0]


def test_programming_errors_are_not_swallowed(clock):
    def request():
        raise RuntimeError("bug")
    with pytest.raises(RuntimeError, match="bug"):
        http.LiteratureHTTP("test", interval_seconds=3).fetch(request, max_attempts=3)


def response_records(caplog):
    return [
        json.loads(record.getMessage().split("Literature HTTP response ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("Literature HTTP response ")
    ]


@pytest.mark.parametrize("status", [200, 406, 429, 503])
def test_response_diagnostics_keep_only_safe_facts(clock, caplog, status):
    caplog.set_level(logging.INFO, logger=http.__name__)
    response = http_response(
        b'{"message": "private-body private-key"}', status=status,
        authenticated=True, headers={
            "Retry-After": "120",
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "0.25",
            "X-RateLimit-Credits-Used": "0.01",
            "X-RateLimit-Reset": "3600",
            "X-Secret": "private-header",
            "Set-Cookie": "private-cookie",
        },
    )
    def request():
        response.raise_for_status()
        return response
    policy = http.LiteratureHTTP("OpenAlex", interval_seconds=1)
    if status == 200:
        assert policy.fetch(request, max_attempts=3) == response.content
    else:
        with pytest.raises(http.LiteratureUnavailableError):
            policy.fetch(request, max_attempts=3)
    [record] = response_records(caplog)
    assert record["source"] == "OpenAlex"
    assert record["status"] == status
    assert record["attempt"] == 1
    assert record["max_attempts"] == 3
    assert record["authentication_configured"] is True
    assert record["retry_after_present"] is True
    assert record["retry_after_seconds"] == 120
    assert record["cooldown_seconds"] == (120 if status in {429, 503} else 0)
    assert record["quota"] == {
        "X-RateLimit-Limit": 100, "X-RateLimit-Remaining": 0.25,
        "X-RateLimit-Credits-Used": 0.01, "X-RateLimit-Reset": 3600,
    }
    assert policy._cooldown_until == record["cooldown_seconds"]
    for private in ("private-key", "private-query", "private-body", "private-header",
                    "private-cookie", "example.test", "Authorization"):
        assert private not in caplog.text
    expected_level = logging.INFO if status == 200 else logging.WARNING
    assert caplog.records[-1].levelno == expected_level


@pytest.mark.parametrize("raw", ["private-key", "NaN", "inf", "-1", "1" * 100])
def test_diagnostics_discard_invalid_quota_values(clock, caplog, raw):
    caplog.set_level(logging.INFO, logger=http.__name__)
    response = http_response(headers={
        "X-RateLimit-Remaining": raw, "Retry-After": "private-key",
    })
    assert http.LiteratureHTTP("OpenAlex", interval_seconds=1).fetch(
        lambda: response, max_attempts=1,
    ) == b"ok"
    [record] = response_records(caplog)
    assert record["quota"] == {}
    assert record["authentication_configured"] is False
    assert record["retry_after_present"] is True
    assert record["retry_after_seconds"] == 0
    assert "private-key" not in caplog.text


def test_diagnostics_preserve_http_attempt_numbers(clock, caplog):
    caplog.set_level(logging.INFO, logger=http.__name__)
    responses = iter([http_response(status=503), http_response()])
    def request():
        response = next(responses)
        response.raise_for_status()
        return response
    assert http.LiteratureHTTP("test", interval_seconds=1).fetch(
        request, max_attempts=3,
    ) == b"ok"
    records = response_records(caplog)
    assert [(r["status"], r["attempt"], r["cooldown_seconds"]) for r in records] == [
        (503, 1, 0), (200, 2, 0),
    ]
    assert clock[0] == 3


def test_http_date_retry_after_is_logged_as_seconds(clock, caplog):
    until = datetime.now(UTC) + timedelta(seconds=180)
    response = http_response(
        status=429, headers={"Retry-After": format_datetime(until, usegmt=True)},
    )
    policy = http.LiteratureHTTP("OpenAlex", interval_seconds=1)
    def request():
        response.raise_for_status()
        return response
    with pytest.raises(http.LiteratureUnavailableError):
        policy.fetch(request, max_attempts=3)
    [record] = response_records(caplog)
    assert 178 <= record["retry_after_seconds"] <= 180
    assert record["cooldown_seconds"] == record["retry_after_seconds"]
    assert policy._cooldown_until == record["cooldown_seconds"]
    with pytest.raises(http.LiteratureUnavailableError, match="cooling down"):
        policy.fetch(request, max_attempts=3)
    assert response_records(caplog) == [record]  # Skipping HTTP is not another response.


def test_exhausted_transient_attempt_logs_final_cooldown(clock, caplog):
    response = http_response(status=503)
    policy = http.LiteratureHTTP("OpenAlex", interval_seconds=1)
    def request():
        response.raise_for_status()
        return response
    with pytest.raises(http.LiteratureUnavailableError, match="after 3 attempts"):
        policy.fetch(request, max_attempts=3)
    records = response_records(caplog)
    assert [(r["attempt"], r["cooldown_seconds"]) for r in records] == [
        (1, 0), (2, 0), (3, 60),
    ]
    assert policy._cooldown_until - clock[0] == 60



class MetadataBody(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content
        self.closed = False

    async def __aiter__(self):
        for offset in range(0, len(self.content), 65536):
            yield self.content[offset:offset + 65536]

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("provider", ["arxiv", "openalex"])
@pytest.mark.parametrize("operation", ["search", "resolve"])
@pytest.mark.parametrize("extra_bytes", [-1, 0, 1])
@pytest.mark.parametrize("compressed", [False, True], ids=["identity", "gzip"])
def test_metadata_public_entry_uses_shared_decoded_limit(
    monkeypatch, provider, operation, extra_bytes, compressed,
):
    # Use valid complete metadata at the real policy boundary, not a transport stub.
    limit = http.METADATA_RESPONSE_MAX_BYTES
    if provider == "arxiv":
        payload = (b'<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
                   b'<id>https://arxiv.org/abs/2301.00001v2</id><title>Evidence</title>'
                   b'<summary>PAD</summary></entry></feed>')
        backend = literature.ArxivLiteratureBackend()
        identity, attribute, source_name = "arxiv:2301.00001v2", "_ARXIV_HTTP", "arXiv"
    else:
        record = {"id": "https://openalex.org/W123", "display_name": "Evidence",
                  "abstract_inverted_index": {"PAD": [0]}}
        payload = json.dumps({"results": [record]} if operation == "search" else record).encode()
        backend = literature.OpenAlexLiteratureBackend(api_key="private-key")
        identity, attribute, source_name = "openalex:W123", "_OPENALEX_HTTP", "OpenAlex"
    abstract_bytes = limit - len(payload) + len(b"PAD") + extra_bytes
    payload = payload.replace(b"PAD", b"x" * abstract_bytes)
    assert len(payload) == limit + extra_bytes
    body = MetadataBody(gzip.compress(payload) if compressed else payload)
    policy = http.LiteratureHTTP(source_name, interval_seconds=0)
    monkeypatch.setattr(literature, attribute, policy)
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, stream=body, headers=(
            {"Content-Encoding": "gzip"} if compressed else {}
        ))  # No Content-Length: enforce the streamed decoded byte count.

    client = httpx.AsyncClient
    monkeypatch.setattr(transport_http.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))

    def invoke():
        if operation == "search":
            return backend.search("private-query", max_results=1).papers[0]
        return backend.resolve(identity)

    if extra_bytes > 0:
        with pytest.raises(http.LiteratureSearchError) as caught:
            invoke()
        error = caught.value
        assert error.error_type == "response_too_large"
        assert not isinstance(error, http.LiteratureUnavailableError)
        assert error.source_attempts[0]["source"] == provider
        assert error.source_attempts[0]["error_type"] == "response_too_large"
        assert error.retry_after is None
        assert "private" not in str(error)
        assert "invalid result count" not in str(error)
    else:
        paper = invoke()
        assert paper.title == "Evidence"
        assert paper.abstract == "x" * abstract_bytes
    assert len(calls) == 1  # Oversized responses are not retried.
    assert policy._cooldown_until == 0
    assert body.closed


def test_metadata_decode_failure_is_domain_error_without_retry_or_cooldown(clock):
    policy = http.LiteratureHTTP("test", interval_seconds=3)
    calls = []

    def request():
        calls.append(clock[0])
        raise httpx.DecodingError("private-body private-query private-key")

    with pytest.raises(http.LiteratureSearchError) as caught:
        policy.fetch(request, max_attempts=3)
    assert not isinstance(caught.value, http.LiteratureUnavailableError)
    assert caught.value.error_type == "invalid_response"
    assert calls == [0]
    assert policy._cooldown_until == 0
    assert "private" not in str(caught.value)
