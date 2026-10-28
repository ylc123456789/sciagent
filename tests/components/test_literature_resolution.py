"""Provider lookup uses exact identities, preserving arXiv versions and errors."""

import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from sciagent_components.literature import (
    ArxivLiteratureBackend, OpenAlexLiteratureBackend, MultiSourceLiteratureBackend,
    LiteratureSearchError, parse_literature_source,
)
from tests.capabilities.test_literature import ARXIV_ATOM
from tests.components.test_openalex import work


@pytest.fixture(autouse=True)
def isolated_http(monkeypatch):
    from sciagent_components.literature import _http, backends
    for attribute, name in (("_ARXIV_HTTP", "arXiv"), ("_OPENALEX_HTTP", "OpenAlex")):
        monkeypatch.setattr(backends, attribute, _http.LiteratureHTTP(name, interval_seconds=0))


@pytest.mark.parametrize("source,expected", [
    ("arxiv:2301.00001v2", ("arxiv", "2301.00001v2")),
    ("arxiv:hep-th/9901001v3", ("arxiv", "hep-th/9901001v3")),
    ("doi:10.1234/ABC", ("doi", "10.1234/abc")),
    ("openalex:W123", ("openalex", "W123")),
    ("artifact:artifact_abc", ("artifact", "artifact_abc")),
])
def test_explicit_source_parser(source, expected):
    assert parse_literature_source(source) == expected


def _atom(identity="2301.00001v2"):
    prefix = ARXIV_ATOM.split("  <entry>", 1)[0]
    entry = ARXIV_ATOM.split("  <entry>", 1)[1].split("  </entry>", 1)[0]
    return (prefix + "  <entry>" + entry.replace("2301.00001v2", identity) + "  </entry>\n</feed>").encode()


@pytest.mark.parametrize("requested", ["2301.00001v2", "2301.00001"])
def test_arxiv_resolves_one_id_and_reports_actual_version(monkeypatch, requested):
    backend, calls = ArxivLiteratureBackend(), []
    def request(url):
        calls.append(url)
        return httpx.Response(200, content=_atom(), request=httpx.Request("GET", url))
    monkeypatch.setattr(backend, "_request", request)
    paper = backend.resolve(f"arxiv:{requested}")
    assert paper.paper_id == "2301.00001v2" and paper.key == "arxiv:2301.00001v2"
    assert parse_qs(urlsplit(calls[0]).query) == {"id_list": [requested]}
    assert "search_query" not in calls[0]


@pytest.mark.parametrize("returned", ["2301.00001v1", "2301.00002v2", "2301.00001"])
def test_arxiv_refuses_wrong_or_missing_explicit_version(monkeypatch, returned):
    backend = ArxivLiteratureBackend()
    monkeypatch.setattr(backend, "_request", lambda url: httpx.Response(
        200, content=_atom(returned), request=httpx.Request("GET", url)))
    with pytest.raises(LiteratureSearchError) as caught:
        backend.resolve("arxiv:2301.00001v2")
    assert caught.value.error_type == "identity_mismatch"
    assert caught.value.source_attempts[0]["source"] == "arxiv"


def test_arxiv_multiple_entries_are_not_a_resolved_paper(monkeypatch):
    backend = ArxivLiteratureBackend()
    monkeypatch.setattr(backend, "_request", lambda url: httpx.Response(
        200, content=ARXIV_ATOM.encode(), request=httpx.Request("GET", url)))
    with pytest.raises(LiteratureSearchError) as caught:
        backend.resolve("arxiv:2301.00001v2")
    assert caught.value.error_type == "not_found"


@pytest.mark.parametrize("source", ["openalex:W123", "doi:10.1234/example"])
def test_openalex_exact_lookup_preserves_auth_headers(monkeypatch, source):
    from sciagent_components.literature import backends
    record = {**work(), "doi": "https://doi.org/10.1234/example"}
    requests = []
    def request(request, *, timeout, max_response_bytes):
        assert max_response_bytes == backends.METADATA_RESPONSE_MAX_BYTES
        requests.append(request)
        return httpx.Response(200, content=json.dumps(record).encode(), request=request)
    monkeypatch.setattr(backends, "send_request", request)
    paper = OpenAlexLiteratureBackend(api_key="secret").resolve(source)
    request, = requests
    assert request.headers["Authorization"] == "Bearer secret"
    assert "secret" not in str(request.url) and "search" not in str(request.url)
    assert paper.paper_id == "openalex:W123"
    assert str(request.url).endswith("/W123" if source.startswith("openalex:") else "/https://doi.org/10.1234/example")


@pytest.mark.parametrize("source", ["openalex:W124", "doi:10.1234/wrong"])
def test_openalex_wrong_identity_does_not_silently_substitute(monkeypatch, source):
    backend = OpenAlexLiteratureBackend()
    record = {**work(), "doi": "https://doi.org/10.1234/example"}
    monkeypatch.setattr(backend, "_request", lambda url: httpx.Response(
        200, content=json.dumps(record).encode(), request=httpx.Request("GET", url)))
    with pytest.raises(LiteratureSearchError) as caught:
        backend.resolve(source)
    assert caught.value.error_type == "identity_mismatch"


def test_openalex_fetch_source_preserves_exact_record_when_doi_is_also_available():
    record = {**work(), "doi": "https://doi.org/10.1234/example"}
    paper = OpenAlexLiteratureBackend()._paper(record)
    assert paper.key == "doi:10.1234/example"
    assert paper.fetch_source == "openalex:W123"


def test_arxiv_search_rejects_unfetchable_provider_identifier(monkeypatch):
    backend = ArxivLiteratureBackend()
    monkeypatch.setattr(backend, "_request", lambda url: httpx.Response(
        200, content=_atom("not-an-id"), request=httpx.Request("GET", url)))
    with pytest.raises(LiteratureSearchError, match="invalid paper"):
        backend.search("paper", max_results=1)


def test_multi_source_dispatches_identity_without_fuzzy_failover():
    calls = []
    class Backend:
        def __init__(self, name):
            self.name = name
        def resolve(self, source):
            calls.append((self.name, source))
            raise LiteratureSearchError("not found", error_type="not_found")
    backend = MultiSourceLiteratureBackend(Backend("arxiv"), Backend("openalex"))
    with pytest.raises(LiteratureSearchError):
        backend.resolve("doi:10.1234/example")
    assert calls == [("openalex", "doi:10.1234/example")]
