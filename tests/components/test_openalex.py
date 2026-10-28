import httpx
import json
from datetime import date
from urllib.parse import parse_qs, urlsplit

import pytest

from resagent2_components import (
    MultiSourceLiteratureBackend,
    LiteraturePaper,
    LiteratureSearchError,
    LiteratureUnavailableError,
    OpenAlexLiteratureBackend,
)
from resagent2_capabilities import (
    LiteratureSearchTool,
    LiteratureSearchToolInput,
)
from resagent2_components.literature import (
    _http as _literature_http,
    backends as openalex,
)
from tests.capabilities.test_literature import _FakeRegister, state
from resagent2_components import LiteratureSearchResult


class _FakeBackend:
    def __init__(self, papers, name="peer"):
        self.papers = papers
        self.name = name
        self.last_kwargs = None

    def search(self, query, **kwargs):
        self.last_kwargs = dict(query=query, **kwargs)
        return LiteratureSearchResult(self.papers, self.name, query, kwargs["page"], None)


@pytest.fixture(autouse=True)
def isolated_http(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(_literature_http.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(_literature_http.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(openalex, "_OPENALEX_HTTP", _literature_http.LiteratureHTTP("OpenAlex", interval_seconds=1))


def work():
    return {
        "id": "https://openalex.org/W123",
        "display_name": "A real-shaped record",
        "publication_date": "2020-01-02",
        "authorships": [{"author": {"display_name": "Alice"}}],
        "abstract_inverted_index": {"evidence.": [2], "Retrieved": [0], "abstract": [1]},
    }


def fake_response(monkeypatch, body):
    requests = []
    def open_request(request, *, timeout, max_response_bytes):
        assert max_response_bytes == _literature_http.METADATA_RESPONSE_MAX_BYTES
        requests.append((request, timeout))
        return httpx.Response(200, content=body, request=request)
    monkeypatch.setattr(openalex, "send_request", open_request)
    return requests


def test_query_auth_and_normalized_record(monkeypatch):
    requests = fake_response(monkeypatch, json.dumps({"results": [work(), work()]}).encode())
    backend = OpenAlexLiteratureBackend(api_key="private-test-key", timeout_seconds=9)
    papers = backend.search("learning rate", max_results=5, start_year=2019, end_year=2024).papers
    assert len(papers) == 1
    assert papers[0] == LiteraturePaper(
        paper_id="openalex:W123", title="A real-shaped record", authors=["Alice"],
        published_at=date(2020, 1, 2), abstract="Retrieved abstract evidence.",
        source_url="https://openalex.org/W123",
    )
    request, timeout = requests[0]
    params = parse_qs(urlsplit(str(request.url)).query)
    assert params["search.title_and_abstract"] == ["learning rate"]
    assert params["per_page"] == ["5"]
    assert params["filter"] == ["from_publication_date:2019-01-01,to_publication_date:2024-12-31"]
    assert request.headers.get("Authorization") == "Bearer private-test-key"
    assert request.headers.get("User-agent").startswith("ResAgent2/")
    assert "private-test-key" not in str(request.url)
    assert "private-test-key" not in papers[0].model_dump_json()
    assert timeout == 9


def test_optional_metadata_and_key_can_be_absent(monkeypatch):
    record = work()
    record.update(publication_date=None, authorships=[], abstract_inverted_index=None)
    requests = fake_response(monkeypatch, json.dumps({"results": [record]}).encode())
    paper = OpenAlexLiteratureBackend().search("x", max_results=1).papers[0]
    assert paper.abstract == ""
    assert paper.authors == []
    assert paper.published_at is None
    assert requests[0][0].headers.get("Authorization") is None
    assert "filter" not in parse_qs(urlsplit(str(requests[0][0].url)).query)


def test_complete_abstract_is_preserved():
    backend = OpenAlexLiteratureBackend()
    record = work()
    record["abstract_inverted_index"] = {"longword": list(range(1000))}
    assert len(backend._paper(record).abstract) > 2000


@pytest.mark.parametrize("body", [
    b"not json", b"[]", b"{}", b'{"results": null}', b'{"results": [{}]}',
    b'{"results": [null]}',
])
def test_invalid_response_is_not_empty_success(monkeypatch, body):
    fake_response(monkeypatch, body)
    with pytest.raises(LiteratureSearchError, match="invalid bibliographic data"):
        OpenAlexLiteratureBackend().search("x", max_results=1)


@pytest.mark.parametrize("index", [[], {"word": "1"}, {"word": [-1]}, {"word": [True]}, {"a": [0], "b": [0]}])
def test_invalid_abstract_does_not_fabricate_evidence(index):
    record = work()
    record["abstract_inverted_index"] = index
    with pytest.raises(LiteratureSearchError):
        OpenAlexLiteratureBackend()._parse(json.dumps({"results": [record]}).encode())


def test_valid_empty_result(monkeypatch):
    fake_response(monkeypatch, b'{"results":[]}')
    assert OpenAlexLiteratureBackend().search("x", max_results=1).papers == []


class Failing:
    def __init__(self, error, name="arxiv"):
        self.name = name
        self.error = error
    def search(self, query, **kwargs):
        raise self.error


def test_source_switch_preserves_bounds_and_actual_source_in_receipt(caplog, tmp_path):
    paper = OpenAlexLiteratureBackend()._paper(work())
    other = _FakeBackend([paper])
    backend = MultiSourceLiteratureBackend(Failing(LiteratureUnavailableError("arXiv HTTP 429")), other)
    register = _FakeRegister(tmp_path)
    observation = LiteratureSearchTool(backend).execute(
        state(), LiteratureSearchToolInput(query="x", max_results=2, start_year=2020, end_year=2024),
    )
    assert other.last_kwargs == dict(query="x", max_results=2, source="peer", scope="topic", page=1, start_year=2020, end_year=2024)
    assert "arXiv HTTP 429" in caplog.text
    assert observation.value["papers"][0]["source_url"] == paper.source_url
    assert observation.value["papers"][0]["fetch_source"] == "openalex:W123"
    assert observation.value["status"] == "results"
    assert observation.memory_updates == {} and register.refs == {}


def test_empty_result_does_not_trigger_source_switch():
    other = _FakeBackend([])
    assert MultiSourceLiteratureBackend(_FakeBackend([], name="first"), other).search("x", max_results=1).papers == []
    assert other.last_kwargs is None


@pytest.mark.parametrize("error", [LiteratureSearchError("HTTP 400"), LiteratureSearchError("invalid XML"), RuntimeError("bug")])
def test_non_availability_errors_do_not_trigger_source_switch(error):
    other = _FakeBackend([])
    with pytest.raises(type(error), match=str(error)):
        MultiSourceLiteratureBackend(Failing(error), other).search("x", max_results=1)
    assert other.last_kwargs is None


def test_both_sources_failing_records_failure_without_registering_papers(tmp_path):
    backend = MultiSourceLiteratureBackend(
        Failing(LiteratureUnavailableError("arXiv HTTP 429")),
        Failing(LiteratureUnavailableError("OpenAlex TimeoutError"), "openalex"),
    )
    register = _FakeRegister(tmp_path)
    result = LiteratureSearchTool(backend).execute(state(), LiteratureSearchToolInput(query="x"))
    assert not result.ok
    assert result.value["status"] == "failed"
    assert "arXiv HTTP 429" in result.value["error"]
    assert "OpenAlex TimeoutError" in result.value["error"]
    assert register.refs == {} and result.memory_updates == {}
    assert result.value["papers"] == []


@pytest.mark.parametrize("remaining,reset,retry_after,expected", [
    ("0", "3600", None, 3600),
    ("0.0", "90", "120", 120),
    ("0", "30", None, 60),
    ("1", "3600", None, 60),
    ("0.5", "3600", None, 60),
    (None, "3600", None, 60),
    ("-1", "3600", None, 60),
    ("NaN", "3600", None, 60),
    ("1e-999", "3600", None, 60),
    ("0", None, None, 60),
    ("0", "0", None, 60),
    ("0", "-1", None, 60),
    ("0", "NaN", None, 60),
    ("0", "inf", None, 60),
    ("0", "unknown", None, 60),
    ("0", "1" * 100, None, 60),
])
def test_429_only_explicit_exhausted_quota_extends_cooldown(
    monkeypatch, caplog, remaining, reset, retry_after, expected,
):
    from tests.components.test_literature_http import response_records

    headers = {
        key: value for key, value in {
            "X-RateLimit-Remaining": remaining,
            "X-RateLimit-Reset": reset,
            "Retry-After": retry_after,
        }.items() if value is not None
    }
    calls = []
    def open_request(request, *, timeout, max_response_bytes):
        assert max_response_bytes == _literature_http.METADATA_RESPONSE_MAX_BYTES
        calls.append(request)
        response = httpx.Response(429, request=request, headers=headers)
        response.raise_for_status()

    monkeypatch.setattr(openalex, "send_request", open_request)
    backend = OpenAlexLiteratureBackend()
    with pytest.raises(LiteratureUnavailableError, match="429"):
        backend.search("calibration", max_results=3)
    assert openalex._OPENALEX_HTTP._cooldown_until == expected
    assert response_records(caplog)[-1]["cooldown_seconds"] == expected
    with pytest.raises(LiteratureUnavailableError, match="cooling down"):
        backend.search("other keywords", max_results=3)
    assert len(calls) == 1


def test_exhausted_openalex_quota_does_not_block_peer_search(monkeypatch):
    def open_request(request, *, timeout, max_response_bytes):
        assert max_response_bytes == _literature_http.METADATA_RESPONSE_MAX_BYTES
        httpx.Response(429, request=request, headers={
            "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "3600",
        }).raise_for_status()

    monkeypatch.setattr(openalex, "send_request", open_request)
    peer = _FakeBackend([OpenAlexLiteratureBackend()._paper(work())])
    backend = MultiSourceLiteratureBackend(OpenAlexLiteratureBackend(), peer)
    assert len(backend.search("calibration", max_results=3).papers) == 1
    assert openalex._OPENALEX_HTTP._cooldown_until == 3600
    assert peer.last_kwargs["query"] == "calibration"


def test_quota_reset_does_not_change_non_429_retries(monkeypatch):
    calls = []
    def open_request(request, *, timeout, max_response_bytes):
        assert max_response_bytes == _literature_http.METADATA_RESPONSE_MAX_BYTES
        calls.append(request)
        response = httpx.Response(503, request=request, headers={
            "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "3600",
        })
        response.raise_for_status()

    monkeypatch.setattr(openalex, "send_request", open_request)
    with pytest.raises(LiteratureUnavailableError, match="after 3 attempts"):
        OpenAlexLiteratureBackend().search("calibration", max_results=3)
    assert len(calls) == 3
    assert openalex._OPENALEX_HTTP._cooldown_until == 69


@pytest.mark.parametrize('scope,key', [('topic', 'search.title_and_abstract'), ('title', 'search.title')])
def test_scope_page_and_count(monkeypatch, scope, key):
    requests = fake_response(monkeypatch, json.dumps({'meta': {'count': 12}, 'results': [work()]}).encode())
    result = OpenAlexLiteratureBackend().search('precise title', max_results=5, source='openalex', scope=scope, page=2)
    params = parse_qs(urlsplit(str(requests[0][0].url)).query)
    assert params[key] == ['"precise title"' if scope == 'title' else 'precise title']
    assert params['page'] == ['2']
    assert 'search' not in params
    assert result.total_results == 12
    assert result.next_page == 3
    assert result.source == 'openalex'


def test_missing_total_is_unknown_not_exhausted(monkeypatch):
    fake_response(monkeypatch, b'{"results": []}')
    result = OpenAlexLiteratureBackend().search('q', max_results=5)
    assert result.total_results is None
    assert result.next_page is None


def test_arxiv_topic_title_paging(monkeypatch):
    from resagent2_components import ArxivLiteratureBackend
    from tests.capabilities.test_literature import ARXIV_ATOM
    captured = []
    backend = ArxivLiteratureBackend()
    monkeypatch.setattr(backend, '_fetch', lambda url: (captured.append(url) or ARXIV_ATOM.encode()))
    result = backend.search('precise title', max_results=3, source='arxiv', scope='topic', page=2)
    params = parse_qs(urlsplit(captured[-1]).query)
    assert params['start'] == ['3']
    assert params['sortBy'] == ['relevance']
    assert params['search_query'] == ['(ti:"precise" OR abs:"precise") AND (ti:"title" OR abs:"title")']
    assert result.executed_query == params['search_query'][0]
    backend.search('precise title', max_results=3, scope='title')
    assert parse_qs(urlsplit(captured[-1]).query)['search_query'] == ['ti:"precise title"']




def test_duplicate_raw_page_still_exposes_next(monkeypatch):
    fake_response(monkeypatch, json.dumps({'results': [work(), work()]}).encode())
    result = OpenAlexLiteratureBackend().search('q', max_results=2)
    assert len(result.papers) == 1
    assert result.next_page == 2
    assert result.total_results is None


def test_openalex_last_supported_page_has_no_invalid_next(monkeypatch):
    fake_response(monkeypatch, json.dumps({'meta': {'count': 20000}, 'results': [work()]}).encode())
    result = OpenAlexLiteratureBackend().search('q', max_results=100, source='openalex', page=100)
    assert result.next_page is None
    assert result.total_results == 20000


def test_openalex_unbalanced_phrase_rejected_before_request(monkeypatch):
    requests = fake_response(monkeypatch, b'{"results": []}')
    with pytest.raises(LiteratureSearchError) as caught:
        OpenAlexLiteratureBackend().search('"broken', max_results=5)
    assert caught.value.error_type == 'invalid_query'
    assert requests == []


KNOWN_TITLES = [
    'On Calibration of Modern Neural Networks',
    'Attention Is All You Need',
    'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
    'Language Models are Few-Shot Learners',
    'A Simple Framework for Contrastive Learning of Visual Representations',
]


@pytest.mark.parametrize('title', KNOWN_TITLES)
@pytest.mark.parametrize('quoted', [False, True])
def test_complete_title_phrase_for_both_sources(monkeypatch, title, quoted):
    from resagent2_components import ArxivLiteratureBackend
    # Request translation only; remote retrieval quality is measured separately.
    query = '"' + title + '"' if quoted else title
    expected = '"' + title + '"'
    assert ArxivLiteratureBackend._search_query(query, 2015, 2025, 'title') == (
        'ti:' + expected + ' AND submittedDate:[201501010000 TO 202512312359]')
    requests = fake_response(monkeypatch, b'{"results": []}')
    result = OpenAlexLiteratureBackend().search(query, max_results=5, scope='title')
    params = parse_qs(urlsplit(str(requests[0][0].url)).query)
    assert params['search.title'] == [expected]
    assert result.executed_query == expected
    assert 'search.title_and_abstract' not in params


def test_title_phrase_normalizes_spacing_and_preserves_escaped_punctuation(monkeypatch):
    from resagent2_components import ArxivLiteratureBackend
    query = r'"On   \"Quoted\" Titles: A Few-Shot Study"'
    expected = r'"On \"Quoted\" Titles: A Few-Shot Study"'
    assert ArxivLiteratureBackend._search_query(query, None, None, 'title') == 'ti:' + expected
    requests = fake_response(monkeypatch, b'{"results": []}')
    result = OpenAlexLiteratureBackend().search(query, max_results=3, scope='title')
    assert parse_qs(urlsplit(str(requests[0][0].url)).query)['search.title'] == [expected]
    assert result.executed_query == expected


def test_topic_keeps_keyword_and_phrase_query_unchanged():
    from resagent2_components import ArxivLiteratureBackend
    assert ArxivLiteratureBackend._search_query('calibration "temperature scaling"', None, None) == (
        '(ti:"calibration" OR abs:"calibration") AND '
        '(ti:"temperature scaling" OR abs:"temperature scaling")')


@pytest.mark.parametrize('quoted', [False, True])
def test_title_phrase_preserves_apostrophe_and_escaped_backslash(monkeypatch, quoted):
    from resagent2_components import ArxivLiteratureBackend
    title = r"A Researcher's Guide to C:\\Models"
    query = '"' + title + '"' if quoted else title
    expected = r'''"A Researcher's Guide to C:\\Models"'''
    assert ArxivLiteratureBackend._search_query(query, None, None, 'title') == 'ti:' + expected
    requests = fake_response(monkeypatch, b'{"results": []}')
    result = OpenAlexLiteratureBackend().search(query, max_results=3, scope='title')
    assert parse_qs(urlsplit(str(requests[0][0].url)).query)['search.title'] == [expected]
    assert result.executed_query == expected
