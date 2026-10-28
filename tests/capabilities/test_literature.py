"""Tests for the literature search capability (DEVELOPMENT_PLAN §7.3)."""

from datetime import UTC, datetime, date
from urllib.parse import parse_qs, urlsplit

import httpx
import re

from resagent2_orchestrator import ArtifactRegistry
import pytest

from resagent2_components import (
    ArxivLiteratureBackend,
    LiteraturePaper,
    LiteratureSearchError,
)
from resagent2_capabilities import (
    LiteratureSearchTool,
    LiteratureSearchToolInput,
)
from resagent2_contracts import (
    AgentOwner,
    SessionStatus,
)
from resagent2_runtime import AgentState


NOW = datetime(2026, 8, 28, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolated_http(monkeypatch, request):
    if request.node.name == "test_arxiv_backend_live_smoke":
        return
    from resagent2_components.literature import (
        _http as _literature_http,
        backends as literature,
    )

    clock = [0.0]
    monkeypatch.setattr(_literature_http.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        _literature_http.time, "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(
        literature, "_ARXIV_HTTP",
        _literature_http.LiteratureHTTP("arXiv", interval_seconds=3),
    )


def state(*, memory: dict | None = None) -> AgentState:
    return AgentState(
        session_id="session_sci",
        agent_name="scientific",
        owner=AgentOwner.SCIENTIFIC,
        run_id="run_example",
        task_id="task_sci",
        attempt_number=1,
        status=SessionStatus.ACTIVE,
        memory=memory or {},
        created_at=NOW,
        updated_at=NOW,
    )


class _FakeBackend:
    """Returns fixed papers, recording the last query arguments."""

    def __init__(self, papers: list[LiteraturePaper], name="peer") -> None:
        self.name = name
        self._papers = papers
        self.last_kwargs: dict | None = None

    def search(self, query, *, max_results, start_year=None, end_year=None, source="auto", scope="topic", page=1):
        self.last_kwargs = {
            "query": query,
            "max_results": max_results,
            "start_year": start_year,
            "end_year": end_year, "source": source, "scope": scope, "page": page,
        }
        result = _search_result(self._papers)
        result.source = self.name
        result.source_attempts[0]["source"] = self.name
        return result


class _FakeRegister:
    """Exercise actual freezing while keeping the Run store out of tool tests."""

    def __init__(self, root) -> None:
        self.registry = ArtifactRegistry(root)
        self.refs = {}
        self.last_run_id = self.last_session_id = self.last_candidate = None

    def register_scientific(self, candidate, *, run_id, session_id, content_bytes=None):
        self.last_run_id, self.last_session_id = run_id, session_id
        self.last_candidate = candidate
        ref = self.registry.register_scientific(
            candidate, run_id=run_id, session_id=session_id, content_bytes=content_bytes,
        )
        self.refs[ref.id] = ref
        return ref

    def list_artifacts(self, *, run_id):
        return [ref for ref in self.refs.values() if ref.run_id == run_id]

    def resolve(self, artifact_id, *, run_id):
        ref = self.refs.get(artifact_id)
        return ref if ref and ref.run_id == run_id else None


def paper(paper_id: str) -> LiteraturePaper:
    return LiteraturePaper(
        paper_id=paper_id,
        title=f"Title {paper_id}",
        authors=["Alice"],
        published_at=date(2024, 1, 1),
        abstract="An abstract.",
        source_url=f"https://arxiv.org/abs/{paper_id}",
    )


def test_search_returns_deduplicated_leads_without_registration():
    backend = _FakeBackend([paper("2301.00001v2"), paper("2301.00001v2"), paper("2301.00002v1")])
    original = state(memory={"artifact_index": [{"artifact_id": "artifact_existing"}]})
    before = original.model_copy(deep=True)
    result = LiteratureSearchTool(backend).execute(original, LiteratureSearchToolInput(query="electron"))
    assert result.ok and result.value["status"] == "results"
    assert [item["fetch_source"] for item in result.value["papers"]] == [
        "arxiv:2301.00001v2", "arxiv:2301.00002v1",
    ]
    assert all("artifact_id" not in item for item in result.value["papers"])
    assert "artifact" not in result.value
    assert result.memory_updates == {} and original == before


def test_search_preserves_provided_abstract_without_claiming_fulltext():
    abstract = "Scientific evidence. " * 100
    candidate = paper("2301.00001v2").model_copy(update={"abstract": abstract})
    result = LiteratureSearchTool(_FakeBackend([candidate])).execute(
        state(), LiteratureSearchToolInput(query="evidence"),
    )
    assert result.value["papers"][0]["abstract"] == abstract
    assert result.memory_updates == {}
    assert "fulltext_artifact_id" not in result.value


@pytest.mark.parametrize("failure", [False, True])
def test_empty_or_failed_search_does_not_create_artifacts(failure):
    class Backend:
        def search(self, *args, **kwargs):
            if failure:
                raise LiteratureSearchError("provider unavailable")
            return _search_result([])
    result = LiteratureSearchTool(Backend()).execute(state(), LiteratureSearchToolInput(query="missing"))
    assert result.value["status"] == ("failed" if failure else "empty")
    assert result.ok is (not failure)
    assert result.value["papers"] == []
    assert result.memory_updates == {} and "artifact" not in result.value


def test_tool_forwards_query_and_bounds():
    backend = _FakeBackend([paper("2301.00001v2")])
    LiteratureSearchTool(backend).execute(
        state(), LiteratureSearchToolInput(
            query="graph neural networks", max_results=7, start_year=2020, end_year=2024,
        ),
    )
    assert backend.last_kwargs == {
        "query": "graph neural networks", "max_results": 7, "start_year": 2020,
        "end_year": 2024, "source": "auto", "scope": "topic", "page": 1,
    }


ARXIV_ATOM = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v2</id>
    <title>  First Paper   </title>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <published>2023-01-05T12:00:00Z</published>
    <summary>  A line break
      abstract that keeps going.  </summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00001v2</id>
    <title>Duplicate</title>
    <author><name>Alice</name></author>
    <published>2023-01-05T12:00:00Z</published>
    <summary>duplicate</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00002v1</id>
    <title>Second</title>
    <published>2023-01-06T12:00:00Z</published>
    <summary>short</summary>
  </entry>
</feed>
"""


class _FakeArxivBackend(ArxivLiteratureBackend):
    """Override _request so no network is touched, keeping the retry logic."""

    def __init__(self, body: bytes) -> None:
        super().__init__()
        self._body = body

    def _request(self, url: str) -> httpx.Response:
        self.last_url = url
        return httpx.Response(200, content=self._body, request=httpx.Request("GET", url))


def test_arxiv_backend_parses_deduplicates_and_normalizes() -> None:
    backend = _FakeArxivBackend(ARXIV_ATOM.encode("utf-8"))
    papers = backend.search("electron", max_results=10).papers

    assert [p.paper_id for p in papers] == ["2301.00001v2", "2301.00002v1"]
    first = papers[0]
    assert first.title == "First Paper"
    assert first.authors == ["Alice", "Bob"]
    assert first.published_at == date(2023, 1, 5)
    assert first.abstract == "A line break abstract that keeps going."
    assert first.source_url == "https://arxiv.org/abs/2301.00001v2"
    assert first.pdf_url == "https://arxiv.org/pdf/2301.00001v2"


def test_arxiv_backend_preserves_complete_abstract() -> None:
    long_abstract = "x" * 5000
    atom = ARXIV_ATOM.replace(
        "short", long_abstract
    )
    backend = _FakeArxivBackend(atom.encode("utf-8"))
    papers = backend.search("electron", max_results=10).papers

    assert papers[-1].abstract == long_abstract


def test_arxiv_backend_builds_year_bounded_query() -> None:
    backend = _FakeArxivBackend(ARXIV_ATOM.encode("utf-8"))
    backend.search("electron", max_results=5, start_year=2020, end_year=2024)

    assert "submittedDate" in backend.last_url
    assert "202001010000" in backend.last_url
    assert "202412312359" in backend.last_url


@pytest.mark.parametrize("query,expected", [
    ("electron", 'all:"electron"'),
    ("graph neural networks", 'all:"graph" AND all:"neural" AND all:"networks"'),
    ('"temperature scaling" calibration', 'all:"temperature scaling" AND all:"calibration"'),
    ("C++ O'Neill", "all:\"C++\" AND all:\"O'Neill\""),
    (r'"quote \"inside\"" "path\\name"', r'all:"quote \"inside\"" AND all:"path\\name"'),
    ("a OR ti:b", 'all:"a" AND all:"OR" AND all:"ti:b"'),
])
def test_arxiv_translates_keywords_and_phrases_without_query_operators(query, expected):
    backend = _FakeArxivBackend(ARXIV_ATOM.encode())
    backend.search(query, max_results=3, start_year=2020, end_year=2024)
    params = parse_qs(urlsplit(backend.last_url).query)
    expected = re.sub(r'all:("(?:\\.|[^"\\])*")', lambda m: f"(ti:{m[1]} OR abs:{m[1]})", expected)
    assert params["search_query"] == [
        expected + " AND submittedDate:[202001010000 TO 202412312359]"
    ]
    assert params["max_results"] == ["3"]


@pytest.mark.parametrize("query", ['"unclosed', '""', " \t ", 'keyword ""'])
def test_arxiv_rejects_malformed_keyword_input_before_http(query):
    backend = _FakeArxivBackend(ARXIV_ATOM.encode())
    with pytest.raises(LiteratureSearchError):
        backend.search(query, max_results=3)
    assert not hasattr(backend, "last_url")


def test_arxiv_backend_raises_clear_error_instead_of_empty_result() -> None:
    class Failing(ArxivLiteratureBackend):
        def _request(self, url: str) -> httpx.Response:
            raise TimeoutError("connection timed out")

    backend = Failing(max_retries=2)
    with pytest.raises(LiteratureSearchError, match="failed after 2 attempts"):
        backend.search("electron", max_results=5)


def test_arxiv_backend_rejects_invalid_xml() -> None:
    backend = _FakeArxivBackend(b"not xml")
    with pytest.raises(LiteratureSearchError, match="invalid XML"):
        backend.search("electron", max_results=5)


@pytest.mark.parametrize("body", [
    b"<html>service error</html>",
    b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors</id></entry></feed>',
    b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2301.00001</id></entry></feed>',
])
def test_arxiv_invalid_feed_is_not_an_empty_search(body):
    with pytest.raises(LiteratureSearchError):
        _FakeArxivBackend(body).search("x", max_results=1)


def test_arxiv_request_identifies_application(monkeypatch):
    from resagent2_components.literature import backends as literature

    requests = []
    def open_request(request, *, timeout, max_response_bytes):
        assert max_response_bytes == literature.METADATA_RESPONSE_MAX_BYTES
        requests.append((request, timeout))
        return httpx.Response(200, content=ARXIV_ATOM.encode(), request=request)
    monkeypatch.setattr(literature, "send_request", open_request)
    ArxivLiteratureBackend(timeout_seconds=7).search("x", max_results=1)
    request, timeout = requests[0]
    assert request.headers.get("User-agent").startswith("ResAgent2/")
    assert timeout == 7


@pytest.mark.skipif(
    not __import__("os").environ.get("RESAGENT2_LITERATURE_SMOKE"),
    reason="network smoke test is opt-in via RESAGENT2_LITERATURE_SMOKE",
)
def test_arxiv_backend_live_smoke() -> None:
    """Opt-in real network test; must not fake success on rate limit/timeout."""
    backend = ArxivLiteratureBackend(timeout_seconds=10, max_retries=2)
    papers = backend.search("graph neural network", max_results=3).papers
    assert len(papers) >= 1
    assert all(p.title and p.source_url for p in papers)


def _search_result(papers):
    from resagent2_components import LiteratureSearchResult
    return LiteratureSearchResult(papers=papers, source="arxiv", executed_query="test query",
                                  page=1, next_page=None, source_attempts=[{
                                      "source": "arxiv", "status": "success",
                                      "error_type": None, "error": None,
                                  }])


@pytest.mark.parametrize("arguments", [
    {"source": "auto", "page": 2}, {"page": 0},
    {"start_year": 2025, "end_year": 2020}, {"max_results": 21},
    {"source": "unknown"}, {"scope": "abstract"},
])
def test_search_input_rejects_invalid_request(arguments):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LiteratureSearchToolInput(query="topic", **arguments)


def test_search_receipt_exposes_real_source_and_pinned_next_request(tmp_path):
    from resagent2_components import LiteratureSearchResult
    attempts = [
        {"source": "arxiv", "status": "failed", "error_type": "unavailable", "error": "HTTP 429"},
        {"source": "openalex", "status": "success", "error_type": None, "error": None},
    ]
    class Backend:
        def search(self, *args, **kwargs):
            return LiteratureSearchResult([paper("1")], "openalex", "executed topic", 1, 2, attempts)
    register = _FakeRegister(tmp_path)
    observation = LiteratureSearchTool(Backend()).execute(
        state(), LiteratureSearchToolInput(query="topic", scope="title", start_year=2020),
    )
    value = observation.value
    assert value["source"] == "openalex"
    assert value["executed_query"] == "executed topic"
    assert value["source_attempts"] == attempts
    assert value["total_results"] is None
    assert value["papers"][0]["fetch_source"] == "arxiv:1"
    assert value["next_request"] == {
        "query": "topic", "source": "openalex", "scope": "title", "page": 2,
        "max_results": 5, "start_year": 2020, "end_year": None,
    }
    assert value["requested_source"] == "auto"
    assert observation.memory_updates == {} and register.refs == {}


def test_search_failure_preserves_source_errors(tmp_path):
    attempts = [{"source": "arxiv", "status": "failed", "error_type": "unavailable", "error": "HTTP 429"}]
    class Backend:
        def search(self, *args, **kwargs):
            raise LiteratureSearchError("HTTP 429", error_type="unavailable", source_attempts=attempts, retry_after=30)
    register = _FakeRegister(tmp_path)
    value = LiteratureSearchTool(Backend()).execute(
        state(), LiteratureSearchToolInput(query="topic", source="arxiv"),
    ).value
    assert value["status"] == "failed"
    assert value["source_attempts"] == attempts
    assert value["error_type"] == "unavailable"
    assert value["retry_after"] == 30
    assert value["next_request"] is None
