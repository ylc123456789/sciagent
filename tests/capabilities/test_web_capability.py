"""Search stays in history; fetch explicitly preserves authorized snapshots."""

from dataclasses import replace
from datetime import UTC, datetime
import hashlib

import httpx
import pytest

from resagent2_contracts import AgentOwner, SessionStatus
from resagent2_orchestrator import ArtifactRegistry
from resagent2_components import (
    DeepSeekWebSearchBackend, WebFetchError, WebPage, WebPageFetcher, WebSearchError,
    WebSearchItem, WebSearchResult, RegisteredArtifactReader,
)
from resagent2_capabilities import WebFetchInput, WebFetchTool, WebSearchInput, WebSearchTool
from resagent2_runtime import AgentState


def _state():
    now = datetime.now(UTC)
    return AgentState(session_id="session_sci", agent_name="scientific", owner=AgentOwner.SCIENTIFIC,
                      run_id="run_example", status=SessionStatus.ACTIVE, created_at=now, updated_at=now, current_operation_key="session:session_sci:turn:0:call:web")


class _Register:
    def __init__(self, root):
        self.registry = ArtifactRegistry(root)
        self.refs = {}

    def register_scientific(self, candidate, *, run_id, session_id, content_bytes=None, operation_key=None, output_position=None):
        ref = self.registry.register_scientific(candidate, run_id=run_id, session_id=session_id,
                                                content_bytes=content_bytes)
        self.refs[ref.id] = ref
        return ref

    def list_artifacts(self, *, run_id, session_id):
        return [ref for ref in self.refs.values() if ref.run_id == run_id and ref.session_id == session_id]


class _Backend:
    name = "fake"
    def __init__(self, *, items=None, failure=None, incomplete_reason=None):
        self.items = items or []
        self.failure = failure
        self.incomplete_reason = incomplete_reason
        self.calls = []
    def search(self, query):
        self.calls.append(query)
        if self.failure:
            raise self.failure
        return WebSearchResult(self.name, query, self.items, self.incomplete_reason)


@pytest.mark.parametrize("mode", ["results", "empty", "partial", "failed"])
def test_search_all_outcomes_are_transient_receipts(mode):
    items = [WebSearchItem("A", "https://example.test/a", "snippet")] if mode in {"results", "partial"} else []
    backend = _Backend(items=items, incomplete_reason="max_uses_exceeded" if mode == "partial" else None,
                       failure=WebSearchError("limited", error_type="rate_limited", retry_after=4)
                       if mode == "failed" else None)
    original = _state()
    before = original.model_copy(deep=True)
    result = WebSearchTool(backend).execute(original, WebSearchInput(query="q"))
    assert backend.calls == ["q"]
    assert result.value["status"] == mode and result.ok == (mode != "failed")
    assert result.memory_updates == {} and original == before
    assert "artifact" not in result.value and "artifact_id" not in result.value
    assert "omitted_count" not in result.value
    if items:
        assert result.value["results"][0]["fetch_source"] == items[0].url
    if mode == "failed":
        assert result.value["retry_after"] == 4


def test_search_sources_are_returned_without_preview_cap_or_duplicates():
    class Client:
        def set_trace_context(self, **kwargs):
            pass
        def request(self, body):
            return {"results": [
                {"title": str(i), "url": f"https://example.test/{i}",
                 "snippet": "Excerpt", "summary": "Provider summary", "content": "",
                 "rerank_score": 1.5, "authority": -1, "date": None,
                 "site_name": None, "site_icon": None}
                for i in [*range(28), 0, 1]
            ]}
    result = WebSearchTool(DeepSeekWebSearchBackend(Client(), count=50)).execute(_state(), WebSearchInput(query="q"))
    assert [item["url"] for item in result.value["results"]] == [f"https://example.test/{i}" for i in range(28)]
    assert result.value["result_count"] == 28 and result.memory_updates == {}
    first = result.value["results"][0]
    assert first["snippet"] == "Excerpt" and first["summary"] == "Provider summary"
    assert first["score"] == 1.5 and first["authority"] == -1
    assert first["summary_truncated"] is False and "content" not in first


def test_removed_preview_input_is_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        WebSearchInput(query="q", max_results=1)


@pytest.mark.parametrize("query", ["", "   ", "a" * 129, "搜" * 129])
def test_search_schema_rejects_invalid_query_before_execution(query):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        WebSearchInput(query=query)


def test_search_schema_exposes_provider_limit_without_truncating_unicode():
    from resagent2_runtime.tool_calling import native_tool_schemas
    schema = native_tool_schemas((WebSearchTool(_Backend()),))[0]["function"]["parameters"]["properties"]["query"]
    assert schema["maxLength"] == 128
    assert WebSearchInput(query="  " + "搜" * 128 + "  ").query == "搜" * 128


class _Fetcher:
    def __init__(self, page=None, error=None):
        self.page, self.error = page, error
    def fetch(self, url):
        if self.error:
            raise self.error
        return self.page


def _page(**updates):
    page = WebPage("https://example.test/source", "https://example.test/page", "Title",
                   "Body\n    code\n", "text/html", "markdownify/html.parser", "2026-10-06T00:00:00+00:00",
                   "test-version")
    return replace(page, **updates)


def test_fetch_preserves_source_hash_and_time_in_metadata(tmp_path):
    register, state = _Register(tmp_path), _state()
    page = _page()
    result = WebFetchTool(_Fetcher(page), register).execute(state, WebFetchInput(url=page.source_url))
    ref, = register.refs.values()
    content = RegisteredArtifactReader([ref], run_id=state.run_id).read_text(ref.id)["content"]
    assert result.ok and result.value["artifact_id"] == ref.id and not result.value["reused"]
    assert "Fetched at:" not in content and page.fetched_at not in content
    assert "    code\n" in content
    assert hashlib.sha256(content.encode()).hexdigest() == ref.sha256
    assert ref.metadata["fetched_at"] == result.value["fetched_at"] == page.fetched_at
    assert ref.metadata["parser_version"] == "test-version"


def test_repeated_fetch_reuses_snapshot_but_keeps_current_observation_time(tmp_path):
    register, state = _Register(tmp_path), _state()
    first = WebFetchTool(_Fetcher(_page()), register).execute(state, WebFetchInput(url=_page().source_url))
    state.authorized_artifact_ids = list(register.refs)
    next_page = _page(fetched_at="2026-10-07T00:00:00+00:00")
    second = WebFetchTool(_Fetcher(next_page), register).execute(state, WebFetchInput(url=next_page.source_url))
    ref, = register.refs.values()
    assert second.value["artifact_id"] == first.value["artifact_id"] and second.value["reused"]
    assert ref.metadata["fetched_at"] == _page().fetched_at
    assert second.value["fetched_at"] == next_page.fetched_at
    assert ref.producer == AgentOwner.SCIENTIFIC


def test_unaccepted_same_session_material_is_not_reused(tmp_path):
    register, state = _Register(tmp_path), _state()
    first = WebFetchTool(_Fetcher(_page()), register).execute(state, WebFetchInput(url=_page().source_url))
    # The Run may already hold bytes from another interrupted receipt. Mere
    # production in the same Session is not a model grant.
    second = WebFetchTool(_Fetcher(_page()), register).execute(state, WebFetchInput(url=_page().source_url))
    assert state.authorized_artifact_ids == []
    assert second.value["artifact_id"] != first.value["artifact_id"] and not second.value["reused"]


def test_authorized_input_page_is_reused_without_changing_producer(tmp_path):
    register, state = _Register(tmp_path), _state()
    first = WebFetchTool(_Fetcher(_page()), register).execute(state, WebFetchInput(url=_page().source_url))
    original = register.refs[first.value["artifact_id"]]
    other = state.model_copy(update={"session_id": "session_other"})
    result = WebFetchTool(_Fetcher(_page()), register, input_artifacts=[original]).execute(
        other, WebFetchInput(url=_page().source_url))
    assert result.value["artifact_id"] == original.id and result.value["reused"]
    assert result.artifacts == [] and register.refs[original.id] == original


@pytest.mark.parametrize("updates", [
    {"text": "Changed"}, {"source_url": "https://example.test/another"},
    {"final_url": "https://example.test/new"}, {"parser_version": "new-version"},
])
def test_changed_content_source_or_parser_has_new_snapshot(tmp_path, updates):
    register, state = _Register(tmp_path), _state()
    first = WebFetchTool(_Fetcher(_page()), register).execute(state, WebFetchInput(url=_page().source_url))
    state.authorized_artifact_ids = list(register.refs)
    page = _page(**updates)
    second = WebFetchTool(_Fetcher(page), register).execute(state, WebFetchInput(url=page.source_url))
    assert not second.value["reused"] and second.value["artifact_id"] != first.value["artifact_id"]
    assert len(register.refs) == 2


def test_fetch_failure_creates_no_artifact(tmp_path):
    register = _Register(tmp_path)
    result = WebFetchTool(_Fetcher(error=WebFetchError("bad", error_type="parse_failed")), register).execute(
        _state(), WebFetchInput(url="https://example.test/page"))
    assert not result.ok and result.value["error_type"] == "parse_failed"
    assert register.refs == {} and result.memory_updates == {}


def test_html_links_survive_freeze_and_exact_read(tmp_path, monkeypatch):
    final_url = "https://example.test/papers/index.html"
    response = httpx.Response(200, headers={"content-type": "text/html"},
                              content=b"<title>Paper page</title><p><a href='paper.pdf'>Download PDF</a></p>",
                              request=httpx.Request("GET", final_url))
    fetcher = WebPageFetcher()
    monkeypatch.setattr(fetcher, "_request", lambda url: response)
    register, state = _Register(tmp_path), _state()
    result = WebFetchTool(fetcher, register).execute(state, WebFetchInput(url="https://example.test/original"))
    ref, = register.refs.values()
    content = RegisteredArtifactReader([ref], run_id=state.run_id).read_text(ref.id)["content"]
    assert result.value["final_url"] == final_url
    assert "https://example.test/papers/paper.pdf" in content
    assert ref.metadata["parser"] == "markdownify/html.parser"


def test_corrupt_compressed_search_body_returns_failure_without_artifact(monkeypatch, tmp_path):
    from resagent2_runtime import ServiceRequestClient, http
    from resagent2_runtime.budget import execution_budget
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-encoding": "gzip"},
                              stream=httpx.ByteStream(b"PRIVATE_CORRUPT_RESPONSE"))
    original = httpx.AsyncClient
    monkeypatch.setattr(http.httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setenv("TEST_SEARCH_KEY", "secret")
    backend = DeepSeekWebSearchBackend(ServiceRequestClient(
        endpoint="https://model.example.test/search", api_key_env="TEST_SEARCH_KEY",
        trace_dir=tmp_path, trace_level="metadata"))
    with execution_budget(max_llm_calls=1, timeout_seconds=5) as budget:
        result = WebSearchTool(backend).execute(_state(), WebSearchInput(query="q"))
        assert budget.usage.used == 0 and budget.usage.requests == {}
    assert len(calls) == 1 and not result.ok
    assert result.value["error_type"] == "invalid_response" and result.memory_updates == {}
    assert "PRIVATE_CORRUPT_RESPONSE" not in str(result.value)

    trace_files = list(tmp_path.rglob("*.jsonl"))
    assert len(trace_files) == 1 and trace_files[0].name == "service_requests.jsonl"
    assert len(trace_files[0].read_text().splitlines()) == 1
