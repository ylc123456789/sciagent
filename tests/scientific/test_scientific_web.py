"""Scientific web tools follow the existing runtime and artifact handoff."""

import hashlib
import json

import pytest

from resagent2_components import (
    RegisteredArtifactReader, WebFetchError, WebPage, WebSearchError,
    WebSearchItem, WebSearchResult,
)
from resagent2_contracts import (
    AgentPermissions, AgentRequest, ArtifactCandidate, ArtifactRef, TaskBudget,
)
from resagent2_orchestrator import ArtifactRegistry
from resagent2_runtime import NativeToolCall, ToolCallTurn
from resagent2_scientific import ScientificAgent


RUN_ID = "run_web_integration"
URL = "https://example.test/source"


class Registration:
    def __init__(self, root):
        self.registry = ArtifactRegistry(root)
        self.refs = {}

    def register_scientific(self, candidate, **kwargs):
        kwargs.pop("operation_key", None)
        kwargs.pop("output_position", None)
        ref = self.registry.register_scientific(candidate, **kwargs)
        self.refs[ref.id] = ref
        return ref

    def resolve(self, artifact_id, *, run_id, session_id):
        ref = self.refs.get(artifact_id)
        return ref if ref and ref.run_id == run_id and ref.session_id == session_id else None

    def list_artifacts(self, *, run_id, session_id):
        return [ref for ref in self.refs.values() if ref.run_id == run_id and ref.session_id == session_id]

    def acknowledge_registration(self, **kwargs):
        pass

    def recover_registration(self, **kwargs):
        return []

    def of_kind(self, kind):
        return next(ref for ref in self.refs.values() if ref.kind == kind)


class SearchBackend:
    name = "test-search"

    def __init__(self, *, error=None, empty=False, results=None, incomplete_reason=None):
        self.error, self.empty, self.results = error, empty, results
        self.incomplete_reason = incomplete_reason
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return WebSearchResult(
            provider=self.name, query=query,
            results=[] if self.empty else (
                self.results if self.results is not None else [WebSearchItem("Source", URL, "A lead")]
            ), incomplete_reason=self.incomplete_reason,
        )


class PageFetcher:
    def __init__(self, *, error=None):
        self.error, self.calls = error, []

    def fetch(self, url):
        self.calls.append(url)
        if self.error:
            raise self.error
        return WebPage(
            source_url=url, final_url=url, title="Source", text="Original page detail",
            content_type="text/html", parser="test", fetched_at="2026-10-06T00:00:00+00:00",
        )


class UnusedLiteratureBackend:
    def search(self, *args, **kwargs):
        pytest.fail("No literature step was requested by this action plan")


class NativePlanClient:
    tool_session_key = "scientific-web-test"

    def __init__(self, actions):
        self.actions = iter(actions)
        self.contexts, self.schemas, self.turns = [], [], []

    def next_tool_call(self, context, schemas, turns, **kwargs):
        self.contexts.append(context)
        self.schemas.append(schemas)
        self.turns.append(turns)
        action = next(self.actions)
        action = action() if callable(action) else action
        return ToolCallTurn(tool_calls=[NativeToolCall(
            id=f"call_{len(self.contexts)}", name=action["tool"],
            arguments=json.dumps(action["arguments"]),
        )])


def request(*, artifacts=(), budget=8):
    return AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1,
        run_id=RUN_ID, agent="scientific", instruction="Answer using the needed sources",
        input_artifacts=list(artifacts), permissions=AgentPermissions(),
        budget=TaskBudget(max_llm_calls=budget, timeout_seconds=30),
    )


def finish(*refs):
    return {"tool": "finish", "arguments": {
        "report": "The available source material has been assessed",
        "artifacts": [{
            "kind": "scientific_opinion", "path": "opinion.json",
            "media_type": "application/json", "summary": "Judgment",
            "content": {
                "verdict": "inconclusive", "statement": "Available source assessed",
                "evidence_artifact_ids": [ref.id for ref in refs],
                "limitations": ["No independent research measurements"],
            },
        }],
    }}


def read(ref):
    return {"tool": "read_artifact", "arguments": {"artifact_id": ref.id}}


@pytest.mark.parametrize("search,fetch,registration", [
    (False, False, True), (True, False, True), (False, True, True),
    (True, True, True), (True, True, False),
])
def test_scientific_exposes_only_injected_web_tools(tmp_path, search, fetch, registration):
    client = NativePlanClient([finish()])
    result = ScientificAgent(
        client, web_search_backend=SearchBackend() if search else None,
        web_page_fetcher=PageFetcher() if fetch else None,
        registration_port=Registration(tmp_path) if registration else None,
    ).invoke(request())

    assert result.status == "completed", result.report
    names = {schema["function"]["name"] for schema in client.schemas[0]}
    assert ("web_search" in names) == search
    assert ("web_fetch" in names) == (fetch and registration)
    assert {"run_shell", "run_verification", "create_file"}.isdisjoint(names)


def test_web_artifacts_are_live_readable_citable_and_returned_without_copying(tmp_path):
    register = Registration(tmp_path)
    backend, fetcher = SearchBackend(), PageFetcher()
    client = NativePlanClient([
        {"tool": "web_search", "arguments": {"query": "a source"}},
        {"tool": "web_fetch", "arguments": {"url": URL}},
        lambda: read(register.of_kind("web_page")),
        lambda: finish(register.of_kind("web_page")),
    ])
    result = ScientificAgent(
        client, web_search_backend=backend, web_page_fetcher=fetcher,
        registration_port=register, literature_backend=UnusedLiteratureBackend(),
    ).invoke(request())

    assert result.status == "completed", result.report
    assert backend.calls == ["a source"]
    assert fetcher.calls == [URL]
    refs = [ref for ref in result.artifacts if isinstance(ref, ArtifactRef)]
    assert {ref.id for ref in refs} == set(register.refs)
    assert len(refs) == 1 and refs[0].kind == "web_page"
    reader = RegisteredArtifactReader(refs, run_id=RUN_ID)
    page = reader.read_text(register.of_kind("web_page").id)["content"]
    assert "Original page detail" in page
    assert hashlib.sha256(page.encode()).hexdigest() == register.of_kind("web_page").sha256
    receipt = json.loads(client.turns[-1][0].tool_results["call_1"])["value"]
    assert receipt["provider"] == "test-search" and receipt["results"][0]["url"] == URL
    trace = next(item for item in result.artifacts if item.kind == "observation_trace")
    assert json.loads(trace.content)["observed_artifact_ids"] == [register.of_kind("web_page").id]
    assert "Original page detail" in json.dumps(client.turns[-1], default=lambda x: x.model_dump())


def test_web_fetch_can_be_used_without_search_or_literature_step(tmp_path):
    register, fetcher = Registration(tmp_path), PageFetcher()
    backend = SearchBackend()
    client = NativePlanClient([
        {"tool": "web_fetch", "arguments": {"url": URL}},
        lambda: read(register.of_kind("web_page")),
        lambda: finish(register.of_kind("web_page")),
    ])
    result = ScientificAgent(
        client, web_search_backend=backend, web_page_fetcher=fetcher,
        registration_port=register, literature_backend=UnusedLiteratureBackend(),
    ).invoke(request())

    assert result.status == "completed", result.report
    assert backend.calls == [] and fetcher.calls == [URL]
    assert {ref.kind for ref in register.refs.values()} == {"web_page"}


@pytest.mark.parametrize("failure", [False, True])
def test_empty_or_failed_search_receipt_cannot_supply_required_paper_evidence(tmp_path, failure):
    register = Registration(tmp_path / "artifacts")
    requirements = register.registry.register_system_artifact(
        ArtifactCandidate(
            kind="conclusion_requirements", path="requirements.json",
            media_type="application/json", summary="A paper is required",
            content=json.dumps({"required_evidence_kinds": ["literature_paper"]}),
        ), run_id=RUN_ID, source_type="conclusion_requirement",
    )
    backend = SearchBackend(
        error=WebSearchError("limited", error_type="rate_limited", retry_after=3) if failure else None,
        empty=True,
    )
    client = NativePlanClient([
        {"tool": "web_search", "arguments": {"query": "paper"}},
        finish(),
        {"tool": "ask_user", "arguments": {
            "assessment": {"statement": "No required paper evidence is available"},
            "text": "The search supplied no paper. Please provide one or restore the service.",
            "requested_fields": ["paper"],
        }},
    ])
    result = ScientificAgent(
        client, web_search_backend=backend, registration_port=register,
    ).invoke(request(artifacts=[requirements]).model_copy(update={
        "task_context": {"conclusion_requirements": {"required_evidence_kinds": ["literature_paper"]}},
    }))

    assert result.status == "needs_user_input", result.report
    assert "literature_paper" in client.contexts[-1].text
    assert register.refs == {}
    receipt = json.loads(client.turns[-1][0].tool_results["call_1"])["value"]
    assert receipt["status"] == ("failed" if failure else "empty")
    if failure:
        assert receipt["error_type"] == "rate_limited" and receipt["retry_after"] == 3
    assert all(ref.kind != "web_search" for ref in result.artifacts)


def test_failed_page_fetch_does_not_create_page_or_observation_record(tmp_path):
    register = Registration(tmp_path)
    client = NativePlanClient([
        {"tool": "web_fetch", "arguments": {"url": URL}}, finish(),
    ])
    result = ScientificAgent(
        client, web_page_fetcher=PageFetcher(error=WebFetchError(
            "unsupported PDF", error_type="unsupported_content_type",
        )), registration_port=register,
    ).invoke(request())

    assert result.status == "completed", result.report
    assert register.refs == {}
    trace = next(item for item in result.artifacts if item.kind == "observation_trace")
    assert json.loads(trace.content)["observed_artifact_ids"] == []
    assert "unsupported_content_type" in json.dumps(client.turns[-1], default=lambda x: x.model_dump())


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_direct_search_uses_service_trace_and_preserves_scientific_model_budget(tmp_path, monkeypatch, budget):
    import httpx
    from resagent2_components import DeepSeekWebSearchBackend
    from resagent2_runtime import ServiceRequestClient

    monkeypatch.setenv("TEST_SEARCH_KEY", "test-only-secret")
    calls = []

    def send(wire_request, **kwargs):
        calls.append(wire_request)
        return httpx.Response(200, request=wire_request, json={
            "results": [{
                "title": "Source", "url": URL, "snippet": "Short excerpt",
                "summary": "Provider summary", "content": "", "rerank_score": 2.4,
                "authority": 3, "date": None, "site_name": "Example", "site_icon": None,
            }],
        })

    monkeypatch.setattr("resagent2_runtime.service_request.send_request", send)
    register = Registration(tmp_path / "artifacts")
    direct = ServiceRequestClient(
        endpoint="https://api.test/v1/tools/search", api_key_env="TEST_SEARCH_KEY",
        trace_dir=tmp_path / "trace", trace_level="full",
    )
    client = NativePlanClient([
        {"tool": "web_search", "arguments": {"query": "find source"}},
        finish(),
    ])
    agent = ScientificAgent(
        client, web_search_backend=DeepSeekWebSearchBackend(direct), registration_port=register,
    )
    result = agent.invoke(request(budget=budget))

    # One model decision requests search; the service itself spends no model call.
    # A second model decision is still needed to assess the receipt and finish.
    assert result.llm_calls == min(budget, 2)
    assert len(calls) == 1
    trace, = [json.loads(line) for line in (tmp_path / "trace" / "service_requests.jsonl").read_text().splitlines()]
    assert trace["run_id"] == RUN_ID
    assert trace["session_id"] == result.session.id
    assert trace["tool"] == "web_search" and trace["provider"] == "deepseek"
    assert trace["request_kind"] == "external_service"
    assert trace["protocol"] == "json-service-request/v1"
    assert "model" not in trace and "usage" not in trace
    assert "test-only-secret" not in json.dumps(trace)
    assert not (tmp_path / "trace" / "llm_traces.jsonl").exists()
    assert json.loads(trace["request_text"]) == {
        "query": "find source", "count": 10, "need_content": False,
    }
    receipt = json.loads(client.turns[-1][0].tool_results["call_1"])["value"] if client.turns[-1] else None
    if receipt is None:
        session = agent.store.load(result.session.id)
        receipt = json.loads(session.tool_turns[0].tool_results["call_1"])["value"]
    assert receipt["provider"] == "deepseek"
    assert receipt["results"][0]["url"] == URL
    assert receipt["results"][0]["summary"] == "Provider summary"
    assert (result.status == "completed") == (budget >= 2)


@pytest.mark.parametrize("incomplete_reason", [None, "max_uses_exceeded"])
def test_scientific_follows_later_source_directly_from_search_receipt(tmp_path, incomplete_reason):
    register, fetcher = Registration(tmp_path), PageFetcher()
    backend = SearchBackend(results=[
        WebSearchItem("First lead", "https://example.test/first", ""),
        WebSearchItem("Later source", URL, ""),
    ], incomplete_reason=incomplete_reason)
    client = NativePlanClient([
        {"tool": "web_search", "arguments": {"query": "sources"}},
        {"tool": "web_fetch", "arguments": {"url": URL}},
        lambda: read(register.of_kind("web_page")),
        lambda: finish(register.of_kind("web_page")),
    ])
    result = ScientificAgent(
        client, web_search_backend=backend, web_page_fetcher=fetcher,
        registration_port=register, literature_backend=UnusedLiteratureBackend(),
    ).invoke(request())
    assert result.status == "completed", result.report
    first = json.loads(client.turns[1][0].tool_results["call_1"])["value"]
    assert first["result_count"] == 2 and "omitted_count" not in first
    assert first["incomplete_reason"] == incomplete_reason
    assert first["status"] == ("partial" if incomplete_reason else "results")
    assert [item["url"] for item in first["results"]] == ["https://example.test/first", URL]
    assert backend.calls == ["sources"] and fetcher.calls == [URL]
    assert {ref.kind for ref in register.refs.values()} == {"web_page"}
    assert "Original page detail" not in client.contexts[-1].material_index
    assert "artifact_reads" not in client.contexts[-1].included_sections
    assert "Original page detail" in json.dumps(client.turns[-1], default=lambda x: x.model_dump())
