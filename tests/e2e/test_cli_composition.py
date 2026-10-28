"""Real E2E scenarios use the production CLI composition without networking."""

import pytest

from e2e import real_e2e
from e2e.native_fixtures import tool_turn
from sciagent_cli import composition
from sciagent_contracts import (
    AgentOwner,
    AgentResult,
    ModuleStatus,
    RunStatus,
    WorkflowAgentKind,
    WorkspaceSourceKind,
)
from sciagent_runtime import JsonSessionStore, ScriptedLLMClient, ServiceRequestClient


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    """Ignore host deployment settings and fail if any test sends HTTP."""
    for suffix in (
        "MODEL", "API_BASE", "API_KEY_ENV", "CONTEXT_WINDOW",
        "RESERVED_OUTPUT_TOKENS", "CONTEXT_SAFETY_MARGIN_TOKENS",
        "LLM_TIMEOUT_SECONDS", "LLM_TRACE_LEVEL", "LLM_TRACE_DIR",
        "CODING_CONTEXT_TOKENS", "EXPERIMENT_CONTEXT_TOKENS",
        "SCIENTIFIC_CONTEXT_TOKENS", "COMPILER_CONTEXT_TOKENS",
        "PERMISSION_REVIEWER_CONTEXT_TOKENS",
        "PDF_PARSE_TIMEOUT_SECONDS", "WEB_SEARCH_MODEL",
        "WEB_SEARCH_TIMEOUT_SECONDS", "WEB_TIMEOUT_SECONDS",
        "WEB_MAX_RESPONSE_BYTES", "DATA_ROOT", "RESOURCE_ROOT",
        "DATASET_ROOT", "ENV_ROOT",
    ):
        monkeypatch.delenv(f"SCIAGENT_{suffix}", raising=False)
    monkeypatch.setenv("SCIAGENT_WEB_SEARCH_PROVIDER", "off")
    monkeypatch.setenv("SCIAGENT_API_KEY_ENV", "E2E_TEST_KEY")
    monkeypatch.setenv("E2E_TEST_KEY", "test-only")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    def no_http(*args, **kwargs):
        pytest.fail("composition tests must not send HTTP requests")

    monkeypatch.setattr("httpx.Client.send", no_http)
    monkeypatch.setattr("httpx.AsyncClient.send", no_http)


def test_e2e_application_applies_production_configuration(tmp_path, monkeypatch):
    """Check deployed values, rather than comparing two calls to one factory."""
    settings = {
        "MODEL": "test-agent-model",
        "API_BASE": "https://model.invalid/v1",
        "CONTEXT_WINDOW": "120000",
        "RESERVED_OUTPUT_TOKENS": "8000",
        "CONTEXT_SAFETY_MARGIN_TOKENS": "1234",
        "LLM_TIMEOUT_SECONDS": "77",
        "CODING_CONTEXT_TOKENS": "12000",
        "EXPERIMENT_CONTEXT_TOKENS": "13000",
        "SCIENTIFIC_CONTEXT_TOKENS": "14000",
        "COMPILER_CONTEXT_TOKENS": "15000",
        "PERMISSION_REVIEWER_CONTEXT_TOKENS": "16000",
        "PDF_PARSE_TIMEOUT_SECONDS": "91",
        "WEB_SEARCH_PROVIDER": "deepseek",
        "WEB_SEARCH_TIMEOUT_SECONDS": "11",
        "WEB_TIMEOUT_SECONDS": "7",
        "WEB_MAX_RESPONSE_BYTES": "12345",
        "LLM_TRACE_LEVEL": "metadata",
        "LLM_TRACE_DIR": str(tmp_path / "traces"),
    }
    for suffix, value in settings.items():
        monkeypatch.setenv(f"SCIAGENT_{suffix}", value)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-search-key")
    repo = tmp_path / "repo"
    repo.mkdir()

    app = real_e2e._application(tmp_path, repo)
    controller = app.controller
    scientific = controller.scientific_port
    executor = controller.scheduler.bindings[WorkflowAgentKind.EXECUTOR].port
    registry = controller.registry
    assert set(controller.scheduler.bindings) == {WorkflowAgentKind.SCIENTIFIC, WorkflowAgentKind.EXECUTOR}
    assert [registry.execution_definitions[ref].config.max_context_tokens for ref in ("coding@1", "experiment@1")] == [12000, 13000]
    assert scientific.max_context_tokens == 14000
    assert not hasattr(controller.compiler, "_client")
    reviewer = executor.operation_reviewer
    assert reviewer._max_context_tokens == 16000
    assert reviewer._context_provider.func is composition.approval_context
    assert reviewer._context_provider.args == (app.run_store,)
    reviewer_client = reviewer._client
    execution_client = executor.model_resolver("default")
    assert reviewer_client is not execution_client and reviewer_client is not scientific.llm_client
    for client in (reviewer_client, execution_client, scientific.llm_client):
        assert client.model == "test-agent-model"
        assert client.endpoint == "https://model.invalid/v1/chat/completions"
        assert client.api_key_env == "E2E_TEST_KEY"
        assert client.timeout_seconds == 77
        assert client.model_profile.context_window == 120000
        assert client.model_profile.reserved_output_tokens == 8000
        assert client.model_profile.safety_margin_tokens == 1234
        assert client.trace_level == "metadata"
        assert client.trace_dir == tmp_path / "traces"

    backend = scientific.web_search_backend
    assert backend.count == 10
    assert isinstance(backend.client, ServiceRequestClient)
    assert backend.client.endpoint == "https://api.deepseek.com/v1/tools/search"
    assert backend.client.api_key_env == "DEEPSEEK_API_KEY"
    assert backend.client.trace_level == "metadata"
    assert backend.client.timeout_seconds == 11
    assert backend.client.max_response_bytes == 12345
    assert backend.client.trace_dir == tmp_path / "traces"
    assert scientific.web_page_fetcher.timeout_seconds == 7
    assert scientific.web_page_fetcher.max_response_bytes == 12345
    assert scientific.literature_parser.keywords["timeout_seconds"] == 91
    assert executor.resource_layout is scientific.resource_layout
    assert controller.dataset_ref_source.dataset_root == scientific.resource_layout.dataset_root.resolve()
    assert scientific.resource_layout.resource_root == tmp_path / "data" / "resources"
    spec = controller.scheduler.workspace_specs["ws_main"]
    assert spec.source_kind == WorkspaceSourceKind.LOCAL
    assert spec.location == str(repo)
    assert spec.access.read_paths == spec.access.write_paths == ["."]
    assert app.run_store.root == tmp_path / "data" / "state"
    assert controller.scheduler.artifact_registry.root == tmp_path / "data" / "artifacts"
    assert scientific.store.root == tmp_path / "data" / "sessions" / "scientific"
    assert executor.loop.store.root == tmp_path / "data" / "sessions" / "executor"


def test_e2e_ask_resume_reopens_production_stores(tmp_path, monkeypatch):
    ask_client = ScriptedLLMClient([tool_turn("ask_user", {
        "assessment": {"statement": "need the user's primary metric"},
        "text": "Which primary metric should be recorded?",
        "requested_fields": ["metric"],
    }, call_id="call_ask")])
    monkeypatch.setattr(composition, "_client", lambda: ask_client)
    assert real_e2e.run_ask_start(tmp_path)

    # Reopen from disk, as ask-resume does in a separate process.
    paused_app = real_e2e._application(tmp_path)
    paused = paused_app.run_store.load("run_ask")
    assert paused.status == RunStatus.PAUSED
    assert paused.usage.used == 1
    initial_requests = dict(paused.usage.requests)
    assert set(initial_requests.values()) == {"succeeded"}
    session_id = paused.scientific_session.id
    session_store = JsonSessionStore(tmp_path / "data" / "sessions" / "scientific")
    before = session_store.load(session_id)
    assert before.llm_calls_used == 1
    assert (tmp_path / "data" / "state" / "run_ask.json").is_file()

    resume_client = ScriptedLLMClient([tool_turn("finish", {
        "report": "The user selected top-1 accuracy; no experiment was requested.",
        "artifacts": [{
            "kind": "scientific_opinion", "path": "opinion.json",
            "media_type": "application/json", "summary": "Recorded metric preference",
            "content": {'verdict': 'inconclusive', 'statement': 'The user selected top-1 accuracy; no measurements are available.'},
        }],
    }, call_id="call_finish")])
    monkeypatch.setattr(composition, "_client", lambda: resume_client)
    assert real_e2e.run_ask_resume(tmp_path, "top-1 accuracy")

    completed = real_e2e._application(tmp_path).run_store.load("run_ask")
    after = session_store.load(session_id)
    assert completed.status == RunStatus.COMPLETED
    assert completed.scientific_session.id == session_id
    assert completed.request.budget == paused.request.budget
    assert completed.usage.used == after.llm_calls_used == 2
    assert all(completed.usage.requests[key] == value for key, value in initial_requests.items())
    assert set(completed.usage.requests.values()) == {"succeeded"}
    assert len(completed.answers) == 1
    assert completed.answers[0].values == {"metric": "top-1 accuracy"}
    assert after.tool_turns[:len(before.tool_turns)] == before.tool_turns
    assert [call.id for turn in after.tool_turns for call in turn.tool_calls] == [
        "call_ask", "call_finish",
    ]
    assert all(set(turn.tool_results) == {call.id for call in turn.tool_calls}
               and turn.executing_call_id is None for turn in after.tool_turns)
    assert resume_client.histories[0] == before.tool_turns
    assert "top-1 accuracy" in resume_client.contexts[0].text
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "scientific_sessions").exists()


@pytest.mark.parametrize("stage,execution_ref", [
    ("run_code", "coding@1"),
    ("run_experiment", "experiment@1"),
])
def test_single_agent_stage_invokes_its_production_binding(tmp_path, monkeypatch, stage, execution_ref):
    monkeypatch.setattr(composition, "_client", lambda: ScriptedLLMClient([]))
    repo = tmp_path / "repo"
    repo.mkdir()
    app = real_e2e._application(tmp_path, repo)
    agent = app.controller.scheduler.bindings[WorkflowAgentKind.EXECUTOR].port
    layout = agent.resource_layout
    layout.dataset_root.mkdir(parents=True)
    (layout.dataset_root / "catalog.json").write_text('{"demo":"demo"}', encoding="utf-8")
    requests = []
    composed = []
    expected = AgentResult(status=ModuleStatus.COMPLETED, report="Invoke spy; no model request")

    def existing_app(workdir, workspace_repo=None):
        composed.append((workdir, workspace_repo))
        return app

    def invoke(request):
        requests.append(request)
        return expected

    monkeypatch.setattr(real_e2e, "_repo", lambda workdir: repo)
    monkeypatch.setattr(real_e2e, "_application", existing_app)
    monkeypatch.setattr(agent, "invoke", invoke)
    assert getattr(real_e2e, stage)(tmp_path) is expected
    assert composed == [(tmp_path, repo)]
    assert len(requests) == 1
    request = requests[0]
    assert request.agent == AgentOwner.EXECUTOR
    assert request.execution_definition.ref == execution_ref
    assert request.execution_definition == app.controller.registry.execution_definitions[execution_ref]
    assert request.permissions.execute_commands
    assert not request.permissions.request_work
    assert request.workspace.root == str(repo)
    assert request.workspace_id == "ws_main"
    assert request.output_dir == str(tmp_path / "out")
    assert request.task_context["dataset_refs"][0]["dataset_id"] == "demo"
    assert request.input_artifacts == []
    assert not app.run_store.exists(request.run_id)
    assert not (tmp_path / "artifacts").exists()
