"""Deterministic checks for shared application state across multiple Runs."""

from e2e.native_fixtures import tool_turns

from resagent2_contracts import AgentPermissions, RunPermissions, ExecutionLimits

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from resagent2_components import (
    ArtifactReadError,
    LiteraturePaper,
    LiteratureSearchResult,
    RegisteredArtifactReader,
    ResourceLayout,
)
from resagent2_contracts import (
    AgentOwner,
    ArtifactCandidate,
    ArtifactRef,
    WorkflowAgentKind,
    WorkflowAgentRegistry,
    ModuleStatus,
    AgentRequest,
    ResearchRequest,
    RunBudget,
    RunStatus,
    SessionId,
    RecordedAnswer,
    TaskBudget,
    WorkspaceGrant,
    WorkspaceAccess,
    WorkspaceSourceKind,
    task_session_id,
)
from resagent2_coding import NativeCodingAgent
from resagent2_experiment import NativeExperimentAgent
from resagent2_orchestrator import (
    DeterministicWorkInterpreter,
    ArtifactRegistry, InMemoryRunStore, JsonRunStore, ResearchController, ResearchRun,
    ScientificArtifactRegistration, WorkflowScheduler,
)
from resagent2_runtime import JsonSessionStore, ScriptedLLMClient
from resagent2_scientific import ScientificAgent


def _research():
    return ResearchRequest(goal='Check Run isolation', budget=RunBudget(max_llm_calls=10, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=2, max_attempts_per_task=1))


def test_task_session_identity_is_bounded_and_unambiguous():
    identities = [
        ("run_a", "task_shared", 1),
        ("run_b", "task_shared", 1),
        ("run_a", "task_shared", 2),
        ("run_a_task_b", "task_c", 1),
        ("run_a", "task_b_task_c", 1),
        ("run_" + "a" * 128, "task_" + "b" * 128, 1),
        ("run_" + "a" * 127 + "b", "task_" + "b" * 128, 1),
    ]
    names = [task_session_id(*identity) for identity in identities]
    assert len(set(names)) == len(identities)
    for identity, name in zip(identities, names):
        assert name == task_session_id(*identity)
        assert TypeAdapter(SessionId).validate_python(name) == name
        assert len(name) <= 136


def test_long_run_id_uses_one_valid_scientific_session_through_controller(tmp_path):
    run_id = "run_" + "a" * 128
    session_id = task_session_id(run_id, "task_root", 1)
    assert TypeAdapter(SessionId).validate_python(session_id) == session_id
    assert session_id != task_session_id("run_" + "a" * 127 + "b", "task_root", 1)
    short_session_id = task_session_id("run_short", "task_root", 1)
    assert TypeAdapter(SessionId).validate_python(short_session_id) == short_session_id
    assert short_session_id != session_id

    class UnusedCompiler:
        def compile(self, *args, **kwargs):
            pytest.fail("direct conclusion must not compile a workflow")

    sessions = JsonSessionStore(tmp_path / "sessions")
    agent = ScientificAgent(ScriptedLLMClient(tool_turns([
        {"tool": "finish", "arguments": {"report": "Scientific conclusion", "artifacts": [{"kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json", "summary": "Scientific conclusion", "content": {"verdict": "inconclusive", "statement": "No evidence requested"}}]}},
    ])), store=sessions)
    scheduler = WorkflowScheduler(
        bindings={}, store=JsonRunStore(tmp_path / "runs"),
        artifact_root=tmp_path / "artifacts",
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=agent, compiler=UnusedCompiler(), scheduler=scheduler,
        registry=WorkflowAgentRegistry(),
    )
    run = controller.create_run(run_id, _research())
    assert run.status == RunStatus.COMPLETED
    assert run.scientific_session.id == session_id
    assert sessions.load(session_id).run_id == run_id


@pytest.mark.parametrize("agent_type", [NativeCodingAgent, NativeExperimentAgent])
def test_native_agents_share_store_without_cross_run_session_collision(
    tmp_path, monkeypatch, agent_type
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
         "commit", "--allow-empty", "-qm", "baseline"],
        cwd=workspace, check=True,
    )
    ask = {
        "tool": "ask_user",
        "arguments": {
            "text": "Which option?",
            "requested_fields": ["option"],
        },
    }
    store = JsonSessionStore(tmp_path / "sessions")
    agent = agent_type(
        ScriptedLLMClient(tool_turns([ask, ask, ask])), store=store,
        resource_layout=ResourceLayout(resource_root=tmp_path / "resources"),
    )
    coding = agent_type is NativeCodingAgent
    request = AgentRequest(work_request_id="work_direct", run_id='run_a', task_id='task_shared', attempt_number=1, agent=AgentOwner.EXECUTOR, instruction='Ask first', budget=TaskBudget(max_llm_calls=5, timeout_seconds=30), workspace=WorkspaceGrant(root=str(workspace), source=WorkspaceSourceKind.LOCAL, access=WorkspaceAccess(read_paths=['.'], write_paths=['.'])), workspace_id='ws_test', output_dir=str(tmp_path / 'output'), permissions=AgentPermissions(execute_commands=True))
    first = agent.invoke(request)
    second = agent.invoke(request.model_copy(update={"run_id": "run_b"}))
    assert first.status == second.status == ModuleStatus.NEEDS_USER_INPUT
    assert first.session.id != second.session.id
    assert store.load(first.session.id).run_id == "run_a"
    assert store.load(second.session.id).run_id == "run_b"

    # A normal pause/resume retains the original Session and Attempt.
    resumed = agent.invoke(request.model_copy(update={"parent_session_id": first.session.id}))
    assert resumed.status == ModuleStatus.NEEDS_USER_INPUT
    assert resumed.session.id == first.session.id
    assert store.load(first.session.id).attempt_number == 1
    assert len(list((tmp_path / "sessions").glob("*.json"))) == 2


def _artifact(tmp_path):
    path = tmp_path / "private.txt"
    content = b"private evidence"
    path.write_bytes(content)
    return ArtifactRef(
        id="artifact_private", kind="literature_paper", producer=AgentOwner.SCIENTIFIC,
        run_id="run_a", session_id="session_a", uri=path.as_uri(),
        sha256=hashlib.sha256(content).hexdigest(), media_type="text/plain",
        summary="private evidence",
    )


@pytest.mark.parametrize("live", [False, True])
def test_reader_rejects_other_run_before_reading_bytes(tmp_path, monkeypatch, live):
    artifact = _artifact(tmp_path)
    reader = RegisteredArtifactReader(
        [] if live else [artifact], run_id="run_b",
        resolve=(lambda _: artifact) if live else None,
    )
    def forbidden_read(_path):
        pytest.fail("a foreign Run artifact must be rejected before file I/O")
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(ArtifactReadError, match="unknown artifact"):
        reader.read_text(artifact.id)


def test_reader_rejects_resolver_returning_a_different_id(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    reader = RegisteredArtifactReader([], run_id="run_a", resolve=lambda _: artifact)
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("must not read"))
    with pytest.raises(ArtifactReadError, match="unknown artifact"):
        reader.read_text("artifact_not_granted")


def test_scientific_does_not_observe_another_runs_live_artifact(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)

    class WrongResolver:
        def resolve(self, artifact_id, *, run_id, session_id):
            assert run_id == "run_b"
            assert session_id == task_session_id("run_b", "task_root", 1)
            return artifact  # Even a misbehaving injected resolver cannot bypass the reader.

    client = ScriptedLLMClient(tool_turns([
        {"tool": "read_artifact", "arguments": {"artifact_id": artifact.id}},
        {"tool": "finish", "arguments": {"report": "Scientific conclusion", "artifacts": [{"kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json", "summary": "Scientific conclusion", "content": {"verdict": "inconclusive", "statement": "Evidence unavailable"}}]}},
    ]))
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("must not read"))
    agent = ScientificAgent(client, registration_port=WrongResolver())
    result = agent.invoke(AgentRequest(work_request_id="work_direct", task_id="task_root", attempt_number=1, agent=AgentOwner.SCIENTIFIC, run_id='run_b', instruction='Check Run isolation', budget=TaskBudget(max_llm_calls=3, timeout_seconds=30), permissions=AgentPermissions(execute_commands=True)))
    assert result.status == "completed"
    assert json.loads(next(item.content for item in result.artifacts if item.kind == "observation_trace"))["observed_artifact_ids"] == []
    assert "private evidence" not in client.contexts[-1].text


@pytest.fixture(params=["memory", "json"])
def registration(tmp_path, request):
    store = (
        InMemoryRunStore() if request.param == "memory"
        else JsonRunStore(tmp_path / "runs")
    )
    now = datetime.now(UTC)
    for run_id in ("run_a", "run_b"):
        store.save(ResearchRun(
            run_id=run_id, request=_research(), status=RunStatus.RUNNING,
            created_at=now, updated_at=now,
        ))
    return ScientificArtifactRegistration(ArtifactRegistry(tmp_path / "artifacts"), store)


def test_live_registration_binds_identity_and_content_to_run_scope(registration):
    candidate = ArtifactCandidate(
        kind="literature_paper", path="paper.json", media_type="application/json",
        summary="shared source", metadata={"paper": _paper().model_dump(mode="json")},
    )
    first = registration.register_scientific(candidate, run_id="run_a", session_id="session_a",
                                               operation_key="session:session_a:turn:0:call:fetch", output_position=0)
    assert registration._store.load("run_a").artifacts[first.id] == first
    assert first.session_id == "session_a"
    assert registration.resolve(first.id, run_id="run_a", session_id="session_a") is None
    registration.acknowledge_registration(run_id="run_a", operation_key="session:session_a:turn:0:call:fetch")
    assert registration.resolve(first.id, run_id="run_b", session_id="session_a") is None
    assert registration.resolve(first.id, run_id="run_a", session_id="session_other") is None
    second = registration.register_scientific(candidate, run_id="run_b", session_id="session_b",
                                                operation_key="session:session_b:turn:0:call:fetch", output_position=0)
    registration.acknowledge_registration(run_id="run_b", operation_key="session:session_b:turn:0:call:fetch")
    assert first.id != second.id
    assert registration._store.load("run_b").artifacts[second.id] == second
    assert first.uri != second.uri
    assert registration.resolve(first.id, run_id="run_a", session_id="session_a") == first
    assert registration.resolve(second.id, run_id="run_b", session_id="session_b") == second


def _paper():
    return LiteraturePaper(
        paper_id="2401.00001", title="Run-scoped literature source",
        abstract="The registered abstract remains available across a pause.",
        source_url="https://arxiv.org/abs/2401.00001",
    )


def _paper_ref(registration):
    return next(ref for ref in registration.list_artifacts(run_id="run_a", session_id=task_session_id("run_a", "task_root", 1))
                if ref.kind == "literature_paper")


def test_scientific_reads_new_literature_in_the_same_turn(registration):
    class Backend:
        def resolve(self, source):
            assert source == _paper().fetch_source
            return _paper()

        def search(self, query, **kwargs):
            return LiteratureSearchResult(
                papers=[_paper()], source="arxiv", executed_query=query,
                page=kwargs["page"], next_page=None, total_results=1,
            )

    class Client:
        tool_session_key = "test-native-tools/v1"

        step = 0
        artifact_id = None

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            self.step += 1
            if self.step == 1:
                return tool_turns([{"tool": "literature_search", "arguments": {"query": "scope test"}}])[0]
            if self.step == 2:
                assert registration.list_artifacts(run_id="run_a", session_id=task_session_id("run_a", "task_root", 1)) == []
                assert _paper().abstract in json.loads(turns[-1].tool_results[turns[-1].tool_calls[0].id])["value"]["papers"][0]["abstract"]
                return tool_turns([{"tool": "fetch_literature", "arguments": {"source": _paper().fetch_source}}])[0]
            if self.step == 3:
                self.artifact_id = _paper_ref(registration).id
                return tool_turns([{"tool": "read_artifact", "arguments": {"artifact_id": self.artifact_id}}])[0]
            assert self.step == 4
            assert "artifact_reads" not in context.included_sections
            receipt = json.loads(turns[-1].tool_results[turns[-1].tool_calls[0].id])
            assert receipt["value"]["artifact_id"] == self.artifact_id
            assert _paper().abstract in receipt["value"]["content"]
            directory = json.loads(context.material_index.split("\n", 1)[1])
            registered = next(node for node in directory["children"] if node["name"] == "registered_materials")
            literature = next(node for node in registered["children"] if node["name"] == "literature")
            assert [node["location"]["artifact_id"] for node in literature["children"]] == [self.artifact_id]
            assert directory["artifact_count"] == 1
            assert _paper().abstract not in context.material_index
            return tool_turns([{"tool": "finish", "arguments": {"report": "Scientific conclusion", "artifacts": [{"kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json", "summary": "Scientific conclusion", "content": {"verdict": "inconclusive", "statement": "Only abstract-level evidence is available",
                            "evidence_artifact_ids": [self.artifact_id]}}]}}])[0]

    client = Client()
    agent = ScientificAgent(client, literature_backend=Backend(), registration_port=registration)
    result = agent.invoke(AgentRequest(work_request_id="work_direct", task_id="task_root", attempt_number=1, agent=AgentOwner.SCIENTIFIC, run_id='run_a', instruction='Check Run isolation', budget=TaskBudget(max_llm_calls=5, timeout_seconds=30), permissions=AgentPermissions(execute_commands=True)))
    assert result.status == "completed", result.model_dump(mode="json")
    state = agent.store.load(result.session.id)
    assert state.memory["read_artifact_ids"] == [client.artifact_id]
    reads = [e for e in state.events if e.type == "observation" and e.tool == "read_artifact"]
    assert len(reads) == 1
    assert _paper().abstract in reads[0].data["value"]["content"]
    assert "read_artifact_summaries" not in state.memory
    assert registration.resolve(client.artifact_id, run_id="run_b", session_id=task_session_id("run_a", "task_root", 1)) is None


def test_resumed_scientific_does_not_return_historical_input_refs(registration):
    class Backend:
        def resolve(self, source):
            assert source == _paper().fetch_source
            return _paper()

        def search(self, query, **kwargs):
            return LiteratureSearchResult(
                papers=[_paper()], source="arxiv", executed_query=query,
                page=kwargs["page"], next_page=None, total_results=1,
            )

    class Client:
        tool_session_key = "test-native-tools/v1"

        step = 0

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            self.step += 1
            if self.step == 1:
                return tool_turns([{"tool": "literature_search", "arguments": {"query": "history"}}])[0]
            if self.step == 2:
                return tool_turns([{"tool": "fetch_literature", "arguments": {"source": _paper().fetch_source}}])[0]
            if self.step == 3:
                return tool_turns([{"tool": "read_artifact", "arguments": {"artifact_id": _paper_ref(registration).id}}])[0]
            if self.step == 4:
                return tool_turns([{"tool": "ask_user", "arguments": {
                    "assessment": {"statement": "Need permission to conclude"},
                    "text": "Proceed with this evidence?", "requested_fields": ["approve"],
                }}])[0]
            assert self.step == 5
            assert any(_paper().abstract in receipt for turn in turns for receipt in turn.tool_results.values())
            return tool_turns([{"tool": "finish", "arguments": {
                "report": "Only abstract-level evidence is available",
                "artifacts": [{
                    "kind": "scientific_opinion", "path": "opinion.json",
                    "media_type": "application/json", "summary": "Conclusion",
                    "content": {
                        "verdict": "inconclusive", "statement": "Only abstract-level evidence is available",
                        "evidence_artifact_ids": [_paper_ref(registration).id],
                    },
                }],
            }}])[0]

    agent = ScientificAgent(Client(), literature_backend=Backend(), registration_port=registration)
    request = AgentRequest(work_request_id="work_direct", task_id="task_root", attempt_number=1, run_id='run_a', agent=AgentOwner.SCIENTIFIC, instruction='Review evidence', budget=TaskBudget(max_llm_calls=5, timeout_seconds=30), permissions=AgentPermissions(execute_commands=True))
    first = agent.invoke(request)
    assert first.status == "needs_user_input", first.model_dump(mode="json")
    historical_refs = [item for item in first.artifacts if isinstance(item, ArtifactRef)]
    assert {ref.kind for ref in historical_refs} == {"literature_paper"}
    ref = next(item for item in historical_refs if item.kind == "literature_paper")
    question = first.control.question
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text, requested_fields=question.requested_fields,
        options=question.options, values={"approve": "yes"}, answered_at=datetime.now(UTC),
        run_id=question.run_id, session_id=question.session_id,
        task_id=question.task_id, attempt_number=question.attempt_number,
    )
    second = agent.invoke(request.model_copy(update={
        "parent_session_id": first.session.id, "input_artifacts": historical_refs,
        "user_answers": [answer], "resume_answers": [answer], "control_ack": first.control.origin,
        "input_ids": [answer.question_id],
    }))
    assert second.status == "completed", second.report
    assert second.session.id == first.session.id
    historical_ids = {item.id for item in historical_refs}
    assert not any(isinstance(item, ArtifactRef) and item.id in historical_ids for item in second.artifacts)
    trace = json.loads(next(item.content for item in second.artifacts if item.kind == "observation_trace"))
    assert trace["observed_artifact_ids"] == [ref.id]
    reads = [event for event in agent.store.load(second.session.id).events
             if event.type == "observation" and event.tool == "read_artifact"]
    assert len(reads) == 1  # Resume retains the actual historical read, not a search-as-read flag.
    assert reads[0].data["value"]["artifact_id"] == ref.id
