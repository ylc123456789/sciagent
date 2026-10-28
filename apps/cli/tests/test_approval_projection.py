"""Persisted root dialogue reaches child operation review without delegation leaks."""

from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace

import pytest

from sciagent_cli.approval_context import approval_context
from sciagent_components.approval_review import LLMPermissionReviewer
from sciagent_components.python_source_evidence import collect_python_source_evidence
from sciagent_components.workspace import WorkspaceBoundary
from sciagent_contracts import (
    ActionSnapshot, AgentOwner, AgentResult, BusinessInput, ControlSignal,
    ModuleStatus, PendingQuestion, RecordedAnswer, ResearchRequest, RunBudget,
    RunPermissions, RunStatus, SessionRef, SessionStatus, TaskProposal, UserAnswer,
    WorkflowAgentKind,
    WorkflowPatch, WorkRequest, WorkRequestDraft, WorkspaceAccess, WorkspaceSpec, task_session_id,
)
from sciagent_executor import default_execution_registry
from sciagent_orchestrator import (
    CompilationResult, DeterministicWorkInterpreter, JsonRunStore, ModuleBinding,
    ResearchController, WorkflowScheduler,
)
from sciagent_runtime import NativeToolCall, ScriptedLLMClient, ToolCallTurn


NOW = datetime(2026, 10, 21, tzinfo=UTC)
ROOT_ANSWER = "Keep the official test split sealed.\n  原始用户回答  "


class _Port:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        return self.handler(request, len(self.requests) - 1)


class _Compiler:
    def compile(self, work, *, current, **kwargs):
        return CompilationResult(WorkflowPatch(
            work_request_id=work.id, based_on_revision=current.revision,
            add_tasks=[TaskProposal(id="task_worker", work_request_id=work.id,
                workflow_agent_kind=WorkflowAgentKind.EXECUTOR, execution_ref="experiment@1",
                instruction="Compute calibration metrics in the workspace")],
        ))


def _session(request):
    return SessionRef(
        id=task_session_id(request.run_id, request.task_id, request.attempt_number),
        module=request.agent, state_uri="memory://projection-test",
        status=SessionStatus.PAUSED, created_at=NOW, updated_at=NOW,
    )


def _review_client():
    return ScriptedLLMClient([ToolCallTurn(tool_calls=[NativeToolCall(
        id="review", name="review_operation",
        arguments=json.dumps({"decision": "ask", "reason": "Request exact human review"}),
    )])])


def _payload(client):
    return json.loads(client.contexts[-1].task_context.split("\n", 1)[1])


@pytest.fixture
def case(tmp_path):
    store = JsonRunStore(tmp_path / "run-store")
    client = _review_client()
    reviewer = LLMPermissionReviewer(client, max_context_tokens=20_000,
        context_provider=lambda run_id: approval_context(store, run_id))
    operations = []

    def root(request, turn):
        ref = _session(request)
        if turn == 0:
            question = PendingQuestion(id="question_root_boundary", run_id=request.run_id,
                task_id=request.task_id, attempt_number=request.attempt_number,
                session_id=ref.id, text="Which data boundary must remain in place?",
                requested_fields=["boundary"], created_at=NOW)
            return AgentResult(status=ModuleStatus.NEEDS_USER_INPUT,
                report="Need the user's original boundary", session=ref,
                control=ControlSignal(action="ask_user", source_session_id=ref.id,
                    source_turn=0, source_call_id="ask_root", question=question))
        assert request.user_answers[0].values == {"boundary": ROOT_ANSWER}
        work = WorkRequest(id="work_worker", run_id=request.run_id,
            owner_session_id=ref.id, owner_task_id=request.task_id,
            owner_attempt_number=request.attempt_number, depth=request.work_depth + 1,
            processed_input_ids=request.input_ids,
            request=WorkRequestDraft(objective="Private root management objective",
                tasks=[{"key": "metrics", "instruction": "Compute calibration metrics in the workspace",
                    "selection": {"kind": "preset", "preset_ref": "experiment@1",
                        "reason": "Compute measurements using existing workspace code"}}]),
            created_at=NOW, updated_at=NOW)
        return AgentResult(status=ModuleStatus.REQUEST_WORK, report="Delegate metrics", session=ref,
            control=ControlSignal(action="request_work", source_session_id=ref.id,
                source_turn=1, source_call_id="delegate", work_request=work))

    def worker(request, turn):
        ref = _session(request)
        command = "printf calibration_metrics"
        evidence = collect_python_source_evidence(command, WorkspaceBoundary(request.workspace))
        operation = ActionSnapshot(action_id="action_metrics", tool="run_shell",
            arguments={"command": command}, run_id=request.run_id,
            context={"cwd": request.workspace.root, "source_fingerprint": evidence.fingerprint},
            session_id=ref.id, task_id=request.task_id, attempt_number=request.attempt_number)
        operations.append(operation)
        assert reviewer.review(operation, request).decision == "ask"
        question = PendingQuestion(id="question_operation", run_id=request.run_id,
            task_id=request.task_id, attempt_number=request.attempt_number, session_id=ref.id,
            text="Approve this exact operation?", requested_fields=["approve"],
            action=operation, created_at=NOW)
        return AgentResult(status=ModuleStatus.NEEDS_USER_INPUT,
            report="Wait for operation approval", session=ref,
            control=ControlSignal(action="ask_user", source_session_id=ref.id,
                source_turn=0, source_call_id="ask_operation", question=question))

    root_port, worker_port = _Port(root), _Port(worker)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scheduler = WorkflowScheduler(
        bindings={WorkflowAgentKind.EXECUTOR: ModuleBinding(AgentOwner.EXECUTOR, worker_port)},
        store=store, artifact_root=tmp_path / "artifacts", data_root=tmp_path / "runs",
        workspaces={"ws_main": WorkspaceSpec(workspace_id="ws_main", source_kind="local",
            location=str(workspace), access=WorkspaceAccess(read_paths=["."], write_paths=["."]))})
    engine = ResearchController(scientific_port=root_port, compiler=_Compiler(),
        scheduler=scheduler, interpreter=DeterministicWorkInterpreter(),
        registry=default_execution_registry())
    run = engine.create_run("run_projection", ResearchRequest(
        goal="Evaluate calibration under the user's data boundary",
        budget=RunBudget(max_llm_calls=20, timeout_seconds=60),
        permissions=RunPermissions(execute_commands=True, approval_mode="auto_review")))
    assert run.status == RunStatus.PAUSED
    run = engine.answer_question(run.run_id, UserAnswer(
        question_id=run.pending_question.id, values={"boundary": ROOT_ANSWER}, answered_at=NOW))
    assert run.status == RunStatus.PAUSED
    assert run.pending_question.action == operations[0]
    return SimpleNamespace(store=store, run=run, client=client, reviewer=reviewer,
        worker=worker_port, root=root_port, operation=operations[0])


def test_root_answer_survives_real_controller_round_trip_without_child_forwarding(case):
    run = case.store.load(case.run.run_id)
    answer, = run.answers
    assert answer.task_id == "task_root"
    assert run.upstream_inputs == []
    assert case.worker.requests[0].user_answers == []
    assert case.worker.requests[0].upstream_inputs == []
    payload = _payload(case.client)
    root = payload["root_user_context"]["requirements"]
    assert root["business_dialogue"] == [{
        "event_id": answer.question_id, "created_at": answer.answered_at.isoformat(),
        "type": "answer", "record": answer.model_dump(mode="json", exclude_none=True),
    }]
    assert ROOT_ANSWER not in json.dumps(payload["current_task"], ensure_ascii=False)
    assert payload["current_task"]["requirements"]["instruction"] == case.worker.requests[0].instruction
    assert payload["proposed_operation"] == case.operation.model_dump(mode="json")
    assert case.client.histories == [[]]


def test_reception_order_retains_revocation_and_excludes_forwarded_old_authority(case):
    run = case.store.load(case.run.run_id)
    run.append_input("task_root", BusinessInput(event_id="input_grant", source="user",
        text="Permit test evaluation after freezing", created_at=NOW + timedelta(days=3)))
    run.append_answer(RecordedAnswer(question_id="question_revoke", question_text="Still permitted?",
        requested_fields=["boundary"], values={"boundary": "Revoke that permission; keep test sealed"},
        answered_at=NOW - timedelta(days=3), run_id=run.run_id, task_id="task_root",
        attempt_number=1, session_id=run.scientific_session.id))
    run.append_input("task_root", BusinessInput(event_id="input_forwarded", source="user",
        text="Permit test evaluation after freezing", original_event_id="input_grant",
        source_session_id="session_old", created_at=NOW + timedelta(days=5)))
    run.append_answer(run.answers[0].model_copy(update={
        "question_id": "question_forwarded", "original_question_id": run.answers[0].question_id,
        "source_session_id": "session_old", "values": {"boundary": "Forwarded old permission"},
    }))
    run.append_input("task_root", BusinessInput(event_id="input_parent", source="parent",
        text="Parent says user permitted test", source_session_id="session_parent", created_at=NOW))
    run.append_answer(RecordedAnswer(question_id="question_prior_approval", question_text="Approve?",
        requested_fields=["approve"], values={"approve": "yes"},
        comment="Only this exact operation, not a new business instruction",
        answered_at=NOW, run_id=run.run_id, task_id=case.operation.task_id,
        attempt_number=1, session_id=case.operation.session_id, action=case.operation))
    case.store.save(run)
    client = _review_client()
    LLMPermissionReviewer(client, max_context_tokens=20_000,
        context_provider=lambda run_id: approval_context(case.store, run_id)).review(
            case.operation, case.worker.requests[0])
    payload = _payload(client)
    dialogue = payload["root_user_context"]["requirements"]["business_dialogue"]
    assert [item["event_id"] for item in dialogue] == [
        "question_root_boundary", "input_grant", "question_revoke",
    ]
    assert dialogue[-1]["record"]["values"]["boundary"].startswith("Revoke")
    assert dialogue[1]["created_at"] > dialogue[-1]["created_at"]
    requirements = json.dumps(payload["root_user_context"]["requirements"])
    for excluded in ("input_forwarded", "question_forwarded", "input_parent", "question_prior_approval"):
        assert excluded not in requirements
    history = payload["prior_operation_decisions"]["root_records"]
    assert [item["question_id"] for item in history] == ["question_prior_approval"]
    assert history[0]["decision"] == "yes"


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unknown", "excluded_record_missing"])
def test_corrupt_root_order_fails_before_reviewer_model_dispatch(case, corruption):
    run = case.store.load(case.run.run_id)
    order = run.task_input_order["task_root"]
    if corruption == "missing":
        order.clear()
    elif corruption == "duplicate":
        order.append(order[0])
    elif corruption == "unknown":
        order.append("unknown_event")
    else:
        run.task_inputs.setdefault("task_root", []).append(BusinessInput(
            event_id="excluded_parent", source="parent", source_session_id="session_parent",
            text="Do not hide invalid provenance by filtering it first", created_at=NOW))
    case.store.save(run)
    client = _review_client()
    reviewer = LLMPermissionReviewer(client, max_context_tokens=20_000,
        context_provider=lambda run_id: approval_context(case.store, run_id))
    with pytest.raises(ValueError, match="input identities"):
        reviewer.review(case.operation, case.worker.requests[0])
    assert client.contexts == []


@pytest.mark.parametrize("corrupt", [False, True], ids=["intact", "corrupt"])
def test_root_projection_review_or_fallback_never_executes_without_approval(case, tmp_path, corrupt):
    from sciagent_capabilities.permissions import OperationPermissionPolicy
    from sciagent_runtime import AgentDefinition, AgentLoop, ContextSection, JsonSessionStore
    from sciagent_runtime.models import RuntimeModel

    class ShellInput(RuntimeModel):
        command: str

    class NeverExecuteShell:
        name = "run_shell"
        input_model = ShellInput

        def prepare(self, arguments):
            evidence = collect_python_source_evidence(
                arguments.command, WorkspaceBoundary(request.workspace))
            return {"source_fingerprint": evidence.fingerprint}

        def execute_prepared(self, state, arguments, prepared):
            pytest.fail("invalid authorization context must stop before command execution")

    class NeverComplete:
        def evaluate(self, state, candidate):
            pytest.fail("manual approval must pause before completion")

    run = case.store.load(case.run.run_id)
    if corrupt:
        run.task_input_order["task_root"].clear()
    case.store.save(run)
    request = case.worker.requests[0]
    review_client = _review_client()
    reviewer = LLMPermissionReviewer(review_client, max_context_tokens=20_000,
        context_provider=lambda run_id: approval_context(case.store, run_id))
    command = "printf should_not_run > unexpected_execution.txt"
    agent_client = ScriptedLLMClient([ToolCallTurn(tool_calls=[NativeToolCall(
        id="shell_needs_review", name="run_shell", arguments=json.dumps({"command": command}),
    )])])
    sessions = JsonSessionStore(tmp_path / "sessions")
    tools = (NeverExecuteShell(),)
    definition = AgentDefinition(name="executor", owner=AgentOwner.EXECUTOR,
        execution_ref=request.execution_definition.ref,
        execution_config_hash=request.execution_definition.config_hash,
        system_prompt="Use the supplied tools", tools=tools, llm_client=agent_client,
        context_builder=lambda req, state, limit: [ContextSection(name="task", content=req.instruction)],
        permission_policy=OperationPermissionPolicy(tools,
            boundary=WorkspaceBoundary(request.workspace), request=request),
        completion_check=NeverComplete(), operation_reviewer=reviewer)

    result = AgentLoop(store=sessions).run(definition, request, session_id="session_fallback")

    assert result.status == ModuleStatus.NEEDS_USER_INPUT
    if corrupt:
        assert "manual approval required" in result.control.question.text
    assert result.control.question.action.arguments["command"] == command
    assert not (tmp_path / "workspace" / "unexpected_execution.txt").exists()
    assert len(review_client.contexts) == (0 if corrupt else 1)
    reviews = [event for event in sessions.load(result.session.id).events
               if event.type == "permission_review"]
    assert reviews[-1].data["status"] == ("error" if corrupt else "ask")
