"""Supplemental review reads survive approval and are revalidated before launch."""

from datetime import UTC, datetime
from types import SimpleNamespace
import sys
import json

import pytest

from resagent2_capabilities import RunShellInput, RunShellTool
from resagent2_capabilities.permissions import OperationPermissionPolicy
from resagent2_components import ProcessRunner, WorkspaceBoundary
from resagent2_components.approval_review import LLMPermissionReviewer
from resagent2_contracts import (
    ActionSnapshot, AgentOwner, AgentPermissions, AgentRequest, RecordedAnswer, TaskBudget,
    WorkspaceAccess, WorkspaceGrant,
)
from resagent2_runtime import (
    AgentDefinition, AgentLoop, AgentState, CompletionDecision, FinishTool,
    JsonSessionStore, ScriptedLLMClient,
)
from e2e.native_fixtures import tool_turns


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="actual Linux Bash execution")


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "extra.py").write_text("VALUE = 1\n")
    (root / "lookup").mkdir()
    grant = WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(
        read_paths=["."], write_paths=["."],
    ))
    request = AgentRequest(
        work_request_id="work_binding", run_id="run_binding", task_id="task_binding",
        attempt_number=1, agent=AgentOwner.EXECUTOR, instruction="Run the requested command",
        permissions=AgentPermissions(execute_commands=True),
        budget=TaskBudget(max_llm_calls=10, timeout_seconds=30), workspace=grant,
    )
    boundary = WorkspaceBoundary(grant)
    tool = RunShellTool(
        ProcessRunner(boundary), timeout_seconds=5, log_dir=str(root / ".resagent2" / "shell"),
    )
    arguments = RunShellInput(command="printf executed > executed.marker")
    action = SimpleNamespace(tool="run_shell", arguments=arguments.model_dump(mode="json"))
    now = datetime.now(UTC)
    state = AgentState(
        session_id="session_binding", agent_name="coding", owner=AgentOwner.EXECUTOR,
        run_id=request.run_id, task_id=request.task_id, attempt_number=1,
        created_at=now, updated_at=now,
    )
    return SimpleNamespace(
        root=root, request=request, boundary=boundary, tool=tool,
        arguments=arguments, action=action, state=state,
        policy=OperationPermissionPolicy((tool,), boundary=boundary, request=request),
    )


def _pending(case, context):
    return ActionSnapshot(
        action_id="action_binding", tool="run_shell", arguments=case.action.arguments,
        context=context, run_id=case.request.run_id, session_id=case.state.session_id,
        task_id=case.request.task_id, attempt_number=case.request.attempt_number,
    )


def test_prepare_preserves_inspection_order_without_aliasing_the_callers_list(case):
    inspections = [
        {"kind": "directory", "path": "lookup"},
        {"kind": "file", "path": "extra.py"},
    ]
    prepared = case.tool.prepare(case.arguments, inspections=inspections)
    assert prepared["review_inspections"] == inspections
    assert "review_inspections" not in case.tool.prepare(case.arguments)
    inspections[0]["path"] = "changed"
    assert prepared["review_inspections"][0] == {"kind": "directory", "path": "lookup"}


def test_expected_context_rebuilds_the_fingerprint_instead_of_trusting_it(case):
    inspections = [{"kind": "file", "path": "extra.py"}]
    baseline = case.policy.check(case.action, case.state, case.request)
    expected = case.tool.prepare(case.arguments, inspections=inspections)
    expected["source_fingerprint"] = "0" * 64
    checked = case.policy.check(
        case.action, case.state, case.request, expected_context=expected,
    )
    assert checked.outcome == "ask"
    assert checked.context["review_inspections"] == inspections
    assert checked.context["source_fingerprint"] != expected["source_fingerprint"]
    assert checked.context["source_fingerprint"] != baseline.context["source_fingerprint"]
    assert checked.prepared == checked.context
    assert "review_inspections" not in case.policy.check(
        case.action, case.state, case.request,
    ).context


@pytest.mark.parametrize("changed", [None, "file", "directory"])
def test_matching_human_approval_rechecks_the_same_inspected_files_and_directories(case, changed):
    inspections = [
        {"kind": "file", "path": "extra.py"},
        {"kind": "directory", "path": "lookup"},
    ]
    prepared = case.tool.prepare(case.arguments, inspections=inspections)
    pending = _pending(case, prepared)
    case.state.pending_action = pending
    case.policy.answers.append(SimpleNamespace(
        action=pending, run_id=case.request.run_id, task_id=case.request.task_id,
        attempt_number=case.request.attempt_number, session_id=case.state.session_id,
        values={"approve": "yes"}, comment="Approve the inspected operation",
    ))
    if changed == "file":
        (case.root / "extra.py").write_text("VALUE = 2\n")
    elif changed == "directory":
        (case.root / "lookup" / "new.py").write_text("VALUE = 2\n")

    checked = case.policy.check(case.action, case.state, case.request)
    assert checked.context["review_inspections"] == inspections
    assert checked.outcome == ("ask" if changed else "allow")
    if not changed:
        assert checked.approval_id == pending.action_id
        assert checked.user_comment == "Approve the inspected operation"
    result = case.tool.execute_prepared(case.state, case.arguments, prepared)
    if changed:
        assert not result.ok
        assert result.value["execution_status"] == "not_executed"
        assert not (case.root / "executed.marker").exists()
        assert not (case.root / ".resagent2").exists()
    else:
        assert result.ok
        assert result.value["exit_code"] == 0
        assert (case.root / "executed.marker").read_text() == "executed"


@pytest.mark.parametrize("updates", [
    {"run_id": "another_run"},
    {"session_id": "another_session"},
    {"task_id": "another_task"},
    {"attempt_number": 2},
    {"tool": "another_tool"},
    {"arguments": {"command": "printf another"}},
])
def test_unrelated_pending_operation_cannot_supply_inspection_paths(case, updates):
    prepared = case.tool.prepare(
        case.arguments, inspections=[{"kind": "file", "path": "extra.py"}],
    )
    case.state.pending_action = _pending(case, prepared).model_copy(update=updates)
    checked = case.policy.check(case.action, case.state, case.request)
    assert checked.outcome == "ask"
    assert "review_inspections" not in checked.context
    assert checked.context == case.tool.prepare(case.arguments)


@pytest.mark.parametrize("invalid", [{}, "", 0, False])
def test_invalid_prepared_inspection_values_cannot_be_ignored_at_dispatch(case, invalid):
    prepared = case.tool.prepare(case.arguments)
    prepared["review_inspections"] = invalid
    result = case.tool.execute_prepared(case.state, case.arguments, prepared)
    assert not result.ok
    assert result.value["execution_status"] == "not_executed"
    assert not (case.root / "executed.marker").exists()
    assert not (case.root / ".resagent2").exists()


@pytest.mark.parametrize("source", ["expected", "pending"])
def test_false_valued_inspection_objects_cannot_bypass_policy_validation(case, source):
    expected = {**case.tool.prepare(case.arguments), "review_inspections": {}}
    if source == "pending":
        case.state.pending_action = _pending(case, expected)
    with pytest.raises(ValueError, match="review inspection"):
        case.policy.check(
            case.action, case.state, case.request,
            **({"expected_context": expected} if source == "expected" else {}),
        )
    assert not (case.root / "executed.marker").exists()


class _Complete:
    def evaluate(self, state, candidate):
        return CompletionDecision(
            complete=candidate is not None, report=candidate.report if candidate else "",
        )


def _real_invoke(case, reviewer, *, request=None, actions=None):
    request = request or case.request.model_copy(update={
        "permissions": AgentPermissions(execute_commands=True, approval_mode="auto_review"),
    })
    if not hasattr(case, "store"):
        case.store = JsonSessionStore(case.root.parent / "sessions")
    tools = (case.tool, FinishTool())
    definition = AgentDefinition(
        name="executor", owner=AgentOwner.EXECUTOR, system_prompt="Use the supplied tools.",
        tools=tools, llm_client=ScriptedLLMClient(tool_turns(actions or [
            {"tool": "run_shell", "arguments": case.action.arguments},
            {"tool": "finish", "arguments": {"report": "Recorded the actual outcome"}},
        ])),
        context_builder=lambda *_: [], completion_check=_Complete(),
        permission_policy=OperationPermissionPolicy(tools, boundary=case.boundary, request=request),
        operation_reviewer=reviewer,
    )
    return AgentLoop(store=case.store).run(
        definition, request, session_id=case.state.session_id,
    )


def _inspecting_reviewer(decision):
    client = ScriptedLLMClient(tool_turns([
        {"tool": "review_read_file", "arguments": {"path": "extra.py"}},
        {"tool": "review_operation", "arguments": {
            "decision": decision, "reason": "The supplemental file was inspected",
        }},
    ]))
    return LLMPermissionReviewer(client, max_context_tokens=20_000), client


def _real_receipt(case, result):
    state = case.store.load(result.session.id)
    receipts = [json.loads(turn.tool_results[call.id]) for turn in state.tool_turns
                for call in turn.tool_calls
                if call.name == "run_shell" and call.id in turn.tool_results]
    assert len(receipts) == 1
    return state, receipts[0]


def test_real_reviewer_supplemental_read_binds_auto_allow_to_actual_shell_execution(case):
    reviewer, client = _inspecting_reviewer("allow")
    result = _real_invoke(case, reviewer)
    assert result.status == "completed", result.report
    assert len(client.contexts) == 2
    assert (case.root / "executed.marker").read_text() == "executed"
    state, receipt = _real_receipt(case, result)
    assert receipt["ok"] and receipt["value"]["exit_code"] == 0
    assert receipt["permission_review"]["status"] == "allow"
    review = [event for event in state.events if event.type == "permission_review"][-1]
    expected = case.tool.prepare(
        case.arguments, inspections=[{"kind": "file", "path": "extra.py"}],
    )
    assert review.data["operation"]["context"] == expected
    payload = json.loads(client.contexts[-1].task_context.split("\n", 1)[1])
    initial = json.loads(client.contexts[0].task_context.split("\n", 1)[1])
    assert payload == initial
    assert "review_inspections" not in payload["proposed_operation"]["context"]
    inspection_turn = client.histories[-1][0]
    observation = json.loads(inspection_turn.tool_results[inspection_turn.tool_calls[0].id])
    assert observation["source_fingerprint"] == expected["source_fingerprint"]
    assert observation["inspection"]["status"] == "available"
    assert observation["files"][0]["content"] == (case.root / "extra.py").read_text()


@pytest.mark.parametrize("changed", [False, True])
def test_real_reviewer_ask_resume_preserves_supplemental_binding_and_does_not_rereview(case, changed):
    reviewer, client = _inspecting_reviewer("ask")
    paused = _real_invoke(case, reviewer)
    assert paused.status == "needs_user_input"
    assert len(client.contexts) == 2
    assert not (case.root / "executed.marker").exists()
    request = case.request.model_copy(update={
        "permissions": AgentPermissions(execute_commands=True, approval_mode="auto_review"),
    })
    question = paused.control.question
    assert question.action.context == case.tool.prepare(
        case.arguments, inspections=[{"kind": "file", "path": "extra.py"}],
    )
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, comment="Approve only the inspected version",
        answered_at=datetime.now(UTC), run_id=request.run_id,
        session_id=paused.session.id, task_id=request.task_id, attempt_number=1,
        action=question.action,
    )
    resumed = request.model_copy(update={
        "parent_session_id": paused.session.id, "resume_answers": [answer],
        "control_ack": paused.control.origin,
    })
    if changed:
        (case.root / "extra.py").write_text("VALUE = 2\n")
    result = _real_invoke(case, reviewer, request=resumed, actions=[
        {"tool": "finish", "arguments": {"report": "Recorded the actual outcome"}},
    ])
    assert result.status == "completed", result.report
    assert len(client.contexts) == 2
    _, receipt = _real_receipt(case, result)
    assert receipt["permission_review"]["status"] == "ask"
    assert receipt["user_comment"] == "Approve only the inspected version"
    assert receipt["ok"] == (not changed)
    if changed:
        assert receipt["value"]["execution_status"] == "not_executed"
        assert not (case.root / "executed.marker").exists()
        assert not (case.root / ".resagent2").exists()
    else:
        assert receipt["value"]["exit_code"] == 0
        assert (case.root / "executed.marker").read_text() == "executed"
