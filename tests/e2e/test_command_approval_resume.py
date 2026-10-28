"""Native Agent resumes execute exactly the approved shell operation once."""

from e2e.native_fixtures import tool_turns

from datetime import UTC, datetime
import json
import subprocess
import sys
import shlex
from types import SimpleNamespace

import pytest

from sciagent_coding import NativeCodingAgent
from sciagent_experiment import NativeExperimentAgent
from sciagent_components import ResourceLayout
from sciagent_contracts import (
    AgentOwner, AgentPermissions, AgentRequest,
    RecordedAnswer, TaskBudget, WorkspaceAccess, WorkspaceGrant,
)
from sciagent_orchestrator.artifacts import ArtifactRegistry
from sciagent_runtime import JsonSessionStore, ScriptedLLMClient


FINISH = {"tool": "finish", "arguments": {"report": "Recorded the actual outcome"}}


@pytest.fixture(params=["coding", "experiment"])
def case(tmp_path, monkeypatch, request):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("*.marker\n__pycache__/\n")
    (repo / "measure.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "with Path(sys.argv[1] + '.marker').open('a') as f:\n"
        "    f.write('executed\\n')\n"
    )
    (repo / "test_markers.py").write_text(
        "from pathlib import Path\nimport unittest\n"
        "class Markers(unittest.TestCase):\n"
        "    def test_first(self):\n"
        "        with Path('first.marker').open('a') as f: f.write('executed\\n')\n"
        "    def test_second(self):\n"
        "        with Path('second.marker').open('a') as f: f.write('executed\\n')\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "baseline"], cwd=repo, check=True,
    )

    kind = request.param
    owner = AgentOwner.EXECUTOR
    req = AgentRequest(work_request_id="work_direct",
        run_id="run_approval", task_id="task_approval", attempt_number=1,
        agent=owner, instruction="Execute first then second, with separate approval",
        budget=TaskBudget(max_llm_calls=30, timeout_seconds=60),
        permissions=AgentPermissions(execute_commands=True),
        workspace=WorkspaceGrant(
            root=str(repo), source="local",
            access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
        ),
        workspace_id="ws_main", output_dir=str(tmp_path / "output"),
    )
    return SimpleNamespace(
        root=tmp_path, repo=repo, request=req, kind=kind,
        agent_type=NativeCodingAgent if kind == "coding" else NativeExperimentAgent,
    )


def command(case, tag):
    return {"tool": "run_shell", "arguments": {
        "command": f"{shlex.quote(sys.executable)} measure.py {tag}",
    }}


def invoke(case, req, actions, *, reviewer=None):
    # Recreate both Agent and SessionStore on every invoke, as controller resume does.
    case.last_client = ScriptedLLMClient(tool_turns(actions))
    agent = case.agent_type(
        case.last_client, store=JsonSessionStore(case.root / "sessions"),
        resource_layout=ResourceLayout(resource_root=case.root / "resources"),
        operation_reviewer=reviewer,
    )
    return agent.invoke(req)


def test_auto_review_runs_actual_shell_commands_once_in_both_agents(case):
    from sciagent_components.approval_review import LLMPermissionReviewer
    from sciagent_runtime.budget import execution_budget

    req = case.request.model_copy(update={
        "permissions": AgentPermissions(execute_commands=True, approval_mode="auto_review"),
    })
    client = ScriptedLLMClient(tool_turns([
        {"tool": "review_operation", "arguments": {"decision": "allow", "reason": "First marker is requested"}},
        {"tool": "review_operation", "arguments": {"decision": "allow", "reason": "Second marker is requested"}},
    ]))
    reviewer = LLMPermissionReviewer(client, max_context_tokens=6000)
    with execution_budget(max_llm_calls=5, timeout_seconds=60) as budget:
        result = invoke(case, req, [command(case, "first"), command(case, "second"), FINISH], reviewer=reviewer)
        assert budget.usage.used == 5  # Three Agent turns and two reviews, one wallet.
    assert result.status == "completed", result.report
    assert (case.repo / "first.marker").read_text() == "executed\n"
    assert (case.repo / "second.marker").read_text() == "executed\n"
    state = JsonSessionStore(case.root / "sessions").load(result.session.id)
    assert state.pending_action is None
    receipts = [json.loads(turn.tool_results[call.id]) for turn in state.tool_turns
                for call in turn.tool_calls if call.name == "run_shell"]
    assert len(receipts) == 2
    assert all(receipt["ok"] and receipt["permission_review"]["status"] == "allow" for receipt in receipts)
    reviews = [event for event in state.events if event.type == "permission_review"
               and event.data["status"] == "allow"]
    assert len(reviews) == 2
    assert reviews[0].data["operation"]["action_id"] != reviews[1].data["operation"]["action_id"]
    assert state.llm_calls_used == result.llm_calls == 5


def approve(case, req, result, *, value="yes", comment=None):
    question = result.control.question
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": value}, comment=comment, answered_at=datetime.now(UTC),
        run_id=req.run_id, session_id=result.session.id, task_id=req.task_id, attempt_number=req.attempt_number,
        action=question.action,
    )
    return req.model_copy(update={
        "parent_session_id": result.session.id,
        "resume_answers": [answer], "control_ack": result.control.origin,
    })


@pytest.mark.parametrize("decision", ["yes", "no"])
@pytest.mark.parametrize("comment", [None, "", "  保留原文\nPlease follow the task boundary.  "])
def test_operation_comment_reaches_model_once_without_becoming_task_requirements(
    case, decision, comment,
):
    first = invoke(case, case.request, [command(case, "first")])
    assert first.status == "needs_user_input"
    assert not (case.repo / "first.marker").exists()
    call_id = first.control.source_call_id
    resumed = approve(case, case.request, first, value=decision, comment=comment)
    result = invoke(case, resumed, [FINISH])
    assert result.status == "completed", result.report

    state = JsonSessionStore(case.root / "sessions").load(result.session.id)
    assert state.pending_action is None
    turns = [turn for turn in state.tool_turns if call_id in turn.tool_results]
    assert len(turns) == 1
    receipt_text = turns[0].tool_results[call_id]
    receipt = json.loads(receipt_text)
    history = case.last_client.histories[0]
    assert sum(call_id in turn.tool_results for turn in history) == 1
    assert next(turn.tool_results[call_id] for turn in history
                if call_id in turn.tool_results) == receipt_text
    if comment is None:
        assert "user_comment" not in receipt
    else:
        assert receipt["user_comment"] == comment
    if decision == "yes":
        assert receipt["ok"] is True
        assert (case.repo / "first.marker").read_text() == "executed\n"

    else:
        assert receipt["ok"] is False
        assert receipt["value"]["execution_status"] == "not_executed"
        assert not (case.repo / "first.marker").exists()

    for context in case.last_client.contexts:
        assert "user_comment" not in context.task_context
        assert "user_comment" not in context.material_index
        if comment:
            assert "保留原文" not in context.task_context + context.material_index



@pytest.mark.parametrize("second_tag", ["first", "second"])
def test_native_commands_resume_with_single_use_approval(case, second_tag):
    first = invoke(case, case.request, [command(case, "first")])
    assert first.status == "needs_user_input"

    assert list(case.repo.glob("*.marker")) == []

    resumed = approve(case, case.request, first)
    second = invoke(case, resumed, [command(case, second_tag)])
    assert second.status == "needs_user_input"

    assert (case.repo / "first.marker").read_text() == "executed\n"
    assert not (case.repo / "second.marker").exists()
    first_action = first.control.question.action
    pending = JsonSessionStore(case.root / "sessions").load(second.session.id).pending_action
    assert pending.action_id != first_action.action_id

    final = invoke(
        case, approve(case, resumed, second), [FINISH],
    )
    assert final.status == "completed"

    expected = "executed\n" * (2 if second_tag == "first" else 1)
    assert (case.repo / "first.marker").read_text() == expected
    if second_tag == "second":
        assert (case.repo / "second.marker").read_text() == "executed\n"
    state = JsonSessionStore(case.root / "sessions").load(final.session.id)
    assert state.pending_action is None
    executed = [
        event.data for event in state.events
        if event.type == "observation"
        and event.tool == "run_shell"
        and isinstance(event.data.get("value"), dict)
        and "exit_code" in event.data["value"]
    ]
    assert len(executed) == 2
    assert all(event["value"]["exit_code"] == 0 for event in executed)


@pytest.mark.parametrize("case", ["experiment"], indirect=True)
@pytest.mark.parametrize("mode", ["validate", "all"])
def test_pipeline_module_runs_once_after_approval(case, mode):
    package = case.repo / "pipeline"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "run.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "with Path(sys.argv[1] + '.marker').open('a') as f:\n"
        "    f.write('executed\\n')\n"
    )
    action = {"tool": "run_shell", "arguments": {"command": f"{shlex.quote(sys.executable)} -m pipeline.run {mode}"}}

    first = invoke(case, case.request, [action])
    assert first.status == "needs_user_input"

    assert not (case.repo / f"{mode}.marker").exists()

    result = invoke(case, approve(case, case.request, first), [FINISH])
    assert result.status == "completed", result.report

    assert (case.repo / f"{mode}.marker").read_text() == "executed\n"
    state = JsonSessionStore(case.root / "sessions").load(result.session.id)
    assert state.pending_action is None
    assert sum(1 for event in state.events if event.tool == "run_shell"
               and event.type == "observation"
               and "exit_code" in (event.data.get("value") or {})) == 1


def test_legacy_shell_approval_requires_a_new_call_before_execution(case, monkeypatch):
    import sciagent_components.process as process

    action = command(case, "first")
    with monkeypatch.context() as legacy:
        legacy.setattr(process, "_BASH_ARGV", tuple(arg for arg in process._BASH_ARGV if arg != "-e"))
        first = invoke(case, case.request, [action])
    assert first.status == "needs_user_input"
    assert "-e" not in first.control.question.action.context["shell_args"]
    assert not (case.repo / "first.marker").exists()

    comment = "The original approval was for the previous shell flags."
    resumed = approve(case, case.request, first, comment=comment)
    second = invoke(case, resumed, [action])
    assert second.status == "needs_user_input"
    assert "-e" in second.control.question.action.context["shell_args"]
    assert second.control.source_call_id != first.control.source_call_id
    assert second.control.question.action.action_id != first.control.question.action.action_id
    assert not (case.repo / "first.marker").exists()
    state = JsonSessionStore(case.root / "sessions").load(second.session.id)
    old_call_id = first.control.source_call_id
    receipts = [json.loads(turn.tool_results[old_call_id]) for turn in state.tool_turns
                if old_call_id in turn.tool_results]
    assert len(receipts) == 1
    assert not receipts[0]["ok"]
    assert receipts[0]["value"]["execution_status"] == "not_executed"
    assert receipts[0]["user_comment"] == comment
    assert "snapshot changed" in receipts[0]["summary"].lower()

    final = invoke(case, approve(case, resumed, second), [FINISH])
    assert final.status == "completed", final.report
    assert (case.repo / "first.marker").read_text() == "executed\n"
    state = JsonSessionStore(case.root / "sessions").load(final.session.id)
    executed = [json.loads(turn.tool_results[call.id]) for turn in state.tool_turns
                for call in turn.tool_calls
                if call.name == "run_shell" and call.id in turn.tool_results
                and "exit_code" in json.loads(turn.tool_results[call.id]).get("value", {})]
    assert len(executed) == 1
    assert executed[0]["value"]["exit_code"] == 0
    assert "-e" in executed[0]["value"]["shell_args"]


def test_failed_shell_keeps_approval_comment_and_real_exit_code(case):
    action = {"tool": "run_shell", "arguments": {
        "command": f"{shlex.quote(sys.executable)} -c 'import sys; sys.exit(7)'",
    }}
    first = invoke(case, case.request, [action])
    assert first.status == "needs_user_input"
    comment = "批准这次操作；实际失败也需保留说明。"
    result = invoke(case, approve(case, case.request, first, comment=comment), [FINISH])
    state = JsonSessionStore(case.root / "sessions").load(result.session.id)
    call_id = first.control.source_call_id
    receipts = [json.loads(turn.tool_results[call_id]) for turn in state.tool_turns
                if call_id in turn.tool_results]
    assert len(receipts) == 1
    assert receipts[0]["user_comment"] == comment
    assert receipts[0]["ok"] is False
    assert receipts[0]["value"]["exit_code"] == 7
    assert not (case.repo / "first.marker").exists()


def test_process_permission_denies_command_before_effects(case):
    req = case.request.model_copy(update={"permissions": AgentPermissions(
        execute_commands=False,
    )})
    result = invoke(case, req, [command(case, "first"), FINISH])
    assert result.status == "completed"
    assert not (case.repo / "first.marker").exists()


def test_stale_first_answer_cannot_change_second_command_or_session(case):
    from sciagent_contracts import (
        ResearchRequest, RunBudget, RunPermissions, UserAnswer,
        WorkflowAgentRegistry,
    )
    from sciagent_orchestrator import (
        DeterministicWorkInterpreter,
        JsonRunStore, OrchestrationError, ResearchController, ResearchRun,
    )

    first = invoke(case, case.request, [command(case, "first")])
    resumed = approve(case, case.request, first)
    second = invoke(case, resumed, [command(case, "second")])
    assert second.status == "needs_user_input"
    old_question = first.control.question
    draft = second.control.question
    assert draft.action.arguments == command(case, "second")["arguments"]
    now = datetime.now(UTC)
    pending = draft
    run_store = JsonRunStore(case.root / "runs")
    run_store.save(ResearchRun(
        run_id=case.request.run_id, status="paused", pending_question=pending,
        request=ResearchRequest(
            goal=case.request.instruction,
            budget=RunBudget(max_llm_calls=30, timeout_seconds=60),
            permissions=RunPermissions(execute_commands=True),
        ),
        created_at=now, updated_at=now,
    ))

    def forbidden(*args, **kwargs):
        raise AssertionError("Stale answers must never resume execution")

    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=SimpleNamespace(invoke=forbidden),
        compiler=SimpleNamespace(compile=forbidden),
        scheduler=SimpleNamespace(
            store=run_store, bindings={}, resume_task_in_place=forbidden,
            artifact_registry=ArtifactRegistry(case.root / "artifacts"),
        ),
        registry=WorkflowAgentRegistry(),
    )
    files = [*run_store.root.glob("*.json"), *(case.root / "sessions").glob("*.json")]
    before = {path: path.read_bytes() for path in files}
    with pytest.raises(OrchestrationError, match="answer (?:question_id )?does not match pending question"):
        controller.answer_question(
            case.request.run_id,
            UserAnswer(
                question_id=old_question.id,
                values={"approve": "yes"}, answered_at=now,
            ),
        )
    assert {path: path.read_bytes() for path in files} == before

    assert (case.repo / "first.marker").read_text() == "executed\n"
    assert not (case.repo / "second.marker").exists()

