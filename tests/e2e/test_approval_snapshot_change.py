"""Approval drift settles the original native call before a new proposal."""

from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest

from e2e.native_fixtures import tool_turns
from resagent2_capabilities import DeletePathTool, RunShellInput
from resagent2_capabilities.permissions import OperationPermissionPolicy
from resagent2_components import ProcessRunner, WorkspaceBoundary
from resagent2_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, RecordedAnswer, TaskBudget,
    WorkspaceAccess, WorkspaceGrant,
)
from resagent2_orchestrator.handoffs import control_key, receive_control
from resagent2_runtime import (
    AgentDefinition, AgentLoop, CompletionDecision, FinishTool, JsonSessionStore,
    ScriptedLLMClient, ToolObservation,
)


class Shell:
    name = "run_shell"
    input_model = RunShellInput

    def __init__(self, runner):
        self.runner = runner
        self.extra_env = {"PATH": "/original/environment/bin"}
        self.calls = []

    def prepare(self, arguments):
        return self.approval_context()

    def approval_context(self):
        return self.runner.shell_context(extra_env=self.extra_env)

    def execute_prepared(self, state, arguments, prepared):
        assert prepared == self.approval_context()
        self.calls.append(arguments.command)
        return ToolObservation(summary="Executed approved command", value={"command": arguments.command})

    def execute(self, state, arguments):
        raise AssertionError("Shell must consume the approved startup snapshot")


class Completion:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None, report="Finished")


@pytest.mark.parametrize("changed_target", ["shell_environment", "recursive_delete"])
def test_answered_snapshot_change_does_not_reissue_control_for_same_native_call(tmp_path, changed_target):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    req = AgentRequest(work_request_id="work_direct",
        run_id="run_approval_drift", task_id="task_approval_drift", attempt_number=1,
        agent=AgentOwner.EXECUTOR, instruction="Work within the reviewed scope",
        budget=TaskBudget(max_llm_calls=8, timeout_seconds=60),
        permissions=AgentPermissions(execute_commands=True),
        workspace=WorkspaceGrant(root=str(workspace), source="local",
            access=WorkspaceAccess(read_paths=["."], write_paths=["."])),
    )
    boundary = WorkspaceBoundary(req.workspace)
    if changed_target == "shell_environment":
        tool = Shell(ProcessRunner(boundary))
        action = {"tool": "run_shell", "arguments": {"command": "printf reviewed"}}
    else:
        target = workspace / "cache"
        target.mkdir()
        (target / "old.txt").write_text("old")
        tool = DeletePathTool(boundary)
        action = {"tool": "delete_path", "arguments": {"path": "cache", "recursive": True}}
    store = JsonSessionStore(tmp_path / "sessions")

    def invoke(request, actions):
        tools = (tool, FinishTool())
        client = ScriptedLLMClient(tool_turns(actions))
        definition = AgentDefinition(
            name="approval_drift", owner=AgentOwner.EXECUTOR, system_prompt="Test",
            tools=tools, llm_client=client, context_builder=lambda *_: [],
            permission_policy=OperationPermissionPolicy(tools, boundary=boundary, request=request),
            completion_check=Completion(),
        )
        return AgentLoop(store=store).run(definition, request, session_id="session_approval_drift"), client

    run = SimpleNamespace(run_id=req.run_id, received_controls={}, child_questions={},
                          runtime_notices=[], parent_attention=False, pending_question=None)
    first, _ = invoke(req, [action])
    receive_control(run, req, first)
    original_key = control_key(first.control)
    question = first.control.question
    comment = "  Approval applies only to the original snapshot.\n原文保留  "
    recorded = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, answered_at=datetime.now(UTC), comment=comment,
        run_id=req.run_id, session_id=first.session.id, task_id=req.task_id,
        attempt_number=req.attempt_number, action=question.action,
    )
    resumed = req.model_copy(update={"parent_session_id": first.session.id,
        "resume_answers": [recorded], "control_ack": first.control.origin})
    run.pending_question = None
    if changed_target == "shell_environment":
        tool.extra_env["PATH"] = "/changed/environment/bin"
    else:
        (target / "new.txt").write_text("new")

    # The next model call may propose the same text as a NEW native call. The
    # old operation must first receive not_executed, preserving its comment.
    second, client = invoke(resumed, [action])
    receive_control(run, resumed, second)
    assert second.status == "needs_user_input"
    assert control_key(second.control) != original_key
    assert len(run.received_controls) == 2
    assert run.received_controls[original_key] == question.id
    assert run.pending_question.id == second.control.question.id
    assert len(client.histories) == 1
    state = store.load(first.session.id)
    turn = state.tool_turns[first.control.source_turn]
    original_receipt = json.loads(turn.tool_results[first.control.source_call_id])
    assert original_receipt["ok"] is False
    assert original_receipt["value"]["execution_status"] == "not_executed"
    assert original_receipt["user_comment"] == comment
    assert "new tool call" in original_receipt["summary"]
    assert client.histories[0][first.control.source_turn].tool_results[first.control.source_call_id] == turn.tool_results[first.control.source_call_id]
    assert second.control.question.action.action_id != question.action.action_id
    assert all(control.question.id == question.id for control in state.control_history
               if control.source_turn == first.control.source_turn)
    if changed_target == "shell_environment":
        assert tool.calls == []
    else:
        assert sorted(p.name for p in target.iterdir()) == ["new.txt", "old.txt"]
