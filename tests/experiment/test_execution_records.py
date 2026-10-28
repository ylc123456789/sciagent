"""Actual command receipts remain machine-readable across completion outcomes."""

from e2e.native_fixtures import tool_turns

from sciagent_contracts import AgentPermissions, ErrorCode

from datetime import UTC, datetime
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from sciagent_contracts import (
    AgentOwner, AgentRequest, ArtifactCandidate, QuestionDraft, RecordedAnswer,
    TaskBudget, WorkspaceGrant, WorkspaceAccess, WorkspaceSourceKind,
)
from sciagent_components import WorkspaceBoundary
from sciagent_capabilities.permissions import OperationPermissionPolicy
from sciagent_orchestrator import ArtifactRegistry
from sciagent_experiment.completion import ExperimentCompletionCheck
from sciagent_runtime import (
    AgentDefinition, AgentLoop, AgentState,
    FinishCandidate, FinishTool, ScriptedLLMClient, ToolObservation,
)
from sciagent_runtime.models import RuntimeModel


class CommandInput(RuntimeModel):
    """Fixed command fixture."""

    command: str = "python train.py"


class Command:
    name = "run_shell"
    input_model = CommandInput

    def __init__(self, exit_code):
        self.exit_code = exit_code

    def prepare(self, arguments):
        return self.approval_context()

    def approval_context(self):
        return {"shell": "/bin/bash", "execution_backend": "host"}

    def execute_prepared(self, state, arguments, prepared):
        assert all(prepared[key] == value for key, value in self.approval_context().items())
        assert "cwd" in prepared
        return self.execute(state, arguments)

    def execute(self, state, arguments):
        return ToolObservation(
            summary="Real command result", ok=self.exit_code == 0,
            value={
                "command": "python train.py", "exit_code": self.exit_code, "timed_out": False,
                "stdout_path": "stdout.log", "stderr_path": "stderr.log", "duration_seconds": 0.1,
                "stderr_tail": "error" if self.exit_code else "",
            },
        )


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("status", ["completed", "failed"])
def test_loop_preserves_command_facts_independently_of_task_status(tmp_path, exit_code, status):
    boundary = WorkspaceBoundary(WorkspaceGrant(root=str(tmp_path), source=WorkspaceSourceKind.LOCAL, access=WorkspaceAccess(read_paths=['.'], write_paths=['.'])))
    tools = (Command(exit_code), FinishTool())
    req = AgentRequest(work_request_id="work_test",
        run_id="run_records", task_id="task_records", attempt_number=1,
        agent=AgentOwner.EXECUTOR, instruction="Run",
        budget=TaskBudget(max_llm_calls=5, timeout_seconds=10),
        workspace=boundary.grant,
        permissions=AgentPermissions(execute_commands=True),
    )
    definition = AgentDefinition(
        name="experiment", owner=AgentOwner.EXECUTOR, system_prompt="Run experiment",
        tools=tools, llm_client=ScriptedLLMClient(tool_turns([
            {"tool": "run_shell", "arguments": {"command": "python train.py"}},
            {"tool": "finish", "arguments": {"status": status, "report": "Recorded the outcome"}},
        ])),
        context_builder=lambda *_: [],
        permission_policy=OperationPermissionPolicy(
            tools, boundary=boundary, request=req,
        ),
        completion_check=ExperimentCompletionCheck(boundary),
    )
    loop = AgentLoop()
    first = loop.run(definition, req, session_id="session_records")
    assert first.status == "needs_user_input"
    assert not any(item.kind == "execution_record" for item in first.artifacts)
    question = first.control.question
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, answered_at=datetime.now(UTC),
        run_id=req.run_id, task_id=req.task_id, attempt_number=req.attempt_number,
        session_id=first.session.id, action=question.action,
    )
    resumed = req.model_copy(update={
        "parent_session_id": first.session.id,
        "resume_answers": [answer], "control_ack": first.control.origin,
    })
    definition = replace(definition, permission_policy=OperationPermissionPolicy(
        tools, boundary=boundary, request=resumed,
    ))
    result = loop.run(definition, resumed, session_id="session_records")
    assert result.status == result.session.status == status
    record = json.loads(next(item.content for item in result.artifacts if item.kind == "execution_record"))
    assert record["results"][0]["exit_code"] == exit_code
    assert result.report == "Recorded the outcome"
    assert record["results"][0]["stderr_path"] == "stderr.log"
    if status == "failed":
        assert result.error.code == ErrorCode.AGENT_REPORTED_FAILURE
        assert result.error.retryable is False
    else:
        assert result.error is None


@pytest.mark.parametrize("commands", [
    [("python train.py", 1, False), ("python train.py", 0, False)],
    [("python train.py", 1, False), ("python train.py --fixed", 0, False)],
    [("python train.py", 1, False), ("python --version", 0, False)],
    [("python train.py", 0, True), ("python --version", 0, False)],
    [("python train.py", 1, False), ("python train.py --smoke", 0, False)],
    [("python train.py", 1, False), ("python other.py", 1, False),
     ("python train.py", 0, False)],
])
def test_command_history_is_preserved_without_inferred_task_failure(tmp_path, commands):
    from sciagent_runtime.models import NativeToolCall, ToolCallTurn
    from sciagent_runtime.tool_calling import tool_receipt
    now = datetime.now(UTC)
    state = AgentState(
        session_id="session_test", agent_name="experiment", owner=AgentOwner.EXECUTOR,
        run_id="run_test", task_id="task_test", attempt_number=1, created_at=now, updated_at=now,
    )
    for sequence, (command, exit_code, timed_out) in enumerate(commands, start=1):
        observation = Command(exit_code).execute(state, CommandInput())
        observation.value.update(command=command, timed_out=timed_out,
                                 stdout_path=f"{sequence}.stdout", stderr_path=f"{sequence}.stderr")
        call_id = f"command_{sequence}"
        state.tool_turns.append(ToolCallTurn(
            tool_calls=[NativeToolCall(id=call_id, name="run_shell", arguments=json.dumps({"command": command}))],
            tool_results={call_id: tool_receipt(observation)},
        ))
    boundary = WorkspaceBoundary(WorkspaceGrant(root=str(tmp_path), source=WorkspaceSourceKind.LOCAL, access=WorkspaceAccess(read_paths=['.'], write_paths=[])))
    decision = ExperimentCompletionCheck(boundary).evaluate(
        state, FinishCandidate(report="Recorded all outcomes"))
    assert decision.complete
    assert decision.failure is None
    record = json.loads(next(item.content for item in decision.artifacts if item.kind == "execution_record"))
    assert [(row["command"], row["exit_code"], row["timed_out"])
            for row in record["results"]] == commands
