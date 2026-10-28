"""The same public artifact preserves true command outcomes on every exit path."""

from datetime import UTC, datetime
import json

import pytest

from sciagent_components import execution_record, preserve_execution_record
from sciagent_contracts import AgentOwner, AgentResult, ErrorCode, ModuleError, SessionStatus, VerificationResult
from sciagent_runtime import AgentState, InMemorySessionStore, ToolObservation
from sciagent_runtime.models import HistoryCheckpoint, NativeToolCall, ToolCallTurn
from sciagent_runtime.tool_calling import tool_receipt


def state_and_receipt(*, value, tool="run_shell"):
    now = datetime.now(UTC)
    state = AgentState(session_id="session_commands", agent_name="coding", owner=AgentOwner.EXECUTOR,
        run_id="run_commands", task_id="task_commands", attempt_number=1, created_at=now, updated_at=now)
    state.tool_turns.append(ToolCallTurn(
        tool_calls=[NativeToolCall(id="call_command", name=tool, arguments='{"command":"proposed"}')],
        tool_results={"call_command": tool_receipt(ToolObservation(summary="Actual runner outcome", value=value))},
    ))
    return state


def completed(command="python check.py", code=0):
    return VerificationResult(command=command, exit_code=code, timed_out=False,
                stdout_path="actual.stdout", stderr_path="actual.stderr", duration_seconds=0.25).model_dump(mode="json")


@pytest.mark.parametrize("confirmed", [True, False])
def test_interrupted_record_preserves_unknown_fields_without_inventing_success(confirmed):
    facts = dict(operation="process", command="python train.py", pid=123, exit_code=None,
        execution_status="interrupted" if confirmed else "unknown", stop_confirmed=confirmed,
        stdout_path="partial.stdout", duration_seconds=1.5, files_retained=True,
        unconfirmed_processes=[] if confirmed else [123])
    record = execution_record(state_and_receipt(value=facts))
    assert json.loads(record.content) == {"results": [facts]}
    assert "timed_out" not in json.loads(record.content)["results"][0]


@pytest.mark.parametrize("value,tool", [
    ({"execution_status": "not_executed"}, "run_shell"),
    ({"execution_status": "unknown"}, "run_shell"),
    ({"execution_status": "interrupted", "stop_confirmed": True}, "run_shell"),
    ({"execution_status": "interrupted", "stop_confirmed": True}, "ask_user"),
    ({"results": [completed()]}, "read_artifact"),
])
def test_unstarted_commands_model_waits_and_materials_do_not_create_execution(value, tool):
    assert execution_record(state_and_receipt(value=value, tool=tool)) is None




def test_compaction_does_not_remove_original_receipts_from_frozen_record():
    state = state_and_receipt(value=completed(code=1))
    state.tool_turns.append(ToolCallTurn(runtime_notice="Later model interaction"))
    state.history_checkpoint = HistoryCheckpoint(history_start=1, summary="Old command summary")
    assert json.loads(execution_record(state).content)["results"][0] == completed(code=1)


def test_record_keeps_completed_command_before_later_interruption():
    first = completed("python first.py")
    state = state_and_receipt(value=first)
    facts = dict(operation="process", command="python second.py", pid=234,
        execution_status="interrupted", stop_confirmed=True, files_retained=True)
    turn = state_and_receipt(value=facts).tool_turns[0]
    state.tool_turns.append(turn)
    assert json.loads(execution_record(state).content) == {"results": [first, facts]}


def test_record_keeps_observed_startup_and_code_without_validation_verdict():
    facts = {**completed(), "cwd": "/workspace", "shell": "/bin/bash",
        "shell_args": ["--noprofile", "--norc", "-o", "pipefail", "-c"],
        "pipefail": True, "environment_fingerprint": "actual-startup-fingerprint",
        "code_before": "first-tree", "code_after": "second-tree", "changed_paths": ["check.py"]}
    record = json.loads(execution_record(state_and_receipt(value=facts)).content)["results"][0]
    assert record == facts
    assert "passed" not in record and "environment_certified" not in record


def test_record_keeps_completed_script_when_later_git_stop_is_unconfirmed():
    first = completed("python actual.py")
    diagnostic = dict(operation="process", command=["git", "diff"],
        execution_status="unknown", stop_confirmed=False, unconfirmed_processes=[123])
    state = state_and_receipt(value={**diagnostic, "completed_command": first})
    assert json.loads(execution_record(state).content) == {"results": [first, diagnostic]}


@pytest.mark.parametrize("status,session_status", [("completed", "completed"), ("failed", "failed"),
                                                  ("interrupted", "paused"), ("yielded", "paused")])
def test_terminal_agent_result_reuses_same_record_and_is_idempotent(status, session_status):
    state = state_and_receipt(value=completed())
    state.status = SessionStatus(session_status)
    store = InMemorySessionStore()
    store.save(state)
    session = dict(id=state.session_id, module="executor", state_uri="session://session_commands",
        status=session_status, created_at=state.created_at, updated_at=state.updated_at)
    error = ModuleError(code=ErrorCode.TOOL_FAILED, message="Original failure", retryable=False) if status == "failed" else None
    result = AgentResult(status=status, report="Original report", session=session, error=error)
    attached = preserve_execution_record(result, store)
    assert attached.report == result.report and attached.error == result.error and attached.status == result.status
    assert len(attached.artifacts) == 1
    assert preserve_execution_record(attached, store) == attached
    assert json.loads(attached.artifacts[0].content) == {"results": [completed()]}


@pytest.mark.parametrize("status,session_status", [("failed", "failed"), ("interrupted", "paused")])
def test_record_diagnostic_error_preserves_original_failure_or_stop_status(status, session_status):
    state = state_and_receipt(value=completed())
    error = ModuleError(code=ErrorCode.INTERRUPTED, message="Stop unconfirmed", retryable=False,
        details={"stop_confirmed": False}) if status == "failed" else None
    result = AgentResult(status=status, report="Original outcome", error=error, session=dict(
        id=state.session_id, module="executor", state_uri="session://session_commands", status=session_status,
        created_at=state.created_at, updated_at=state.updated_at))
    restored = preserve_execution_record(result, InMemorySessionStore())
    assert restored.status == result.status and restored.report == result.report
    if error:
        assert restored.error.details["stop_confirmed"] is False
        assert "execution_record_error" in restored.error.details
    else:
        assert restored.warnings[0].code == "execution_record_unavailable"


def test_unconfirmed_timeout_fact_survives_frozen_execution_record():
    facts = dict(operation="process", command="python train.py", pid=123, exit_code=None,
        execution_status="unknown", stop_confirmed=False, timed_out=True,
        stdout_path="partial.stdout", stderr_path="partial.stderr", duration_seconds=1.5,
        files_retained=True, unconfirmed_processes=[123])
    record = execution_record(state_and_receipt(value=facts))
    assert json.loads(record.content) == {"results": [facts]}
