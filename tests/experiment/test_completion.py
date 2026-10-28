"""Experiment completion accepts analysis and preserves actual failure evidence.

Exact metric/path requirements are checked at Scheduler artifact acceptance.
"""

from datetime import UTC, datetime
import json

import pytest

from resagent2_contracts import AgentOwner, ArtifactCandidate, ArtifactRef, WorkspaceGrant, WorkspaceAccess, WorkspaceSourceKind
from resagent2_components import WorkspaceBoundary
from resagent2_runtime import AgentState, FinishCandidate, ToolObservation
from resagent2_runtime.models import NativeToolCall, ToolCallTurn
from resagent2_runtime.tool_calling import tool_receipt
from resagent2_experiment.completion import ExperimentCompletionCheck


def state():
    now = datetime.now(UTC)
    return AgentState(
        session_id="session_test", agent_name="experiment", owner=AgentOwner.EXECUTOR,
        run_id="run_test", task_id="task_test", attempt_number=1,
        created_at=now, updated_at=now,
    )


def check(root):
    return ExperimentCompletionCheck(WorkspaceBoundary(WorkspaceGrant(root=str(root), source=WorkspaceSourceKind.LOCAL, access=WorkspaceAccess(read_paths=['.'], write_paths=[]))))


def evidence(path="metrics.json", **extra):
    return ArtifactCandidate(
        kind="experiment_result", path=path, media_type="application/json",
        summary="Experimental measurements", **extra,
    )


def command_receipt(current, tool, value, *, ok):
    call_id = f"command_{len(current.tool_turns) + 1}"
    current.tool_turns.append(ToolCallTurn(
        tool_calls=[NativeToolCall(id=call_id, name=tool,
            arguments=json.dumps({"command": value.get("command", "python train.py")}))],
        tool_results={call_id: tool_receipt(ToolObservation(
            summary="Actual command receipt", ok=ok, value=value))},
    ))


def test_analysis_does_not_require_execution_or_new_evidence(tmp_path):
    decision = check(tmp_path).evaluate(state(), FinishCandidate(report="Existing results are inconclusive"))
    assert decision.complete
    assert decision.artifacts == []


@pytest.mark.parametrize("filename", ["metrics.json", "./metrics.json"])
def test_authorized_preexisting_file_can_be_analyzed(tmp_path, filename):
    (tmp_path / "metrics.json").write_text('{"accuracy": 0.9}')
    decision = check(tmp_path).evaluate(state(), FinishCandidate(
        report="Analyzed the existing measurement", artifacts=[evidence(filename)],
    ))
    assert decision.complete
    assert decision.artifacts[0].path == filename


def test_report_preserves_limits_without_claiming_measurement(tmp_path):
    report = "Prior result lacks repeated seeds, so uncertainty is unknown."
    decision = check(tmp_path).evaluate(state(), FinishCandidate(
        report=report, artifacts=[evidence(content='{"analysis": "uncertainty unknown"}')],
    ))
    assert decision.complete
    assert decision.report == report
    assert decision.artifacts[0].content == '{"analysis": "uncertainty unknown"}'


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_missing_file_is_not_delivered(tmp_path, status):
    decision = check(tmp_path).evaluate(state(), FinishCandidate(status=status, report="Done", artifacts=[evidence()]))
    assert not decision.complete
    assert decision.failure is None
    assert decision.artifacts == []
    assert "artifact_path_missing" in decision.report
    assert "metrics.json" in decision.report


def test_output_directory_file_is_checked_without_requiring_writable_source(tmp_path):
    workspace = tmp_path / "source"
    output = tmp_path / "out"
    workspace.mkdir()
    output.mkdir()
    (output / "metrics.json").write_text('{"accuracy": 0.9}')
    finalizer = check(workspace)
    finalizer.output_dir = output
    candidate = FinishCandidate(report="Measured", artifacts=[evidence()])
    assert finalizer.evaluate(state(), candidate).complete
    (output / "metrics.json").unlink()
    (tmp_path / "outside.json").write_text("{}")
    (output / "metrics.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(PermissionError):
        finalizer.evaluate(state(), candidate)


def test_report_does_not_self_certify_command_failure(tmp_path):
    decision = check(tmp_path).evaluate(state(), FinishCandidate(report="It failed"))
    assert decision.complete
    assert decision.failure is None


@pytest.mark.parametrize("exit_code,timed_out", [(1, False), (-9, True)])
@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("tool_name", ["run_shell"])
def test_finish_preserves_verified_execution_without_inferred_failure(tmp_path, exit_code, timed_out, status, tool_name):
    current = state()
    command_receipt(current, tool_name, {
        "command": "python train.py", "exit_code": exit_code, "timed_out": timed_out,
        "stdout_path": "out.stdout", "stderr_path": "out.stderr", "stderr_tail": "real error",
        "duration_seconds": 0.1,
    }, ok=False)
    candidate = FinishCandidate(status=status, report="Training failed")
    decision = check(tmp_path).evaluate(current, candidate)
    assert decision.complete
    assert decision.failure is None
    record = json.loads(decision.artifacts[0].content)["results"][0]
    assert record["exit_code"] == exit_code
    assert record["timed_out"] is timed_out
    assert record["stderr_path"] == "out.stderr"
    assert json.loads(current.tool_turns[0].tool_results["command_1"])["value"]["stderr_tail"] == "real error"

    invalid = candidate.model_copy(update={"artifacts": [evidence("missing.json")]})
    rejected = check(tmp_path).evaluate(current, invalid)
    assert not rejected.complete
    assert rejected.failure is None
    assert "artifact_path_missing" in rejected.report


@pytest.mark.parametrize("tool_name", ["run_shell"])
def test_unexecuted_failure_text_is_not_command_evidence(tmp_path, tool_name):
    current = state()
    command_receipt(current, tool_name, {"blocked": True, "reason": "No environment"}, ok=False)
    decision = check(tmp_path).evaluate(current, FinishCandidate(status="failed", report="Cannot execute"))
    assert decision.complete
    assert decision.artifacts == []


def test_installation_and_experiment_execution_facts_are_delivered_in_receipt_order(tmp_path):
    current = state()
    commands = [("run_shell", "python -m pip install -r requirements.txt", 1),
                ("run_shell", "python -m pip install numpy", 0),
                ("run_shell", "python train.py", 0)]
    for sequence, (tool, command, exit_code) in enumerate(commands, start=1):
        command_receipt(current, tool, {
            "command": command, "exit_code": exit_code, "timed_out": False,
            "stdout_path": f"{sequence}.stdout", "stderr_path": f"{sequence}.stderr",
            "duration_seconds": 0.1,
        }, ok=exit_code == 0)
    decision = check(tmp_path).evaluate(current, FinishCandidate(report="Recorded all outcomes"))
    assert decision.complete
    records = json.loads(decision.artifacts[0].content)["results"]
    assert [(row["command"], row["exit_code"]) for row in records] == [
        (command, exit_code) for _, command, exit_code in commands
    ]
    assert [row["stdout_path"] for row in records] == ["1.stdout", "2.stdout", "3.stdout"]


def test_programmatic_registered_output_receives_correctable_feedback(tmp_path):
    ref = ArtifactRef(id="artifact_child", kind="data", producer="executor", run_id="run_test",
        task_id="task_child", attempt_number=1, uri="file:///unread-child.json",
        sha256="a" * 64, media_type="application/json", summary="Child evidence")
    decision = check(tmp_path).evaluate(state(), FinishCandidate(report="Done", artifacts=[ref]))
    assert not decision.complete and decision.failure is None
    assert "existing_artifact_as_output" in decision.report
    assert "report" in decision.report
