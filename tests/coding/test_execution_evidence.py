"""Coding completion preserves command facts without claiming check sufficiency."""

from datetime import UTC, datetime
import json
import subprocess

import pytest

from sciagent_coding.completion import CodingCompletionCheck
from sciagent_components import GitWorkspace, WorkspaceBoundary, WorkspacePermissionError
from sciagent_contracts import AgentOwner, WorkspaceGrant, WorkspaceAccess
from sciagent_runtime import AgentState, FinishCandidate, NativeToolCall, ToolCallTurn, ToolObservation
from sciagent_runtime.tool_calling import tool_receipt


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("edited_after_check", [False, True])
def test_completion_preserves_checks_and_later_changes_without_synthetic_verdict(
    tmp_path, exit_code, edited_after_check,
):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    boundary = WorkspaceBoundary(WorkspaceGrant(
        root=str(tmp_path), source="local",
        access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
    ))
    repository = GitWorkspace(boundary)
    baseline = repository.snapshot()
    now = datetime.now(UTC)
    state = AgentState(
        session_id="session_checks", agent_name="coding", owner=AgentOwner.EXECUTOR,
        run_id="run_checks", task_id="task_checks", attempt_number=1,
        created_at=now, updated_at=now,
    )
    facts = dict(command="python -m pytest", exit_code=exit_code, timed_out=False,
                 stdout_path="pytest.stdout", stderr_path="pytest.stderr", duration_seconds=0.1)
    state.tool_turns.append(ToolCallTurn(
        tool_calls=[NativeToolCall(id="check", name="run_shell",
                                  arguments='{"command":"python -m pytest"}')],
        tool_results={"check": tool_receipt(ToolObservation(
            summary="Observed check outcome", ok=exit_code == 0, value=facts,
        ))},
    ))
    if edited_after_check:
        source.write_text("VALUE = 2\n")
    report = "Recorded checks; later source changes need their own assessment."
    decision = CodingCompletionCheck(repository, boundary, baseline=baseline,
        output_dir=str(tmp_path / "outputs")).evaluate(
        state, FinishCandidate(report=report),
    )
    assert decision.complete
    assert decision.report == report
    kinds = [item.kind for item in decision.artifacts]
    assert kinds.count("execution_record") == 1
    assert "verification_result" not in kinds
    record = json.loads(next(item.content for item in decision.artifacts
                             if item.kind == "execution_record"))
    assert record["results"][0]["exit_code"] == exit_code
    assert record["results"][0]["command"] == "python -m pytest"
    assert "passed" not in record
    assert "covers_current_workspace" not in record
    assert ("code_patch" in kinds) is edited_after_check
    if edited_after_check:
        assert "VALUE = 2" in (tmp_path / "outputs" / next(
            item.path for item in decision.artifacts if item.kind == "code_patch")).read_text()


@pytest.mark.parametrize("change", ["new", "modified", "deleted"])
def test_unsubmitted_git_changes_still_require_write_scope(tmp_path, change):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "dependency.py"
    if change != "new":
        source.write_text("VALUE = 1\n")
    boundary = WorkspaceBoundary(WorkspaceGrant(
        root=str(tmp_path), source="local",
        access=WorkspaceAccess(read_paths=["."], write_paths=["authorized.py"]),
    ))
    repository = GitWorkspace(boundary)
    baseline = repository.snapshot()
    if change == "deleted":
        source.unlink()
    else:
        source.write_text("VALUE = 2\n")
    now = datetime.now(UTC)
    state = AgentState(
        session_id="session_scope", agent_name="coding", owner=AgentOwner.EXECUTOR,
        run_id="run_scope", task_id="task_scope", attempt_number=1,
        created_at=now, updated_at=now,
    )
    with pytest.raises(WorkspacePermissionError):
        CodingCompletionCheck(repository, boundary, baseline=baseline,
            output_dir=str(tmp_path / "outputs")).evaluate(state, FinishCandidate(report="Finished"))
    assert not (tmp_path / "outputs").exists()
