"""One general execution tool preserves permissions and real command outcomes."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from sciagent_contracts import (
    AgentOwner, VerificationResult, WorkspaceGrant, WorkspaceAccess, WorkspaceSourceKind,
)
from sciagent_components import WorkspaceBoundary
from sciagent_runtime import AgentState
from sciagent_capabilities import RunShellTool


def _state() -> AgentState:
    now = datetime.now(UTC)
    return AgentState(
        session_id="session_test", agent_name="experiment", owner=AgentOwner.EXECUTOR,
        run_id="run_test", task_id="task_test", attempt_number=1,
        created_at=now, updated_at=now,
    )


class FakeRunner:
    def __init__(self, root, *, writable=True, exit_code=0, timed_out=False):
        self.boundary = WorkspaceBoundary(WorkspaceGrant(
            root=str(root), source=WorkspaceSourceKind.LOCAL,
            access=WorkspaceAccess(read_paths=["."], write_paths=["."] if writable else []),
        ))
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.calls = []

    def shell_context(self, *, extra_env=None):
        return {"cwd": str(self.boundary.root), "shell": "/bin/bash"}

    def run_shell(self, command, *, log_dir, index, timeout_seconds, extra_env=None, expected_context=None):
        self.calls.append((command, extra_env))
        stdout = Path(log_dir) / "command.stdout"
        stderr = Path(log_dir) / "command.stderr"
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("actual stdout")
        stderr.write_text("actual stderr")
        return VerificationResult(
            command=command, exit_code=self.exit_code, timed_out=self.timed_out,
            stdout_path=str(stdout), stderr_path=str(stderr), duration_seconds=0.1,
        )


@pytest.mark.parametrize("writable,allowed", [(False, True), (True, False)])
def test_run_shell_enforces_permission_before_any_execution(tmp_path, writable, allowed):
    runner = FakeRunner(tmp_path, writable=writable)
    tool = RunShellTool(runner, timeout_seconds=30, log_dir=str(tmp_path / "logs"), allowed=allowed)
    with pytest.raises(PermissionError):
        tool.execute(_state(), tool.input_model(command="python train.py"))
    assert runner.calls == []


@pytest.mark.parametrize("command", [
    "python train.py", "python -m pytest", "ls | sort",
    "python -m pip install numpy", "python -m venv .venv",
    "conda run -p /shared/env python train.py",
])
def test_environment_and_execution_commands_need_no_system_binding(tmp_path, command):
    runner = FakeRunner(tmp_path)
    tool = RunShellTool(runner, timeout_seconds=30, log_dir=str(tmp_path / "logs"),
                        extra_env={"SCIAGENT_DATASET_ROOT": "/datasets"})
    observation = tool.execute(_state(), tool.input_model(command=command))
    assert runner.calls == [(command, {"SCIAGENT_DATASET_ROOT": "/datasets"})]
    assert observation.ok
    assert observation.value["command"] == command
    assert observation.value["stdout_tail"] == "actual stdout"
    assert observation.value["stderr_tail"] == "actual stderr"
    assert "environment_certified" not in observation.value


@pytest.mark.parametrize("exit_code,timed_out", [(1, False), (-9, True), (0, True)])
def test_execution_failure_or_timeout_keeps_its_actual_result(tmp_path, exit_code, timed_out):
    runner = FakeRunner(tmp_path, exit_code=exit_code, timed_out=timed_out)
    tool = RunShellTool(runner, timeout_seconds=30, log_dir=str(tmp_path / "logs"))
    observation = tool.execute(_state(), tool.input_model(command="python train.py"))
    assert not observation.ok
    assert observation.value["exit_code"] == exit_code
    assert observation.value["timed_out"] is timed_out
    assert "passed" not in observation.value
