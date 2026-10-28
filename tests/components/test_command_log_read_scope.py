"""An invocation can inspect its preserved logs without gaining Run-wide reads."""
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sciagent_capabilities import ReadFileTool, RunShellTool
from sciagent_components import ProcessRunner, WorkspaceBoundary
from sciagent_components.read_scope import request_read_boundary
from sciagent_contracts import AgentPermissions, AgentRequest, TaskBudget, WorkspaceAccess, WorkspaceGrant
from sciagent_runtime import AgentState


@pytest.mark.parametrize("external_logs", [False, True])
def test_command_logs_share_exact_read_file_boundary(tmp_path, external_logs):
    root = tmp_path / "workspace"
    root.mkdir()
    grant = WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(read_paths=["."], write_paths=["."]))
    request = AgentRequest(agent="executor", run_id="run_logs", work_request_id="work_logs",
        task_id="task_logs", attempt_number=1, instruction="Run the bounded check",
        permissions=AgentPermissions(execute_commands=True), budget=TaskBudget(max_llm_calls=2, timeout_seconds=20), workspace=grant)
    output = tmp_path / "run" / "attempt" if external_logs else root / ".sciagent" / "task_logs"
    log_dir = str(output / "commands")
    read_boundary = request_read_boundary(request, command_log_dir=log_dir)
    shell = RunShellTool(ProcessRunner(WorkspaceBoundary(grant)), timeout_seconds=10,
        log_dir=log_dir, log_read_boundary=read_boundary)
    now = datetime.now(UTC)
    state = AgentState(session_id="session_logs", agent_name="experiment", owner="executor",
        run_id=request.run_id, task_id=request.task_id, attempt_number=1, created_at=now, updated_at=now)
    result = shell.execute(state, shell.input_model(command="printf 'first\\nsecond\\n'"))
    assert result.ok
    receipt = result.value["stdout_output"]
    assert receipt["read_access"] == "authorized"
    reader = ReadFileTool(read_boundary)
    content = reader.execute(state, reader.input_model(path=receipt["read_path"]))
    assert content.value["content"] == "first\nsecond\n"
    if external_logs:
        private = output.parent / "private-session.json"
        private.write_text("private")
        with pytest.raises(PermissionError):
            read_boundary.resolve_read_file(str(private))


def test_extra_read_scope_preserves_denied_sources_and_other_workspace_grants(tmp_path):
    root, other = tmp_path / "source", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    (root / "secret").write_text("hidden")
    (other / "visible").write_text("authorized")
    request = AgentRequest(agent="executor", run_id="run_scope", work_request_id="work_scope",
        task_id="task_scope", attempt_number=1, instruction="Inspect authorized files",
        permissions=AgentPermissions(), budget=TaskBudget(max_llm_calls=2, timeout_seconds=20),
        workspace=WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(read_paths=["."], denied_paths=["secret"])),
        read_workspaces=[WorkspaceGrant(root=str(other), source="local", access=WorkspaceAccess(read_paths=["."]))])
    boundary=request_read_boundary(request)
    assert boundary.resolve_read_file(str(other / "visible")).read_text() == "authorized"
    with pytest.raises(PermissionError):
        boundary.resolve_read_file("secret")
