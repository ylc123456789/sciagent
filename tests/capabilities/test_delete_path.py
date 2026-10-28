"""Deletion uses the existing edit revision and a trusted approval snapshot."""

from datetime import UTC, datetime

import pytest

from sciagent_capabilities import CreateFileTool, DeletePathTool, ReplaceTextTool
from sciagent_components import WorkspaceBoundary, WorkspacePermissionError
from sciagent_contracts import AgentOwner, WorkspaceAccess, WorkspaceGrant
from sciagent_runtime import AgentState


def state():
    now = datetime.now(UTC)
    return AgentState(
        session_id="session_delete", agent_name="test", owner=AgentOwner.EXECUTOR,
        run_id="run_delete", task_id="task_delete", attempt_number=1,
        created_at=now, updated_at=now,
    )


def boundary(root, write=(".",)):
    return WorkspaceBoundary(WorkspaceGrant(
        root=str(root), source="local", access=WorkspaceAccess(
            read_paths=["."], write_paths=list(write),
        ),
    ))


def test_delete_file_records_actual_deleted_paths(tmp_path):
    (tmp_path / "old.py").write_text("pass")
    tool = DeletePathTool(boundary(tmp_path))
    current = state()
    result = tool.execute(current, tool.input_model(path="old.py"))
    assert result.ok
    assert result.value["deleted_paths"] == ["old.py"]
    assert not (tmp_path / "old.py").exists()


def test_recursive_delete_cannot_execute_without_snapshot_approval(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "old").write_text("pass")
    tool = DeletePathTool(boundary(tmp_path))
    args = tool.input_model(path="build", recursive=True)
    with pytest.raises(WorkspacePermissionError, match="confirmation"):
        tool.execute(state(), args)
    assert (tmp_path / "build" / "old").exists()
    result = tool.execute_prepared(state(), args, tool.prepare(args))
    assert result.ok
    assert result.value["deleted_paths"] == ["build/old", "build"]


def test_snapshot_cannot_authorize_another_target(tmp_path):
    (tmp_path / "first").write_text("first")
    (tmp_path / "second").write_text("second")
    tool = DeletePathTool(boundary(tmp_path))
    first = tool.prepare(tool.input_model(path="first"))
    with pytest.raises(WorkspacePermissionError, match="snapshot"):
        tool.execute_prepared(state(), tool.input_model(path="second"), first)
    assert (tmp_path / "first").exists()
    assert (tmp_path / "second").exists()


def test_create_file_in_missing_authorized_directory(tmp_path):
    tool = CreateFileTool(boundary(tmp_path, write=("results",)))
    result = tool.execute(state(), tool.input_model(path="results/new.txt", content="new"))
    assert result.ok
    assert (tmp_path / "results" / "new.txt").read_text() == "new"


def test_delete_text_remains_a_normal_replace(tmp_path):
    (tmp_path / "source.py").write_text("keep = 1\nremove = 2\n")
    tool = ReplaceTextTool(boundary(tmp_path))
    result = tool.execute(state(), tool.input_model(
        path="source.py", old_text="remove = 2\n", new_text="",
    ))
    assert result.ok
    assert (tmp_path / "source.py").read_text() == "keep = 1\n"
