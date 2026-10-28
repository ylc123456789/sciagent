"""One directory tool serves both workspace and parent read-only scopes."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sciagent_capabilities import ListFilesTool
from sciagent_components import ReadOnlyWorkspaceBoundary, WorkspaceBoundary
from sciagent_contracts import WorkspaceAccess, WorkspaceGrant
from sciagent_runtime import AgentState


def state():
    now = datetime.now(UTC)
    return AgentState(session_id="session_list", agent_name="coding", owner="executor",
                      run_id="run_list", created_at=now, updated_at=now)


def grant(path):
    return WorkspaceGrant(root=str(path), source="local", access=WorkspaceAccess(read_paths=["."]))


@pytest.mark.parametrize("parent", [False, True])
def test_tool_paginates_with_the_same_node_shape_for_both_path_conventions(tmp_path, parent):
    (tmp_path / "a").touch()
    (tmp_path / "b").mkdir()
    boundary = ReadOnlyWorkspaceBoundary([grant(tmp_path)]) if parent else WorkspaceBoundary(grant(tmp_path))
    tool = ListFilesTool(boundary)
    path = str(tmp_path) if parent else "."
    current = state()
    before = current.model_dump_json()
    first = tool.execute(current, tool.input_model(path=path, max_entries=1))
    assert first.ok is True
    assert first.value["children"][0]["name"] == "a"
    cursor = first.value["expansion"]["next_cursor"]
    second = tool.execute(current, tool.input_model(path=path, cursor=cursor))
    assert second.value["children"][0]["name"] == "b"
    assert second.value["expansion"]["state"] == "complete"
    assert second.memory_updates == {}
    assert current.model_dump_json() == before


def test_tool_schema_has_bounded_pages_and_rejects_obsolete_truncation_only_field():
    with pytest.raises(ValidationError):
        ListFilesTool.input_model(max_entries=2001)
    with pytest.raises(ValidationError):
        ListFilesTool.input_model(max_files=20)
    assert "next_cursor" in ListFilesTool.model_guidance
    assert "read_file" not in ListFilesTool.model_guidance
