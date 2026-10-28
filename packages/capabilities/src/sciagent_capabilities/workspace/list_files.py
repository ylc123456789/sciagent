"""List one authorized directory through stable, resumable pages."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from sciagent_runtime import AgentState, ToolObservation
from sciagent_runtime.models import RuntimeModel
from sciagent_components.workspace import (
    FILESYSTEM_PATH_DESCRIPTION,
    FILESYSTEM_PATH_GUIDANCE,
    ReadOnlyWorkspaceBoundary,
    WorkspaceBoundary,
)


class ListFilesInput(RuntimeModel):
    """One directory observation; repeat its path with the returned cursor."""

    path: str = Field(default=".", description=FILESYSTEM_PATH_DESCRIPTION)
    max_entries: int = Field(default=200, ge=1, le=2000)
    cursor: str | None = Field(default=None, max_length=8192)


class ListFilesTool:
    """List immediate readable files and directories without recursive scans."""

    name = "list_files"
    input_model = ListFilesInput
    model_guidance = FILESYSTEM_PATH_GUIDANCE + (
        "List one authorized directory, not its recursive file tree. Returns "
        "file and directory nodes with usable locations and explicit expansion "
        "states. Directory children marked not_expanded can be listed separately. "
        "A partial page provides expansion.next_cursor: repeat the same path "
        "with that cursor to continue in stable filename order. A complete page "
        "exhausts that directory's immediate readable entries, not its descendants. "
        "A partial page may contain no visible entries when its bounded scan "
        "encounters many denied entries; its cursor still advances. "
        "The cursor is valid only for the observed directory and read scope; "
        "restart from cursor=null if they change. Locations are copyable absolute paths."
    )

    def __init__(self, boundary: WorkspaceBoundary | ReadOnlyWorkspaceBoundary) -> None:
        self.boundary = boundary

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(ListFilesInput, arguments)
        node = self.boundary.list_directory(args.path, max_entries=args.max_entries, cursor=args.cursor)
        return ToolObservation(
            summary=f"Listed {len(node['children'])} immediate entries from {args.path}",
            value=node,
        )
