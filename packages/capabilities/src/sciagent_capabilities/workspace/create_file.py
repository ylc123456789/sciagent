"""Create file tool."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from sciagent_runtime import AgentState, ToolObservation
from sciagent_runtime.models import NonEmptyStr, RuntimeModel
from sciagent_components.text import MAX_WORKSPACE_TEXT_BYTES, encode_text
from sciagent_components.workspace import (
    MAIN_WORKSPACE_PATH_DESCRIPTION,
    MAIN_WORKSPACE_PATH_GUIDANCE,
    WorkspaceBoundary,
)


class CreateFileInput(RuntimeModel):
    """Create one new UTF-8 workspace file."""

    path: NonEmptyStr = Field(description=MAIN_WORKSPACE_PATH_DESCRIPTION)
    content: str


class CreateFileTool:
    """Create a file only when its target does not exist."""

    name = "create_file"
    input_model = CreateFileInput
    model_guidance = MAIN_WORKSPACE_PATH_GUIDANCE + (
        "Create UTF-8 text only, without NUL bytes, up to 10 MiB encoded bytes. "
        "Source newline characters are preserved. Existing files are not overwritten."
    )

    def __init__(
        self, boundary: WorkspaceBoundary, *,
        max_bytes: int = MAX_WORKSPACE_TEXT_BYTES,
    ) -> None:
        self.boundary = boundary
        self.max_bytes = max_bytes

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(CreateFileInput, arguments)
        path = self.boundary.resolve_write_file(args.path, must_be_new=True)
        content = encode_text(args.content, max_bytes=self.max_bytes)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self.boundary.resolve_write_file(args.path, must_be_new=True)
        with path.open("xb") as handle:
            handle.write(content)
        return ToolObservation(
            summary=f"Created {args.path}",
            value={"path": args.path, "bytes": len(content)},
        )
