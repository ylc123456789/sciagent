"""Read file tool."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from resagent2_runtime import AgentState, ToolObservation
from resagent2_runtime.models import NonEmptyStr, RuntimeModel

from resagent2_components.context import remember_source as _remember
from resagent2_components.text import (
    MAX_READ_CHARS,
    MAX_WORKSPACE_TEXT_BYTES,
    read_text_file,
    slice_text_lines,
    validate_text_window,
)
from resagent2_components.workspace import (
    FILESYSTEM_PATH_DESCRIPTION,
    FILESYSTEM_PATH_GUIDANCE,
    ReadOnlyWorkspaceBoundary,
    WorkspaceBoundary,
)


class ReadFileInput(RuntimeModel):
    """Read one optional line and character range from a workspace file."""

    path: NonEmptyStr = Field(description=FILESYSTEM_PATH_DESCRIPTION)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    start_char: int = Field(
        default=0, ge=0,
        description="Zero-based character offset within the selected physical lines.",
    )
    end_char: int | None = Field(
        default=None, ge=1,
        description="Exclusive character end within the selected physical lines.",
    )


class ReadFileTool:
    """Read a bounded UTF-8 text window through a WorkspaceBoundary."""

    name = "read_file"
    input_model = ReadFileInput
    model_guidance = FILESYSTEM_PATH_GUIDANCE + (
        "Read UTF-8 text only; NUL bytes or invalid UTF-8 are rejected. "
        "Workspace text files are limited to 10 MiB; each result is bounded to "
        "128000 characters. Line ranges are one-based and inclusive. Character "
        "offsets are zero-based within the selected lines, with an exclusive "
        "end_char. Source newline characters are preserved. Read only needed "
        "portions using bounded line or character ranges. To continue within "
        "the same lines, set start_char to next_start_char and choose a new "
        "bounded end_char. Keep the line range fixed; do not use the requested "
        "end_char as the continuation offset. next_start_char=None means the "
        "selected lines are exhausted, not necessarily the whole file. "
        "truncated=False means the requested window fit the tool output limit, "
        "not that the selected lines are exhausted. total_lines is the physical "
        "line count of the whole file; selected_chars is the character count of "
        "the selected lines before the character window. Use these counts to "
        "distinguish empty sources from line or character ranges beyond the end. "
        "Do not repeat the same unbounded read."
    )

    def __init__(
        self,
        boundary: WorkspaceBoundary | ReadOnlyWorkspaceBoundary,
        *,
        max_chars: int = MAX_READ_CHARS,
        max_bytes: int = MAX_WORKSPACE_TEXT_BYTES,
    ) -> None:
        self.boundary = boundary
        self.max_chars = max_chars
        self.max_bytes = max_bytes

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(ReadFileInput, arguments)
        validate_text_window(
            start_line=args.start_line, end_line=args.end_line,
            start_char=args.start_char, end_char=args.end_char,
        )
        path = self.boundary.resolve_read_file(args.path)
        text = read_text_file(path, max_bytes=self.max_bytes)
        return ToolObservation(
            summary=f"Read {args.path}",
            value={
                "path": args.path,
                **slice_text_lines(
                    text, start_line=args.start_line, end_line=args.end_line,
                    start_char=args.start_char, end_char=args.end_char,
                    max_chars=self.max_chars,
                ),
            },
            memory_updates={"read_paths": _remember(state, "read_paths", args.path)},
        )
