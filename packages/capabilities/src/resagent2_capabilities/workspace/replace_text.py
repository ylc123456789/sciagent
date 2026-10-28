"""Replace text tool."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field, field_validator

from resagent2_runtime import AgentState, ToolObservation
from resagent2_runtime.models import NonEmptyStr, RuntimeModel
from resagent2_components.text import (
    MAX_WORKSPACE_TEXT_BYTES,
    encode_text,
    read_text_file,
)
from resagent2_components.workspace import (
    MAIN_WORKSPACE_PATH_DESCRIPTION,
    MAIN_WORKSPACE_PATH_GUIDANCE,
    WorkspaceBoundary,
)


class ReplaceTextInput(RuntimeModel):
    """Replace one exact text occurrence in an existing file."""

    path: NonEmptyStr = Field(description=MAIN_WORKSPACE_PATH_DESCRIPTION)
    old_text: str
    new_text: str

    @field_validator("old_text")
    @classmethod
    def _old_text_must_not_be_empty(cls, value: str) -> str:
        """Reject empty matches without stripping whitespace.

        ``old_text`` is an exact-text needle: leading/trailing spaces, tabs and
        newlines are significant (they encode Python indentation). Unlike
        ``NonEmptyStr``, it must never be strip-normalized.
        """
        if value == "":
            raise ValueError("old_text must not be empty")
        return value


class ReplaceTextTool:
    """Atomically replace one exact text occurrence in the current file."""

    name = "replace_text"
    input_model = ReplaceTextInput
    model_guidance = MAIN_WORKSPACE_PATH_GUIDANCE + (
        "old_text must match exactly once in the current file per call; whitespace "
        "is significant. Read current content before editing. Multiple calls are "
        "allowed; each successful edit changes the content used by subsequent "
        "matches. Review the resulting diff. UTF-8 text must have no NUL bytes; "
        "both the original file and replacement result are limited to 10 MiB "
        "encoded bytes. Source newline characters are preserved."
    )

    def __init__(
        self, boundary: WorkspaceBoundary, *,
        max_bytes: int = MAX_WORKSPACE_TEXT_BYTES,
    ) -> None:
        self.boundary = boundary
        self.max_bytes = max_bytes

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(ReplaceTextInput, arguments)
        path = self.boundary.resolve_write_file(args.path)
        encode_text(args.old_text, max_bytes=self.max_bytes)
        encode_text(args.new_text, max_bytes=self.max_bytes)
        text = read_text_file(path, max_bytes=self.max_bytes)
        count = text.count(args.old_text)
        if count != 1:
            raise ValueError(f"old_text must match exactly once; found {count}")
        updated = text.replace(args.old_text, args.new_text, 1)
        if updated == text:
            raise ValueError("replacement does not change the file")
        content = encode_text(updated, max_bytes=self.max_bytes)
        mode = path.stat().st_mode
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                delete=False,
            ) as handle:
                handle.write(content)
                temporary = Path(handle.name)
            os.chmod(temporary, mode)
            if self.boundary.resolve_write_file(args.path) != path:
                raise PermissionError("write target changed during replacement")
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return ToolObservation(
            summary=f"Replaced one exact match in {args.path}",
            value={"path": args.path},
        )
