"""Read registered artifacts through the model tool protocol."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from resagent2_runtime import AgentState, ToolObservation
from resagent2_runtime.models import NonEmptyStr, RuntimeModel
from resagent2_components.artifacts import RegisteredArtifactReader
from resagent2_components.context import remember_source as _remember

class ReadArtifactInput(RuntimeModel):
    """Identify one registered ArtifactRef and an optional inclusive line range."""

    artifact_id: NonEmptyStr
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    start_char: int = Field(default=0, ge=0, description="Zero-based character offset within the selected physical lines.")
    end_char: int | None = Field(default=None, ge=1, description="Exclusive character end within the selected physical lines.")


class ReadArtifactTool:
    """Read a provided ArtifactRef after integrity verification."""

    name = "read_artifact"
    input_model = ReadArtifactInput
    model_guidance = (
        "Read UTF-8 text only; NUL bytes or invalid UTF-8 are rejected "
        "and failed reads are not recorded as read. "
        "Read only needed portions using bounded start_line/end_line or "
        "start_char/end_char ranges (zero-based characters, end exclusive). "
        "To continue within the same lines, set start_char to next_start_char "
        "and choose a new bounded end_char. Keep the line range fixed; do not "
        "use the requested end_char as the continuation offset. "
        "next_start_char=None means the selected lines are exhausted, not "
        "necessarily the whole artifact. truncated=False means the requested "
        "window fit the tool output limit, not that the selected lines are exhausted. "
        "total_lines is the physical line count of the whole artifact; "
        "selected_chars is the character count of the selected lines before the "
        "character window. Use these counts to distinguish empty sources from "
        "line or character ranges beyond the end. Do not repeat an unbounded "
        "read. The full frozen file is integrity-checked "
        "before any range is returned."
    )

    def __init__(self, reader: RegisteredArtifactReader) -> None:
        self.reader = reader

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(ReadArtifactInput, arguments)
        value = self.reader.read_text(
            args.artifact_id, start_line=args.start_line, end_line=args.end_line,
            start_char=args.start_char, end_char=args.end_char,
        )
        return ToolObservation(
            summary=f"Read registered Artifact {args.artifact_id}",
            value=value,
            memory_updates={
                "read_artifact_ids": _remember(state, "read_artifact_ids", args.artifact_id),
            } if value["content"] else {},
        )
