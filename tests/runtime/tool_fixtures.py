"""In-memory tools used only to exercise the runtime execution engine."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, JsonValue

from resagent2_runtime import AgentState, ToolObservation
from resagent2_runtime.models import NonEmptyStr, RuntimeModel


class ReadValueInput(RuntimeModel):
    """Input schema for ReadValueTool."""

    key: NonEmptyStr


class ReadValueTool:
    """Read one key from generic in-memory Agent state."""

    name = "read_value"
    input_model = ReadValueInput

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(ReadValueInput, arguments)
        return ToolObservation(
            summary=f"Read memory key {args.key!r}",
            value=state.memory.get(args.key),
        )


class WriteValueInput(RuntimeModel):
    """Input schema for WriteValueTool."""

    key: NonEmptyStr
    value: JsonValue


class WriteValueTool:
    """Propose one generic in-memory state update."""

    name = "write_value"
    input_model = WriteValueInput

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(WriteValueInput, arguments)
        return ToolObservation(
            summary=f"Wrote memory key {args.key!r}",
            value=args.value,
            memory_updates={args.key: args.value},
        )
