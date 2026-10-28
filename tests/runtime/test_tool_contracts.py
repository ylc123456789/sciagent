"""Tests for the model-facing per-tool argument contract."""

import pytest

from pydantic import BaseModel

from resagent2_capabilities import (
    ReplaceTextTool,
)

from resagent2_capabilities import RunShellTool
from resagent2_runtime import AgentAction, AskUserTool, FinishTool
from resagent2_scientific.tools import (
    AskUserTool as ScientificAskUserTool, RequestWorkTool,
)
from resagent2_runtime.models import NativeToolCall, ToolCallTurn
from resagent2_runtime.tool_calling import (
    CONTROL_TOOLS, NATIVE_TOOL_INSTRUCTION, NativeToolCallError,
    contract_fingerprint, native_actions, native_input, native_tool_schemas,
)


class _FinishInput(BaseModel):
    opinion: dict
    summary: str


class _FinishTool:
    name = "finish"
    input_model = _FinishInput


class _AskInput(BaseModel):
    text: str
    reason: str
    optional_note: str | None = None


class _AskTool:
    name = "ask_user"
    input_model = _AskInput


def test_native_schemas_list_required_arguments() -> None:
    schemas = native_tool_schemas((_FinishTool, _AskTool))
    assert schemas[0]["function"]["parameters"]["required"] == ["opinion", "summary"]
    assert schemas[1]["function"]["parameters"]["required"] == ["text", "reason"]


def test_agent_action_has_no_reasoning_summary() -> None:
    # The dead field must be gone so it no longer competes with a tool's
    # required `summary` argument.
    assert "reasoning_summary" not in AgentAction.model_fields


def test_ask_user_contract_requires_requested_fields() -> None:
    from resagent2_runtime import AskUserTool

    required = native_tool_schemas((AskUserTool(),))[0]["function"]["parameters"]["required"]
    assert required == ["text", "requested_fields"]


def test_tool_contracts_include_optional_model_guidance() -> None:
    class _ReadInput(BaseModel):
        path: str

    class _ReadTool:
        name = "read_file"
        input_model = _ReadInput
        model_guidance = "read a bounded start_line/end_line range when truncated"

    schemas = native_tool_schemas((_ReadTool, _FinishTool))
    assert schemas[0]["function"]["parameters"]["required"] == ["path"]
    assert _ReadTool.model_guidance in schemas[0]["function"]["description"]
    assert schemas[1]["function"]["parameters"]["required"] == ["opinion", "summary"]


@pytest.mark.parametrize("tool_class", [
    ReplaceTextTool,
    RequestWorkTool, ScientificAskUserTool, FinishTool, AskUserTool,
    RunShellTool,
])
def test_operational_guidance_reaches_native_tool_schema(tool_class) -> None:
    # Guidance is class-level. Rendering it needs no environment or filesystem.
    tool = object.__new__(tool_class)
    native = native_tool_schemas((tool,))[0]["function"]

    assert tool.model_guidance.strip()
    assert native["name"] == tool.name
    assert tool.model_guidance in native["description"]


@pytest.mark.parametrize("name", sorted(CONTROL_TOOLS))
def test_control_tool_guidance_and_rejection_match_enforced_singleton_rule(name) -> None:
    class ControlTool:
        input_model = _AskInput

    tool = ControlTool()
    tool.name = name
    schema = native_tool_schemas((tool,))[0]["function"]
    assert name in NATIVE_TOOL_INSTRUCTION
    assert "only tool call" in schema["description"]
    control = NativeToolCall(id="control", name=name, arguments="{}")
    ordinary = NativeToolCall(id="read", name="read_file", arguments='{"path":"file.py"}')
    with pytest.raises(NativeToolCallError) as rejected:
        native_actions(ToolCallTurn(tool_calls=[ordinary, control]))
    message = str(rejected.value)
    assert name in message and "No tool in this batch was executed" in message
    assert "exactly one tool" in message
    assert native_actions(ToolCallTurn(tool_calls=[control])) == [{"tool": name, "arguments": {}}]


def test_session_fingerprint_covers_the_actual_fixed_native_system_prompt(monkeypatch) -> None:
    from resagent2_runtime import tool_calling
    import hashlib
    import json

    schemas = native_tool_schemas((_FinishTool,))
    wire = native_input("assigned task", schemas, [])
    fixed = wire["messages"][0]["content"]
    expected = hashlib.sha256(json.dumps(
        {"system": fixed, "tools": schemas}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    before = contract_fingerprint("", schemas)
    assert before == expected
    monkeypatch.setattr(tool_calling, "NATIVE_TOOL_INSTRUCTION", NATIVE_TOOL_INSTRUCTION + " Changed rule.")
    assert contract_fingerprint("", schemas) != before
