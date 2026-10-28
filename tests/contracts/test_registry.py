"""Frozen execution definitions, parent selections and stable catalog contracts."""
import pytest
from pydantic import ValidationError

from sciagent_contracts import (
    AgentRequest, ExecutionCatalogEntry, ExecutionConfig, ExecutionDefinition,
    ExecutionPreset, WorkflowAgentRegistry, WorkTaskDraft, WorkflowTask,
)


def config(**changes):
    data = dict(system_prompt="Preserve the exact task.\n", tools=["finish", "ask_user"],
        preparation="workspace@1", completion="artifacts@1", finalizer="execution@1",
        context_builder="executor@1", implementation_version="1",
        model_profile="host-default@1", max_context_tokens=256000)
    data.update(changes)
    return ExecutionConfig(**data)


def definition(**changes):
    return ExecutionDefinition(ref="analysis@1", config=config(**changes))


def preset(**changes):
    data = dict(preset_ref="analysis@1", name="Analysis", description="Inspect existing evidence",
                execution_ref="analysis@1")
    data.update(changes)
    return ExecutionPreset(**data)


def test_registry_round_trip_and_same_source_parent_projection():
    resolved = definition()
    registry = WorkflowAgentRegistry(execution_definitions={resolved.ref: resolved}, presets=[preset()])
    assert WorkflowAgentRegistry.model_validate_json(registry.model_dump_json()) == registry
    visible = ExecutionCatalogEntry.model_validate(registry.presets[0].model_dump())
    assert visible.execution_ref == resolved.ref
    assert "config" not in visible.model_dump()
    assert WorkflowAgentRegistry().presets == []


@pytest.mark.parametrize("damage", ["duplicate", "unknown", "key_mismatch"])
def test_registry_rejects_inconsistent_definitions(damage):
    definitions = {"analysis@1": definition()}
    presets = [preset()]
    if damage == "duplicate":
        presets.append(preset())
    elif damage == "unknown":
        presets = [preset(execution_ref="missing@1")]
    else:
        definitions = {"other@1": definition()}
    with pytest.raises(ValidationError):
        WorkflowAgentRegistry(execution_definitions=definitions, presets=presets)


def test_definition_hash_covers_nonprompt_strategy_and_model_changes():
    original = definition()
    assert len(original.config_hash) == 64
    assert definition().config_hash == original.config_hash
    for changes in ({"completion": "artifacts@2"}, {"model_profile": "other@1"},
                    {"implementation_version": "2"}, {"max_context_tokens": 1000}):
        assert definition(**changes).config_hash != original.config_hash
    data = original.model_dump(mode="json")
    data["config"]["finalizer"] = "changed@1"
    with pytest.raises(ValidationError, match="config_hash"):
        ExecutionDefinition.model_validate(data)
    with pytest.raises(ValidationError, match="frozen"):
        original.config.completion = "changed@1"


def test_contracts_validate_shape_without_hardcoding_host_components():
    assert config(tools=["host_future_tool"], completion="host_future_strategy@2")
    for changes in ({"permissions": {}}, {"execute_commands": True}, {"import_path": "module.Factory"},
                    {"tools": ["finish", "finish"]}, {"max_context_tokens": True}):
        with pytest.raises(ValidationError):
            config(**changes)


def test_parent_selection_is_required_exclusive_and_catalog_independent_schema():
    common = dict(key="analyze", instruction="  Preserve this original task\n")
    with pytest.raises(ValidationError, match="selection"):
        WorkTaskDraft(**common)
    task = WorkTaskDraft(**common, selection={"kind": "preset", "preset_ref": "analysis@99",
                                              "reason": "Use the visible analysis capability"})
    assert task.instruction == common["instruction"]
    assert task.selection.preset_ref == "analysis@99"
    assert WorkTaskDraft(**common, selection={"kind": "no_match", "reason": "Need unavailable capability"})
    for selection in ({"kind": "preset", "preset_ref": "a@1"},
                      {"kind": "no_match", "reason": "Gap", "preset_ref": "a@1"},
                      {"kind": "preset", "preset_ref": "a@1", "reason": " "}):
        with pytest.raises(ValidationError):
            WorkTaskDraft(**common, selection=selection)
    schema = WorkTaskDraft.model_json_schema()
    assert "enum" not in schema["$defs"]["PresetSelection"]["properties"]["preset_ref"]


def test_execution_reference_is_not_the_task_owner():
    common = dict(id="task_analysis", work_request_id="work_test", instruction="Inspect")
    with pytest.raises(ValidationError, match="requires execution_ref"):
        WorkflowTask(**common, workflow_agent_kind="executor")
    task = WorkflowTask(**common, workflow_agent_kind="executor", execution_ref="custom-analysis@1")
    assert task.workflow_agent_kind == "executor"
    with pytest.raises(ValidationError, match="Scientific"):
        WorkflowTask(**common, workflow_agent_kind="scientific", execution_ref="analysis@1")
    with pytest.raises(ValidationError):
        WorkflowTask(**common, workflow_agent_kind="coding", execution_ref="analysis@1")


def test_root_cannot_receive_executor_definition():
    with pytest.raises(ValidationError, match="Scientific"):
        AgentRequest(run_id="run_test", work_request_id="work_root", task_id="task_root",
                     attempt_number=1, agent="scientific", instruction="Assess evidence", permissions={},
                     budget={"max_llm_calls": 2, "timeout_seconds": 30}, execution_definition=definition())
