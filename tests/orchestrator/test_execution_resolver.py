"""The WorkRequest companion binds choices without changing business content."""
from datetime import UTC, datetime

import pytest

from sciagent_contracts import (
    ExecutionDefinition, ExecutionLimits, WorkRequest, WorkRequestDraft,
    Workflow, WorkflowAgentKind, WorkflowAgentRegistry, WorkspaceDescriptor,
)
from sciagent_executor import default_execution_registry, validate_execution_definition
from sciagent_orchestrator import CompilationError, FixedWorkflowCompiler
from sciagent_orchestrator.compiler import UnrouteableTaskError


def task(key, *, preset="coding@1", **kwargs):
    return {"key": key, "instruction": f"  Original {key}\nKeep conditions exactly.  ",
            "selection": {"kind": "preset", "preset_ref": preset, "reason": "Fits the required work"}, **kwargs}


def work(tasks, work_id="work_resolver"):
    now = datetime.now(UTC)
    return WorkRequest(id=work_id, run_id="run_resolver",
        request=WorkRequestDraft(objective="Parent-only objective", tasks=tasks),
        created_at=now, updated_at=now)


def compile_request(request, *, current=None, registry=None, **kwargs):
    return FixedWorkflowCompiler(validate_execution_definition).compile(request,
        current=current, registry=registry or default_execution_registry(),
        limits=ExecutionLimits(), **kwargs)


def test_binding_preserves_original_tasks_dependencies_and_named_materials():
    request = work([
        task("build", output_names=["implementation"], input_artifact_ids=["artifact_input"]),
        task("evaluate", preset="experiment@1", depends_on=["build"],
            input_artifact_bindings=[{"source_task": "build", "output_selector": "implementation"}],
            output_names=["measurements"]),
    ])
    before = request.model_dump(mode="json")
    result = compile_request(request)
    assert result.llm_calls == 0
    first, second = result.output.tasks
    assert [first.execution_ref, second.execution_ref] == ["coding@1", "experiment@1"]
    assert all(t.workflow_agent_kind == WorkflowAgentKind.EXECUTOR for t in result.output.tasks)
    assert [t.instruction for t in result.output.tasks] == [t.instruction for t in request.request.tasks]
    assert first.input_artifacts == ["artifact_input"]
    assert second.depends_on == [first.id]
    assert second.input_artifact_bindings[0].source_task == first.id
    assert second.input_artifact_bindings[0].output_selector == "implementation"
    assert second.output_names == ["measurements"]
    assert request.model_dump(mode="json") == before


@pytest.mark.parametrize("invalid, category", [
    ({"kind": "no_match", "reason": "No preset provides the requested instrument"}, "no_match"),
    ({"kind": "preset", "preset_ref": "unregistered@1", "reason": "Guessed identity"}, "unknown_preset"),
])
def test_one_unmatched_choice_rejects_entire_batch_without_partial_patch(invalid, category):
    request = work([task("valid"), task("invalid", selection=invalid)])
    current = Workflow(run_id="run_resolver", revision=1, created_from="work_root", tasks=[])
    before = current.model_dump(mode="json")
    with pytest.raises(CompilationError) as caught:
        compile_request(request, current=current)
    assert caught.value.details["category"] == category
    assert caught.value.llm_calls == 0
    assert current.model_dump(mode="json") == before
    if category == "unknown_preset":
        assert caught.value.details["selection_reason"] == invalid["reason"]
    if category == "no_match":
        assert isinstance(caught.value, UnrouteableTaskError)
        assert caught.value.details["unmatched"] == [{"task_key": "invalid", "reason": invalid["reason"]}]


def test_unsupported_host_component_fails_before_materializing_any_tasks():
    registry = default_execution_registry()
    original = registry.execution_definitions["coding@1"]
    altered = ExecutionDefinition(ref=original.ref,
        config=original.config.model_copy(update={"completion": "missing_component"}))
    registry.execution_definitions[altered.ref] = altered
    with pytest.raises(CompilationError) as caught:
        compile_request(work([task("work")]), registry=registry)
    assert caught.value.details["category"] == "unsupported_configuration"
    assert caught.value.details["execution_ref"] == "coding@1"
    assert caught.value.details["selection_reason"] == "Fits the required work"


def test_reusing_local_keys_creates_distinct_ids_and_keeps_new_dependencies_local():
    initial = compile_request(work([task("work")])).output
    from sciagent_contracts import WorkflowTask
    current = Workflow(run_id="run_resolver", revision=1, created_from="work_resolver",
        tasks=[WorkflowTask(**t.model_dump(exclude={"acceptance_spec", "output_names"})) for t in initial.tasks])
    result = compile_request(work([task("work"), task("next", depends_on=["work"])], "work_next"), current=current)
    first, second = result.output.add_tasks
    assert first.id != current.tasks[0].id
    assert second.depends_on == [first.id]
    assert result.output.based_on_revision == current.revision


def test_fixed_binding_revalidates_mutated_typed_input():
    request = work([task("work")])
    request.request.tasks[0].depends_on = ["missing"]
    with pytest.raises(CompilationError) as caught:
        compile_request(request)
    assert caught.value.details["category"] == "invalid_input"
    assert caught.value.llm_calls == 0
