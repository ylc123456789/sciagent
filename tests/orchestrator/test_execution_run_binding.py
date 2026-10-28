"""Execution choices remain bound across registry changes and delegation gates."""
import pytest

from resagent2_contracts import ExecutionDefinition, ExecutionLimits, WorkflowAgentRegistry
from resagent2_orchestrator import FixedWorkflowCompiler, JsonRunStore
from test_recursive_work import Port, complete, conclude, controller, delegate, research_request


class ProcessCrash(BaseException):
    pass


def test_restart_resolves_and_projects_the_runs_frozen_catalog(tmp_path):
    class CrashCompiler:
        def compile(self, *args, **kwargs):
            raise ProcessCrash()

    root = Port(lambda request, turn: delegate(request, "coding", "Original implementation"))
    engine = controller(tmp_path, root, Port(lambda request, turn: complete(request)))
    store = JsonRunStore(tmp_path / "state")
    engine.scheduler.store = engine.scheduler.artifact_registry.store = store
    engine.compiler = CrashCompiler()
    expected = engine.registry.model_copy(deep=True)
    with pytest.raises(ProcessCrash):
        engine.create_run("run_frozen_registry", research_request(goal="Preserve execution selection"))

    # Replacing the live registry after creation also simulates changed defaults
    # in the process that later resumes this persisted Run.
    engine.registry = WorkflowAgentRegistry()
    resumed_root = Port(lambda request, turn: conclude(request))
    child = Port(lambda request, turn: complete(request))
    restored = controller(tmp_path, resumed_root, child)
    restored.scheduler.store = restored.scheduler.artifact_registry.store = JsonRunStore(tmp_path / "state")
    restored.registry = WorkflowAgentRegistry()
    restored.compiler = FixedWorkflowCompiler()
    run = restored.run_until_stable("run_frozen_registry")

    assert run.status == "completed", run.terminal_error
    assert child.requests[0].execution_definition == expected.execution_definitions["coding@1"]
    assert run.execution_definitions == expected.execution_definitions
    assert run.execution_presets == expected.presets
    for request in [*child.requests, *resumed_root.requests]:
        assert [item.preset_ref for item in request.execution_catalog] == [item.preset_ref for item in expected.presets]


@pytest.mark.parametrize("gate", ["configuration", "depth"])
def test_delegation_gate_hides_the_catalog_and_denies_request_work(tmp_path, gate):
    root = Port(lambda request, turn: delegate(request, "coding", "Bound leaf") if turn == 0 else conclude(request))
    child = Port(lambda request, turn: complete(request))
    engine = controller(tmp_path, root, child)
    limits = ExecutionLimits(max_work_depth=2 if gate == "depth" else 3)
    if gate == "configuration":
        original = engine.registry.execution_definitions["coding@1"]
        config = original.config.model_copy(update={"tools": tuple(name for name in original.config.tools if name != "request_work")})
        engine.registry.execution_definitions[original.ref] = ExecutionDefinition(ref=original.ref, config=config)
    run = engine.create_run("run_gate_" + gate, research_request(goal="Run a leaf", execution_limits=limits))
    assert run.status == "completed", run.terminal_error
    assert root.requests[0].execution_catalog
    request = child.requests[0]
    assert request.permissions.request_work is False
    assert request.execution_catalog == []
    assert request.execution_definition == run.execution_definitions[request.execution_definition.ref]
