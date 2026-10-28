"""CLI presets, frozen Run definitions and dispatch use one Executor binding."""

import pytest

from e2e.native_fixtures import tool_turn
from resagent2_cli import composition
from resagent2_contracts import (
    AgentOwner, AgentResult, ExecutionDefinition, ExecutionPreset, ModuleStatus,
    ResearchRequest, RunBudget, RunPermissions, RunStatus, WorkflowAgentKind, WorkflowAgentRegistry,
)
from resagent2_executor import NativeExecutorAgent, default_execution_registry
from resagent2_orchestrator import FixedWorkflowCompiler
from resagent2_runtime import ScriptedLLMClient


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("RESAGENT2_WEB_SEARCH_PROVIDER", "off")
    monkeypatch.setattr("httpx.Client.send", lambda *a, **k: pytest.fail("no HTTP"))


def ask_turn():
    return tool_turn("ask_user", {
        "text": "Review the completed work?", "requested_fields": ["decision"],
        "assessment": {"statement": "The current work result is available."},
    })


def test_cli_has_one_executor_binding_and_separate_trusted_presets(monkeypatch, tmp_path):
    client = ScriptedLLMClient([])
    monkeypatch.setattr(composition, "_client", lambda: client)
    app = composition.build_application(data_root=tmp_path)
    assert set(app.controller.scheduler.bindings) == {
        WorkflowAgentKind.SCIENTIFIC, WorkflowAgentKind.EXECUTOR,
    }
    binding = app.controller.scheduler.bindings[WorkflowAgentKind.EXECUTOR]
    assert binding.owner == AgentOwner.EXECUTOR
    assert isinstance(binding.port, NativeExecutorAgent)
    assert binding.port.loop.store.root == tmp_path / "sessions" / "executor"
    assert isinstance(app.controller.compiler, FixedWorkflowCompiler)
    assert not hasattr(app.controller.compiler, "_client")
    registry = app.controller.registry
    assert {p.preset_ref: p.execution_ref for p in registry.presets} == {
        "coding@1": "coding@1", "experiment@1": "experiment@1",
    }
    assert "create_file" in registry.execution_definitions["coding@1"].config.tools
    assert "create_file" not in registry.execution_definitions["experiment@1"].config.tools
    assert client.contexts == []


def test_unsupported_definition_fails_composition_before_model_invocation(monkeypatch, tmp_path):
    client = ScriptedLLMClient([])
    monkeypatch.setattr(composition, "_client", lambda: client)
    default = default_execution_registry().execution_definitions["coding@1"]
    invalid = ExecutionDefinition(ref="unsupported@1", config=default.config.model_copy(
        update={"implementation_version": "missing"}))
    registry = WorkflowAgentRegistry(execution_definitions={invalid.ref: invalid}, presets=[
        ExecutionPreset(preset_ref="unsupported@1", name="Unsupported", description="Unavailable implementation",
                        execution_ref=invalid.ref),
    ])
    monkeypatch.setattr(composition, "default_execution_registry", lambda **kwargs: registry)
    with pytest.raises(ValueError, match="Unsupported executor implementation version"):
        composition.build_application(data_root=tmp_path)
    assert client.contexts == []


def test_parent_choice_is_frozen_and_dispatched_with_original_instruction(monkeypatch, tmp_path):
    class ReplacementExecutor:
        def __init__(self, model_resolver, **kwargs):
            self.requests = []

        def invoke(self, request):
            self.requests.append(request)
            return AgentResult(status=ModuleStatus.COMPLETED, report="replacement executed")

    instruction = "\n Implement stable_sum(values), preserving input order, then measure its numerical error. \n"
    reason = "This task requires implementation and a continuous numerical measurement."
    client = ScriptedLLMClient([
        tool_turn("request_work", {"assessment": {"statement": "Need delegated work"}, "work_request": {"objective": "Evaluate stable summation", "tasks": [{
            "key": "implement", "instruction": instruction,
            "selection": {"kind": "preset", "preset_ref": "coding@1", "reason": reason},
        }]}}), ask_turn(),
    ])
    monkeypatch.setattr(composition, "_client", lambda: client)
    monkeypatch.setattr(composition, "NativeExecutorAgent", ReplacementExecutor)
    app = composition.build_application(data_root=tmp_path)
    run = app.controller.create_run("run_choice", ResearchRequest(goal="Evaluate stable summation", budget=RunBudget(max_llm_calls=6, timeout_seconds=60), permissions=RunPermissions()))
    assert run.status == RunStatus.PAUSED
    replacement = app.controller.scheduler.bindings[WorkflowAgentKind.EXECUTOR].port
    delivered, = replacement.requests
    assert delivered.agent == AgentOwner.EXECUTOR
    assert delivered.instruction == instruction
    assert delivered.execution_definition == run.execution_definitions["coding@1"]
    child, = [task for task in run.workflow.tasks if task.workflow_agent_kind == WorkflowAgentKind.EXECUTOR]
    assert child.execution_ref == "coding@1"
    work = next(work for work in run.work_requests if work.id == child.work_request_id)
    assert work.request.tasks[0].selection.reason == reason
    assert work.request.tasks[0].instruction == instruction
    assert "shared_goal" not in delivered.task_context
    assert "shared_constraints" not in delivered.task_context
    assert not any(call.name == "route_tasks" for turn in client.histories[-1] for call in turn.tool_calls)
    assert "coding@1" in client.contexts[0].text
    saved_definitions = run.execution_definitions
    monkeypatch.setenv("RESAGENT2_CODING_CONTEXT_TOKENS", "12000")
    reopened = composition.build_application(data_root=tmp_path)
    assert reopened.controller.registry.execution_definitions["coding@1"].config.max_context_tokens == 12000
    assert reopened.run_store.load(run.run_id).execution_definitions == saved_definitions


def test_parent_no_match_creates_no_executor_attempt(monkeypatch, tmp_path):
    client = ScriptedLLMClient([
        tool_turn("request_work", {"assessment": {"statement": "Need delegated work"}, "work_request": {"objective": "Need a missing capability", "tasks": [{
            "key": "unavailable", "instruction": "Acquire a physical sample.",
            "selection": {"kind": "no_match", "reason": "The available presets have no laboratory hardware."},
        }]}}), ask_turn(),
    ])
    monkeypatch.setattr(composition, "_client", lambda: client)
    app = composition.build_application(data_root=tmp_path)
    run = app.controller.create_run("run_no_match", ResearchRequest(goal="Review available capabilities", budget=RunBudget(max_llm_calls=6, timeout_seconds=60), permissions=RunPermissions()))
    assert run.status == RunStatus.PAUSED
    assert all(task.workflow_agent_kind == WorkflowAgentKind.SCIENTIFIC for task in run.workflow.tasks)
    work, = [work for work in run.work_requests if work.owner_task_id is not None]
    assert work.outcome is not None and not work.outcome.tasks
    assert work.outcome.error is not None
    assert "laboratory hardware" in client.contexts[-1].text
