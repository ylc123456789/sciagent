"""Deterministic tests for the multi-round repair flow (scenario 3).

Verifies two things:
- the compiler rejects a patch that supersedes/updates tasks from a previous
  work request (they are immutable history);
- a failed experiment followed by a new WorkRequest that adds repair tasks
  preserves the failed attempt and completes.
"""

from __future__ import annotations
from resagent2_executor import default_execution_registry
from e2e.scheduler_fixtures import executor_bindings

from e2e.native_fixtures import tool_turns

from resagent2_contracts import RunPermissions, ExecutionLimits

import json
import pytest

from resagent2_contracts import (
    AgentOwner,
    AgentResult,
    ArtifactCandidate,
    ErrorCode,
    ModuleError,
    ModuleStatus,
    ResearchRequest,
    RunBudget,
    RunStatus,
    ScientificVerdict,
    TaskProposal,
    TaskStatus,
    WorkflowAgentKind,
    WorkflowAgentRegistry,
    WorkflowPatch,
    WorkflowProposal,
)
from resagent2_orchestrator import (
    DeterministicWorkInterpreter,
    DeterministicWorkflowCompiler,
    InMemoryRunStore,
    FixedWorkflowCompiler,
    ModuleBinding,
    ResearchController,
    ScriptedModulePort,
    WorkflowScheduler,
)
from resagent2_runtime import InMemorySessionStore, ScriptedLLMClient
from resagent2_scientific import ScientificAgent


def _registry() -> WorkflowAgentRegistry:
    return default_execution_registry()


def _completed() -> AgentResult:
    return AgentResult(status=ModuleStatus.COMPLETED, report="done")


def _failed() -> AgentResult:
    return AgentResult(
        status=ModuleStatus.FAILED,
        report="boom",
        error=ModuleError(code=ErrorCode.TOOL_FAILED, message="boom", retryable=False),
    )


def _request_work() -> dict:
    return {
        "tool": "request_work",
        "arguments": {
            "assessment": {"statement": "need evidence"},
            "work_request": {'objective': 'Produce the accuracy evidence', 'tasks': [{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Produce the accuracy evidence'}]},
        },
    }


def _finish() -> dict:
    return {"tool": "finish", "arguments": {"report": "Scientific conclusion", "artifacts": [{'kind': "scientific_opinion", 'path': "scientific_opinion.json", 'media_type': "application/json", 'summary': "Scientific conclusion", 'content': {'verdict': ScientificVerdict.INCONCLUSIVE.value, 'statement': 'the first run failed, then a fix restored it', 'limitations': ['the first run failed before the fix']}}]}}


def test_candidate_rejects_empty_graph() -> None:
    from resagent2_orchestrator.workflow_validation import validate_workflow_candidate

    proposal = WorkflowProposal(
        work_request_id="work_1",
        tasks=[],
    )
    with pytest.raises(ValueError, match="empty task graph"):
        validate_workflow_candidate(proposal)


def test_candidate_rejects_cross_request_dependency() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        WorkflowPatch(
            work_request_id="work_2",
            based_on_revision=1,
            add_tasks=[TaskProposal(execution_ref="experiment@1",
                id="task_exp2", work_request_id="work_2",
                workflow_agent_kind="executor",
                instruction="Rerun",
                depends_on=["task_exp"],  # a prior work request's failed task
            )],
        )


def test_repair_flow_preserves_failed_task_and_completes(tmp_path) -> None:
    # Round 1: a single experiment task that fails.
    proposal = WorkflowProposal(
        work_request_id="work_1",
        tasks=[
            TaskProposal(execution_ref="experiment@1",
                id="task_exp",
                work_request_id="work_1",
                workflow_agent_kind="executor",
                instruction="Run the experiment",

            )
        ],
    )
    # Round 2: a repair that only ADDS a fix + a rerun (no supersede/update).
    patch = WorkflowPatch(
        work_request_id="work_2",
        based_on_revision=2,
        add_tasks=[
            TaskProposal(execution_ref="coding@1",
                id="task_fix",
                work_request_id="work_2",
                workflow_agent_kind="executor",
                instruction="Fix the bug",

            ),
            TaskProposal(execution_ref="experiment@1",
                id="task_exp2",
                work_request_id="work_2",
                workflow_agent_kind="executor",
                instruction="Rerun the experiment",

            ),
        ],
    )

    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "coding": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScriptedModulePort([_completed()]),
            ),
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScriptedModulePort([_failed(), _completed()]),
            ),
        }),
        store=InMemoryRunStore(),
        artifact_root=tmp_path / "artifacts",
    )
    scientific = ScientificAgent(
        ScriptedLLMClient(tool_turns([_request_work(), _request_work(), _finish()])),
        store=InMemorySessionStore(),
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=DeterministicWorkflowCompiler(proposal, patch),
        scheduler=scheduler,
        registry=_registry(),
    )
    request = ResearchRequest(goal='Run the experiment; if it fails, fix and rerun.', budget=RunBudget(max_llm_calls=50, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=5, max_attempts_per_task=3))

    run = controller.create_run("run_repair", request)

    assert run.status == RunStatus.COMPLETED, [v.message for v in run.completion_violations]
    tasks = {task.id: task for task in run.workflow.tasks}
    # The old failed experiment is preserved as history, not overwritten.
    assert tasks["task_exp"].status == TaskStatus.FAILED
    assert tasks["task_exp"].attempts and tasks["task_exp"].attempts[-1].status.value == "failed"
    # The repair tasks were added and completed.
    assert tasks["task_fix"].status == TaskStatus.COMPLETED
    assert tasks["task_exp2"].status == TaskStatus.COMPLETED
    # Two work requests: the initial run and the repair.
    assert len(run.work_requests) >= 2


def _finish_after_repair() -> dict:
    return {"tool": "finish", "arguments": {"report": "Scientific conclusion", "artifacts": [{'kind': "scientific_opinion", 'path': "scientific_opinion.json", 'media_type': "application/json", 'summary': "Scientific conclusion", 'content': {'verdict': ScientificVerdict.INCONCLUSIVE.value, 'statement': 'the first run failed, then a fix restored it', 'limitations': ['the first run failed before the fix']}}]}}


def test_repair_flow_with_fixed_preset_binding(tmp_path) -> None:
    """Parent-selected presets preserve the repair graph and failed history."""
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "coding": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScriptedModulePort([_completed()]),
            ),
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScriptedModulePort([_failed(), _completed()]),
            ),
        }),
        store=InMemoryRunStore(),
        artifact_root=tmp_path / "artifacts",
    )
    scientific = ScientificAgent(
        ScriptedLLMClient(tool_turns([
            {"tool": "request_work", "arguments": {"assessment": {"statement": "Initial experiment"}, "work_request": {
                "objective": "Run the experiment", "tasks": [{"selection":{"kind":"preset","preset_ref":"experiment@1","reason":"Explicit test preset"}, "key": "run_initial", "instruction": "Run the experiment"}],
            }}},
            {"tool": "request_work", "arguments": {"assessment": {"statement": "Repair the observed failure"}, "work_request": {
                "objective": "Fix the bug and rerun", "tasks": [
                    {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "fix", "instruction": "Fix the bug"},
                    {"selection":{"kind":"preset","preset_ref":"experiment@1","reason":"Explicit test preset"}, "key": "rerun", "instruction": "Rerun the experiment", "depends_on": ["fix"]},
                ],
            }}},
            _finish_after_repair(),
        ])),
        store=InMemorySessionStore(),
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=FixedWorkflowCompiler(),
        scheduler=scheduler,
        registry=_registry(),
    )
    request = ResearchRequest(goal='Run the experiment; if it fails, fix and rerun.', budget=RunBudget(max_llm_calls=50, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=5, max_attempts_per_task=3))

    run = controller.create_run("run_repair_semantic", request)

    assert run.status == RunStatus.COMPLETED, [v.message for v in run.completion_violations]
    tasks = {task.id: task for task in run.workflow.tasks}
    # The old failed experiment is preserved, not overwritten.
    assert tasks["task_run_initial"].status == TaskStatus.FAILED
    assert tasks["task_run_initial"].attempts[-1].status.value == "failed"
    # The repair tasks were added and completed (with code-assigned ids).
    assert tasks["task_fix"].status == TaskStatus.COMPLETED
    assert tasks["task_rerun"].status == TaskStatus.COMPLETED
    assert tasks["task_rerun"].depends_on == ["task_fix"]
    assert len(run.work_requests) >= 2
