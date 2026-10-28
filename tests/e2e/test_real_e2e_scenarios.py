from sciagent_executor import default_execution_registry

from e2e.native_fixtures import tool_turns

from sciagent_contracts import RunPermissions, ExecutionLimits
import json
"""Local ScriptedLLM validation of the Phase 7 real E2E scenario acceptance.

Drives the deterministic scenarios (1: direct inconclusive, 4: ask/resume)
through the real ResearchController with a scripted Scientific Agent, then
checks the same acceptance predicates used by ``e2e/real_e2e.py``.
"""

from datetime import UTC, datetime
from pathlib import Path

from sciagent_contracts import (
    AgentOwner,
    WorkflowAgentKind,
    WorkflowAgentRegistry,
    ResearchRequest,
    RunBudget,
    ScientificVerdict,
    UserAnswer,
    WorkflowProposal,
)
from sciagent_orchestrator import (
    DeterministicWorkInterpreter,
    DeterministicWorkflowCompiler,
    JsonRunStore,
    ResearchController,
    WorkflowScheduler,
)
from sciagent_runtime import JsonSessionStore, ScriptedLLMClient
from sciagent_scientific import ScientificAgent

from e2e.real_e2e import (
    _ask_resume_succeeded,
    _ask_start_succeeded,
    _direct_succeeded,
)


def _registry() -> WorkflowAgentRegistry:
    return default_execution_registry()


def _controller(workdir: Path, scientific: ScientificAgent) -> ResearchController:
    scheduler = WorkflowScheduler(
        bindings={},
        store=JsonRunStore(workdir / "state"),
        artifact_root=workdir / "artifacts",
    )
    return ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=DeterministicWorkflowCompiler(
            WorkflowProposal(
                work_request_id="work_x",
                tasks=[],
            ),
            patch=None,
        ),
        scheduler=scheduler,
        registry=_registry(),
    )


def _finish_inconclusive() -> dict:
    return {
        "tool": "finish",
        "arguments": {"report": "Scientific conclusion", "artifacts": [{"kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json", "summary": "Scientific conclusion", "content": {"verdict": ScientificVerdict.INCONCLUSIVE.value, "statement": "insufficient evidence"}}]},
    }


def _ask_user() -> dict:
    return {
        "tool": "ask_user",
        "arguments": {
            "assessment": {"statement": "need clarification"},
            "text": "Which metric should be reported?",
            "requested_fields": ["metric"],
        },
    }


def test_direct_inconclusive(tmp_path) -> None:
    scientific = ScientificAgent(
        ScriptedLLMClient(tool_turns([_finish_inconclusive()])),
        store=JsonSessionStore(tmp_path / "sci"),
    )
    controller = _controller(tmp_path, scientific)
    request = ResearchRequest(goal='Is the improvement causal or correlational?', constraints=['Do not request experiments or additional work.'], budget=RunBudget(max_llm_calls=20, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1))

    run = controller.create_run("run_direct", request)

    assert _direct_succeeded(run)


def test_ask_start_then_resume(tmp_path) -> None:
    # First process: the Scientific Agent asks a question and the run pauses.
    sci_dir = tmp_path / "sci"
    controller1 = _controller(
        tmp_path,
        ScientificAgent(ScriptedLLMClient(tool_turns([_ask_user()])), store=JsonSessionStore(sci_dir)),
    )
    request = ResearchRequest(goal='Compare two methods and report accuracy.', budget=RunBudget(max_llm_calls=20, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=2, max_attempts_per_task=2))
    run = controller1.create_run("run_ask", request)

    assert _ask_start_succeeded(run)

    # Second process: a fresh controller and session store over the same dir.
    controller2 = _controller(
        tmp_path,
        ScientificAgent(ScriptedLLMClient(tool_turns([_finish_inconclusive()])), store=JsonSessionStore(sci_dir)),
    )
    answer = UserAnswer(
        question_id=run.pending_question.id,
        values={"metric": "top-1 accuracy"},
        answered_at=datetime.now(UTC),
    )
    resumed = controller2.answer_question("run_ask", answer)

    assert _ask_resume_succeeded(resumed)
