"""Deterministic scientific -> execution -> evidence -> conclusion loop."""

from sciagent_contracts import RunPermissions, ExecutionLimits

import json
import tempfile
from pathlib import Path
from sciagent_contracts import (
    AgentOwner, AgentResult, ArtifactCandidate, ModuleStatus, ResearchRequest,
    RunBudget, RunStatus, TaskAcceptanceSpec, TaskProposal, WorkflowAgentKind,
    WorkflowProposal,
)
from sciagent_orchestrator import (
    DeterministicWorkInterpreter,
    FixedWorkflowCompiler, JsonRunStore, ModuleBinding, ResearchController,
    ScriptedModulePort, WorkflowScheduler,
)
from sciagent_executor import default_execution_registry
from sciagent_runtime import InMemorySessionStore, ScriptedLLMClient
from e2e.native_fixtures import tool_turn, tool_turns
from sciagent_scientific import ScientificAgent

RUN_ID = "run_golden"
WORK_REQUEST_ID = "work_1"


def registry():
    return default_execution_registry()


def completed_result(kind):
    artifacts = []
    if kind == "experiment":
        artifacts.append(ArtifactCandidate(kind="metrics", path="metrics.json", media_type="application/json",
                                           summary="Measured accuracy", content='{"accuracy": 0.9}'))
    return AgentResult(status=ModuleStatus.COMPLETED, report="done", artifacts=artifacts)


def request_work_action():
    return {"tool": "request_work", "arguments": {
        "assessment": {"statement": "need evidence"},
        "work_request": {"objective": "Produce evidence for the method", "tasks": [
            {"key": "code", "instruction": "Prepare the method implementation",
             "selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Implementation"}},
            {"key": "experiment", "instruction": "Run comparison and provide measured accuracy",
             "depends_on": ["code"],
             "selection": {"kind": "preset", "preset_ref": "experiment@1", "reason": "Measurement"}},
        ]}}}


def finish_action(artifact_id):
    return {"tool": "finish", "arguments": {"report": "done", "artifacts": [{
        "kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json",
        "summary": "Scientific conclusion", "content": {"verdict": "inconclusive", "statement": "done",
        "evidence_artifact_ids": [artifact_id], "limitations": ["Deterministic fixture only"]},
    }]}}


def run_mock_e2e(*, workdir: Path | None = None):
    workdir = workdir or Path(tempfile.mkdtemp(prefix="sciagent-e2e-"))
    scheduler = WorkflowScheduler(
        bindings={WorkflowAgentKind.EXECUTOR: ModuleBinding(
            owner=AgentOwner.EXECUTOR, port=ScriptedModulePort([
                completed_result("coding"), completed_result("experiment"),
            ]))},
        store=JsonRunStore(workdir / "state"), artifact_root=workdir / "artifacts", data_root=workdir,
    )
    class GoldenClient(ScriptedLLMClient):
        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            turn = super().next_tool_call(context, schemas, turns, max_input_tokens=max_input_tokens)
            call = turn.tool_calls[0]
            if call.name == "request_work":
                return turn
            run = scheduler.store.load(RUN_ID)
            evidence = next(ref for ref in run.artifacts.values() if ref.kind == "metrics")
            assert evidence.id in context.text
            if call.name == "read_artifact":
                return tool_turn("read_artifact", {"artifact_id": evidence.id}, call_id=call.id)
            action = finish_action(evidence.id)
            return tool_turn(action["tool"], action["arguments"], call_id=call.id)

    scientific = ScientificAgent(GoldenClient(tool_turns([
        request_work_action(), {"tool": "read_artifact"}, {"tool": "finish"},
    ])), store=InMemorySessionStore())
    controller = ResearchController(interpreter=DeterministicWorkInterpreter(), scientific_port=scientific, compiler=FixedWorkflowCompiler(),
                                    scheduler=scheduler, registry=registry())
    return controller.create_run(RUN_ID, ResearchRequest(goal='Determine whether the method improves accuracy', budget=RunBudget(max_llm_calls=50, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=5, max_attempts_per_task=3)))


def _summarize(run):
    lines = [f"run={run.run_id} status={run.status.value} artifacts={len(run.artifacts)}",
             f"opinion={run.final_opinion.statement if run.final_opinion else None}", f"report={run.final_report_artifact_id}"]
    for task in run.workflow.tasks if run.workflow else []:
        attempts = ", ".join(f"{a.number}:{a.status.value}" for a in task.attempts)
        lines.append(f"  {task.id} [{task.workflow_agent_kind.value}] attempts={attempts}")
    if run.terminal_error:
        lines.append(run.terminal_error.message)
    return "\n".join(lines)


def main():
    run = run_mock_e2e()
    print(_summarize(run))
    assert run.status == RunStatus.COMPLETED


if __name__ == "__main__":
    main()
