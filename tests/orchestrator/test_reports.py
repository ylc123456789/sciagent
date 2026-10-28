from e2e.scheduler_fixtures import executor_bindings

from scheduler_fixtures import create_scheduler_run as _create_run

from resagent2_contracts import RunPermissions, ExecutionLimits

from resagent2_contracts import (
    AgentOwner,
    AgentResult,
    ArtifactCandidate,
    ModuleStatus,
    ResearchRequest,
    RunBudget,
    TaskProposal,
    WorkflowAgentKind,
    WorkflowProposal,
)
from resagent2_orchestrator import (
    InMemoryRunStore,
    ModuleBinding,
    ScriptedModulePort,
    WorkflowScheduler,
)


def request() -> ResearchRequest:
    return ResearchRequest(goal='Record a report and metrics', budget=RunBudget(max_llm_calls=10, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=5, max_attempts_per_task=2))


def proposal() -> WorkflowProposal:
    return WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[
            TaskProposal(execution_ref="experiment@1",
                id="task_experiment",
                work_request_id="work_legacy_initial",
                workflow_agent_kind="executor",
                instruction="Produce metrics",

            )
        ],
    )


def test_attempt_persists_report_and_registered_artifact() -> None:
    result = AgentResult(
        status=ModuleStatus.COMPLETED,
        report="Measured accuracy",
        artifacts=[ArtifactCandidate(kind="metrics", path="metrics.json", media_type="application/json", summary="Measured metric", content='{"accuracy": 0.9}')],

    )
    engine = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScriptedModulePort([result]),
            )
        }),
        store=InMemoryRunStore(),
    )
    _create_run(engine, "run_report", request(), proposal())
    run = engine.run_until_stable("run_report")

    attempt = run.workflow.tasks[0].attempts[0]
    from resagent2_orchestrator.handoffs import read_json
    assert attempt.report == "Measured accuracy"
    assert read_json(run.artifacts[attempt.artifact_ids[0]]) == {"accuracy": 0.9}
    assert "payload" not in type(attempt).model_fields
