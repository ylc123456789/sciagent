from e2e.scheduler_fixtures import create_scheduler_run

from e2e.native_fixtures import tool_turns

from sciagent_contracts import WorkspaceAccess, RunPermissions, ExecutionLimits
from datetime import UTC, datetime
import shlex
import sys
import subprocess

from sciagent_contracts import (
    AgentOwner,
    WorkflowAgentKind,
    ResearchRequest,
    RecordedAnswer,
    RunBudget,
    RunStatus,
    TaskProposal,
    TaskAcceptanceSpec,
    TaskStatus,
    WorkflowProposal,
    WorkspaceSourceKind,
    WorkspaceSpec,
)
from sciagent_experiment import NativeExperimentAgent
from sciagent_runtime import ScriptedLLMClient
from sciagent_orchestrator import (
    InMemoryRunStore,
    ModuleBinding,
    WorkflowScheduler,
)


def _create_run(engine, run_id, request, proposal):
    return create_scheduler_run(engine, run_id, request, proposal, resolve_workspaces=True)


def _native_experiment():
    return NativeExperimentAgent(ScriptedLLMClient(tool_turns([
        {"tool": "run_shell", "arguments": {
            "command": f"{shlex.quote(sys.executable)} train.py",
        }},
        {"tool": "finish", "arguments": {
            "report": "Recorded the actual measured accuracy",
            "artifacts": [{
                "kind": "experiment_result", "path": "metrics.json",
                "media_type": "application/json", "summary": "Measured accuracy",
            }],
        }},
    ])))


def test_scheduler_registers_native_experiment_artifacts(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    (workspace / "train.py").write_text(
        "from pathlib import Path\nPath('metrics.json').write_text('{\"accuracy\": 0.9}')\n"
    )
    scheduler = WorkflowScheduler(bindings={WorkflowAgentKind.EXECUTOR: ModuleBinding(owner=AgentOwner.EXECUTOR, port=_native_experiment())}, store=InMemoryRunStore(), artifact_root=tmp_path / 'artifacts', data_root=tmp_path / 'data', workspaces={'ws_main': WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.LOCAL, location=str(workspace), access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))})
    request = ResearchRequest(goal='Run the experiment', budget=RunBudget(max_llm_calls=10, timeout_seconds=30), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1))
    proposal = WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[
            TaskProposal(
                id="task_experiment_native",
                work_request_id="work_legacy_initial",
                workflow_agent_kind=WorkflowAgentKind.EXECUTOR, execution_ref="experiment@1",
                instruction="Run train.py and record accuracy",
                acceptance_spec=TaskAcceptanceSpec(
                    required_metric_keys=["accuracy"],
                    required_artifact_paths=["metrics.json"],
                ),
            )
        ],
    )

    _create_run(scheduler, "run_native_experiment", request, proposal)
    run = scheduler.run_until_stable("run_native_experiment")

    assert run.status == RunStatus.PAUSED
    assert run.pending_question is not None
    assert not (workspace / "metrics.json").exists()
    question = run.pending_question
    attempt = run.workflow.tasks[0].attempts[0]
    session_id = attempt.session.id
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, answered_at=datetime.now(UTC),
        run_id=run.run_id, session_id=question.session_id, task_id="task_experiment_native", attempt_number=1,
        action=question.action,
    )
    run.answers.append(answer)
    run.pending_question = None
    run.status = RunStatus.RUNNING
    scheduler.resume_task_in_place(run, "task_experiment_native")
    scheduler.store.save(run)
    run = scheduler.run_until_stable(run.run_id)

    assert run.workflow.tasks[0].status == TaskStatus.COMPLETED
    assert len(run.workflow.tasks[0].attempts) == 1
    assert run.workflow.tasks[0].attempts[0].session.id == session_id
    artifacts = list(run.artifacts.values())
    assert {artifact.kind for artifact in artifacts} == {
        "experiment_result", "execution_record", "module_report", "acceptance_requirements",
    }
    assert all(len(artifact.sha256) == 64 for artifact in artifacts)
