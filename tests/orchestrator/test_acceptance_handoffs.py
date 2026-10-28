"""Immutable requirements constrain registered outputs of the accepted Attempt."""
from e2e.scheduler_fixtures import executor_bindings

from scheduler_fixtures import accept_proposal

from sciagent_contracts import RunPermissions, ExecutionLimits, WorkspaceAccess

import json
from datetime import UTC, datetime

import pytest

from sciagent_contracts import (
    AgentOwner,
    AgentResult,
    ArtifactCandidate,
    ResearchRequest,
    RunBudget,
    TaskAcceptanceSpec,
    TaskProposal,
    WorkflowProposal,
)
from sciagent_orchestrator import (
    ArtifactRegistrationError, InMemoryRunStore, ModuleBinding, ResearchRun,
    ScriptedModulePort, WorkflowScheduler,
)
from sciagent_orchestrator.handoffs import check_acceptance, read_json


def metrics(value=0.9, *, output_name="scores", path="metrics.json"):
    return ArtifactCandidate(kind="metrics", path=path, output_name=output_name,
        media_type="application/json", summary="Accuracy", content=json.dumps({"accuracy": value}))


def accepted(tmp_path, outputs, *, spec=None):
    port = ScriptedModulePort([AgentResult(status="completed", report="Measured accuracy", artifacts=outputs)])
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({"experiment": ModuleBinding(owner=AgentOwner.EXECUTOR, port=port)}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data",
    )
    now = datetime.now(UTC)
    run = ResearchRun(run_id='run_acceptance', status='running', request=ResearchRequest(goal='Evaluate', budget=RunBudget(max_llm_calls=10, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=2, max_attempts_per_task=2)), created_at=now, updated_at=now)
    scheduler.store.save(run)
    proposal = WorkflowProposal(work_request_id="work_1", tasks=[TaskProposal(execution_ref="experiment@1",
        id="task_measure", work_request_id="work_1", instruction="Measure accuracy",
        workflow_agent_kind="executor", output_names=["scores"],
        acceptance_spec=spec or TaskAcceptanceSpec(required_metric_keys=["accuracy"]),
    )])
    run = accept_proposal(scheduler, run.run_id, proposal)
    return scheduler, port, run, proposal


def test_acceptance_is_frozen_and_shared_by_task_attempt_and_request(tmp_path):
    scheduler, port, run, proposal = accepted(tmp_path, [metrics()])
    requirement = run.workflow.tasks[0].acceptance_ref
    assert requirement == run.artifacts[requirement.id]
    assert read_json(requirement, TaskAcceptanceSpec).required_output_names == ["scores"]
    proposal.tasks[0].acceptance_spec.required_metric_keys.append("invented_after_acceptance")
    completed = scheduler.run_until_stable(run.run_id)
    task = completed.workflow.tasks[0]
    assert task.status == "completed"
    assert task.acceptance_ref == requirement == task.attempts[0].acceptance_ref
    assert requirement in port.requests[0].input_artifacts
    assert read_json(requirement, TaskAcceptanceSpec).required_metric_keys == ["accuracy"]


@pytest.mark.parametrize("value", [True, "0.9", None, float("nan"), float("inf")])
def test_required_metrics_must_be_finite_numbers(tmp_path, value):
    scheduler, _, run, _ = accepted(tmp_path, [metrics(value)])
    failed = scheduler.run_until_stable(run.run_id)
    task = failed.workflow.tasks[0]
    assert task.status == "failed"
    assert "required numeric metric keys" in task.attempts[0].error.message


@pytest.mark.parametrize("outputs,diagnostic,registered_count", [
    pytest.param([metrics(output_name=None)], "required logical outputs", 1, id="missing-required-name"),
    pytest.param([metrics(), metrics(path="other.json")], "duplicate logical output name", 0,
                 id="duplicate-name-rejected-before-publication"),
])
def test_declared_output_names_require_one_registered_artifact(tmp_path, outputs, diagnostic, registered_count):
    scheduler, _, run, _ = accepted(tmp_path, outputs)
    existing_ids = set(run.artifacts)
    failed = scheduler.run_until_stable(run.run_id)
    task = failed.workflow.tasks[0]
    attempt = task.attempts[0]
    assert task.status == "failed"
    assert diagnostic in attempt.error.message
    assert attempt.error.retryable is False
    # Acceptance runs after freezing, whereas duplicate names reject the entire group before publication.
    assert len(attempt.artifact_ids) == registered_count
    assert set(failed.artifacts) - existing_ids == {*attempt.artifact_ids, attempt.report_artifact_id}
    assert attempt.report == "Measured accuracy"
    assert attempt.report_artifact_id not in attempt.artifact_ids
    if registered_count:
        assert read_json(failed.artifacts[attempt.artifact_ids[0]]) == {"accuracy": 0.9}
    assert not failed.pending_registrations and not failed.pending_module_results


def test_attempt_cannot_replace_its_accepted_requirement(tmp_path):
    scheduler, _, run, _ = accepted(tmp_path, [metrics()])
    completed = scheduler.run_until_stable(run.run_id)
    task = completed.workflow.tasks[0]
    attempt = task.attempts[0].model_copy(update={"acceptance_ref": None})
    with pytest.raises(ArtifactRegistrationError, match="binding differs"):
        check_acceptance(completed, task, attempt, [completed.artifacts[key] for key in attempt.artifact_ids])


def test_semantic_execution_policy_is_not_an_exact_delivery_requirement():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Extra inputs"):
        TaskAcceptanceSpec(require_successful_execution=True)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_acceptance_checks_delivery_without_inferring_task_semantics(tmp_path, exit_code):
    record = ArtifactCandidate(
        kind="execution_record", path="execution_record.json",
        media_type="application/json", summary="Actual execution",
        content=json.dumps({"results": [dict(
            command="python verify.py", exit_code=exit_code, timed_out=False,
            stdout_path="out.log", stderr_path="err.log", duration_seconds=0.1,
        )]}),
    )
    scheduler, _, run, _ = accepted(
        tmp_path, [metrics(), record],
        spec=TaskAcceptanceSpec(required_metric_keys=["accuracy"]),
    )
    outcome = scheduler.run_until_stable(run.run_id)
    task = outcome.workflow.tasks[0]
    assert task.status == "completed"
    execution = next(outcome.artifacts[key] for key in task.attempts[0].artifact_ids
                     if outcome.artifacts[key].kind == "execution_record")
    assert read_json(execution)["results"][0]["exit_code"] == exit_code


def test_registry_derives_required_paths_and_ignores_claimed_metadata(tmp_path):
    from sciagent_contracts import WorkspaceGrant
    from sciagent_orchestrator import ArtifactRegistry
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "actual.json").write_text('{"accuracy":0.9}')
    registry = ArtifactRegistry(tmp_path / "registered")
    common = dict(grant=WorkspaceGrant(root=str(workspace), source='local', access=WorkspaceAccess(read_paths=['.'])), producer="executor", run_id='run_test', task_id='task_test', attempt_number=1)
    candidate = metrics(path="actual.json").model_copy(update={"content": None, "metadata": {"source_path": "fake.json"}})
    ref = registry.register(candidate, index=1, **common)
    assert ref.metadata["source_path"] == "actual.json"
    inline = registry.register(candidate.model_copy(update={"content": "{}"}), index=2, **common)
    assert "source_path" not in inline.metadata


def test_registry_accepts_controlled_output_files_but_rejects_escape_and_ambiguity(tmp_path):
    from sciagent_contracts import WorkspaceGrant
    from sciagent_orchestrator import ArtifactRegistry
    workspace, output = tmp_path / "workspace", tmp_path / "output"
    workspace.mkdir()
    output.mkdir()
    (output / "metrics.json").write_text('{"accuracy":0.9}')
    registry = ArtifactRegistry(tmp_path / "registered")
    candidate = metrics().model_copy(update={"content": None})
    common = dict(grant=WorkspaceGrant(root=str(workspace), source='local', access=WorkspaceAccess(read_paths=['.'])), output_dir=str(output), producer="executor", run_id='run_test', task_id='task_test', attempt_number=1)
    ref = registry.register(candidate, index=1, **common)
    assert ref.metadata["source_root"] == "output_dir"
    (workspace / "metrics.json").write_text("{}")
    with pytest.raises(ArtifactRegistrationError, match="ambiguous"):
        registry.register(candidate, index=2, **common)
    (workspace / "metrics.json").unlink()
    (output / "metrics.json").unlink()
    (tmp_path / "outside.json").write_text("{}")
    (output / "metrics.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ArtifactRegistrationError, match="outside"):
        registry.register(candidate, index=2, **common)
