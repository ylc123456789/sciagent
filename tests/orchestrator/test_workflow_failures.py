"""Task input and dependency failures terminate with persistent, factual outcomes."""
from sciagent_executor import default_execution_registry
from e2e.scheduler_fixtures import executor_bindings

from scheduler_fixtures import add_root_owner, execution_tasks

from datetime import UTC, datetime

import pytest

from sciagent_contracts import (
    AgentOwner, AgentResult, ArtifactCandidate, ModuleError, ResearchRequest, RunBudget,
    RunPermissions, TaskProposal, WorkRequest, WorkRequestDraft, WorkflowProposal,
)
from sciagent_orchestrator import (
    InMemoryRunStore, JsonRunStore, ModuleBinding, ResearchRun, ScriptedModulePort, WorkflowScheduler,
)


def scheduler(tmp_path, proposals, results, *, store=None):
    port = ScriptedModulePort(results)
    engine = WorkflowScheduler(
        bindings=executor_bindings({"experiment": ModuleBinding(owner=AgentOwner.EXECUTOR, port=port)}),
        store=store or InMemoryRunStore(), artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data",
    )
    now = datetime.now(UTC)
    run = ResearchRun(
        run_id="run_failure", status="running", created_at=now, updated_at=now,
        request=ResearchRequest(goal="Evaluate", budget=RunBudget(max_llm_calls=10, timeout_seconds=60),
                                permissions=RunPermissions()),
    )
    run.execution_definitions = default_execution_registry().execution_definitions
    run.execution_presets = default_execution_registry().presets
    engine.store.save(run)
    run = engine.accept_proposal(run.run_id, WorkflowProposal(work_request_id="work_test", tasks=proposals))
    run.work_requests.append(WorkRequest(owner_task_id="task_root", owner_attempt_number=1, depth=2,
        id="work_test", run_id=run.run_id, owner_session_id="session_test",
        request=WorkRequestDraft(objective='Evaluate the task results', tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Evaluate the task results'}]), status="executing",
        workflow_revision=run.workflow.revision, created_at=now, updated_at=now,
    ))
    add_root_owner(run, "session_test")
    engine.store.save(run)
    return engine, port, run.run_id


def task(task_id, dependencies=(), **kwargs):
    return TaskProposal(execution_ref="experiment@1", id=task_id, work_request_id="work_test", workflow_agent_kind="executor",
                        instruction=f"Complete {task_id}", depends_on=list(dependencies), **kwargs)


@pytest.mark.parametrize("stage", ["request", "invoke", "reception"])
@pytest.mark.parametrize("error", [RuntimeError(), OSError(), RuntimeError("   "), RuntimeError("Original message")],
                         ids=["empty-runtime-error", "empty-os-error", "whitespace-only", "with-message"])
def test_boundary_exceptions_persist_failed_attempt(tmp_path, monkeypatch, stage, error):
    store = JsonRunStore(tmp_path / "state")
    engine, port, run_id = scheduler(tmp_path, [
        task("task_source"), task("task_followup", ["task_source"]),
    ], [AgentResult(status="completed", report="Task completed")], store=store)

    def reject(*args, **kwargs):
        raise error

    if stage == "request":
        monkeypatch.setattr(engine, "_module_request", reject)
    elif stage == "invoke":
        monkeypatch.setattr(port, "invoke", reject)
    else:
        monkeypatch.setattr("sciagent_orchestrator.scheduler.receive_artifacts", reject)

    result = engine.run_until_stable(run_id)
    persisted = JsonRunStore(store.root).load(run_id)

    assert result == persisted
    source, followup = execution_tasks(persisted)
    assert source.status == "failed"
    assert len(source.attempts) == 1
    attempt = source.attempts[0]
    assert attempt.status == "failed"
    assert attempt.finished_at is not None
    assert attempt.error.message == (str(error).strip() or type(error).__name__)
    if stage == "reception":
        assert attempt.report == "Task completed" and attempt.reception_error is not None
    else:
        assert attempt.error.message in attempt.report
    expected_code = "artifact_missing" if stage == "reception" and isinstance(error, OSError) else "contract_error"
    assert attempt.error.code == expected_code
    assert not attempt.error.retryable
    assert followup.status == "blocked"
    assert not followup.attempts
    assert persisted.work_requests[0].status == "stable"


def test_empty_reception_error_preserves_original_agent_failure(tmp_path, monkeypatch):
    original = ModuleError(
        code="tool_failed", message="Training failed", retryable=True,
        details={"stderr_tail": "Diagnostic"},
    )
    engine, _, run_id = scheduler(tmp_path, [task("task_source")], [
        AgentResult(status="failed", report="Training failed", error=original),
    ], store=JsonRunStore(tmp_path / "state"))

    def reject(*args, **kwargs):
        raise RuntimeError()

    monkeypatch.setattr("sciagent_orchestrator.scheduler.receive_artifacts", reject)
    result = engine.run_until_stable(run_id)
    assert result == engine.store.load(run_id)
    attempt = result.workflow.tasks[0].attempts[0]
    assert attempt.status == "failed"
    assert attempt.error.code == original.code
    assert attempt.error.message == original.message
    assert attempt.error.details == {
        "stderr_tail": "Diagnostic", "artifact_registration_error": "RuntimeError",
    }
    assert attempt.error.retryable is False
    assert attempt.report == "Training failed"
    assert attempt.reception_error.details["artifact_registration_error"] == "RuntimeError"


@pytest.mark.parametrize("fault", ["missing", "ambiguous"])
def test_future_artifact_binding_failure_persists_failed_attempt(tmp_path, fault):
    engine, port, run_id = scheduler(tmp_path, [
        task("task_producer", output_names=["analysis"]),
        task("task_consumer", ["task_producer"], input_artifact_bindings=[
            {"source_task": "task_producer", "output_selector": "analysis"},
        ]),
        task("task_followup", ["task_consumer"]),
    ], [AgentResult(status="completed", report="Produced analysis", artifacts=[ArtifactCandidate(
        kind="analysis", output_name="analysis", path="analysis.txt", media_type="text/plain",
        summary="Analysis", content="Ready for follow-up",
    )])])
    run = engine.execute_task(run_id, "task_producer")
    source_attempt = run.workflow.tasks[0].attempts[0]
    ref = next(run.artifacts[key] for key in source_attempt.artifact_ids
               if run.artifacts[key].output_name == "analysis")
    # Fault-inject the persisted output registry after the producer was accepted.
    if fault == "missing":
        run.artifacts[ref.id] = ref.model_copy(update={"output_name": "unavailable"})
    else:
        duplicate = ref.model_copy(update={"id": "artifact_duplicate"})
        run.artifacts[duplicate.id] = duplicate
        source_attempt.artifact_ids.append(duplicate.id)
    engine.store.save(run)

    result = engine.run_until_stable(run_id)

    assert result == engine.store.load(run_id)
    producer, consumer, followup = execution_tasks(result)
    assert producer.status == "completed"
    assert consumer.status == "failed"
    assert len(consumer.attempts) == 1
    attempt = consumer.attempts[0]
    assert attempt.status == "failed"
    assert attempt.finished_at is not None
    assert attempt.error.code == "contract_error"
    assert not attempt.error.retryable
    assert "missing or ambiguous" in attempt.error.message
    assert followup.status == "blocked"
    assert len(port.requests) == 1
    assert result.work_requests[0].status == "stable"


@pytest.mark.parametrize("order", [
    ["task_d", "task_c", "task_b", "task_a"],
    ["task_c", "task_a", "task_d", "task_b"],
])
@pytest.mark.parametrize("failure_status", ["failed", "blocked"])
def test_unordered_dependency_failure_reaches_terminal_closure(tmp_path, order, failure_status):
    dependencies = {"task_a": [], "task_b": ["task_a"], "task_c": ["task_b"], "task_d": ["task_c"]}
    engine, port, run_id = scheduler(
        tmp_path, [task(key, dependencies[key]) for key in order],
        [AgentResult(status=failure_status, report="Cannot execute", error=ModuleError(
            code="tool_failed", message="Source task cannot complete", retryable=False,
        ))],
    )

    result = engine.run_until_stable(run_id)

    assert result == engine.store.load(run_id)
    expected = {key: failure_status if key == "task_a" else "blocked" for key in order}
    assert {item.id: item.status.value for item in execution_tasks(result)} == expected
    work = result.work_requests[0]
    assert work.status == "stable"
    assert {item.task_id: item.status for item in work.outcome.tasks} == expected
    assert [request.task_id for request in port.requests] == ["task_a"]
    assert all(not item.attempts for item in execution_tasks(result) if item.id != "task_a")
