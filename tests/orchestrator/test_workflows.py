from e2e.scheduler_fixtures import executor_bindings

from scheduler_fixtures import create_scheduler_run as _create_run
from scheduler_fixtures import accept_proposal

from control_fixtures import question_signal, ScopedScriptedModulePort
from resagent2_contracts import RunPermissions, ExecutionLimits
from datetime import UTC, datetime


import pytest
from resagent2_orchestrator.handoffs import read_json, system_artifact

from resagent2_contracts import (
    AgentOwner,
    AgentResult,
    ArtifactCandidate,
    Attempt,
    AttemptStatus,
    ControlSignal,
    ErrorCode,
    ModuleError,
    ModuleStatus,
    QuestionDraft,
    RecordedAnswer,
    ResearchRequest,
    RunBudget,
    RunStatus,
    SessionRef,
    SessionStatus,
    TaskProposal,
    TaskStatus,
    UserAnswer,
    WorkflowAgentKind,
    WorkflowPatch,
    WorkflowProposal,
)
from resagent2_orchestrator import (
    InMemoryRunStore,
    ModuleBinding,
    OrchestrationError,
    ResearchRun,
    ScriptedModulePort,
    WorkflowScheduler,
)


NOW = datetime(2026, 8, 26, tzinfo=UTC)


def research_request() -> ResearchRequest:
    return ResearchRequest(goal='Evaluate a method', budget=RunBudget(max_llm_calls=50, timeout_seconds=3600), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=12, max_attempts_per_task=3))


def task(
    task_id: str,
    agent_kind: WorkflowAgentKind,
    depends_on=(),
    *,
    work_request_id: str = "work_legacy_initial",
) -> TaskProposal:
    return TaskProposal(execution_ref=(None if str(agent_kind) == "scientific" else str(agent_kind) + "@1"),
        id=task_id,
        work_request_id=work_request_id,
        workflow_agent_kind=("scientific" if str(agent_kind) == "scientific" else "executor"),
        instruction=f"Complete {task_id}",
        depends_on=list(depends_on),

    )


def completed(summary="done") -> AgentResult:
    return AgentResult(status=ModuleStatus.COMPLETED, report=summary)


def scheduler(scripts: dict[WorkflowAgentKind, list[AgentResult]]) -> WorkflowScheduler:
    owners = {
        "coding": AgentOwner.EXECUTOR,
        "experiment": AgentOwner.EXECUTOR,
    }
    bindings = {
        agent_kind: ModuleBinding(
            owner=owners[agent_kind],
            port=ScopedScriptedModulePort(results),
        )
        for agent_kind, results in scripts.items()
    }
    return WorkflowScheduler(bindings=executor_bindings(bindings), store=InMemoryRunStore())


def _ask_result() -> AgentResult:
    """A paused needs_user_input result, for question-flow tests."""
    return AgentResult(
        status=ModuleStatus.NEEDS_USER_INPUT,
        report="input required",
        artifacts=[], control=question_signal("session_child", 'pick one', ['x']),
        session=SessionRef(
            id="session_child",
            module=AgentOwner.EXECUTOR,
            state_uri="memory://child",
            status=SessionStatus.PAUSED,
            created_at=NOW,
            updated_at=NOW,
        ),
    )


def _pause_with_question(task_id: str, run_id: str) -> ResearchRun:
    engine = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([_ask_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    proposal = WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[task(task_id, "experiment")],
    )
    _create_run(engine, run_id, research_request(), proposal)
    return engine.run_until_stable(run_id)


def test_linear_workflow_runs_to_completion() -> None:
    workflow = WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[
            task("task_code", "coding"),
            task("task_experiment", "experiment", ["task_code"]),
            task("task_analyze", "coding", ["task_experiment"]),
        ],
    )
    engine = scheduler(
        {
            "coding": [completed(), completed()],
            "experiment": [completed()],
        }
    )

    _create_run(engine, "run_linear", research_request(), workflow)
    result = engine.run_until_stable("run_linear")

    assert result.status == RunStatus.RUNNING  # the scheduler never completes a run
    assert [item.status for item in result.workflow.tasks] == [
        TaskStatus.COMPLETED,
        TaskStatus.COMPLETED,
        TaskStatus.COMPLETED,
    ]
    assert [item.attempts[0].number for item in result.workflow.tasks] == [1, 1, 1]


def test_parallel_ready_set_is_stable_and_dependency_driven() -> None:
    proposal = WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[
            task("task_baseline", "experiment"),
            task("task_treatment", "experiment"),
            task(
                "task_analyze",
                "coding",
                ["task_baseline", "task_treatment"],
            ),
        ],
    )
    engine = scheduler(
        {
            "experiment": [completed("baseline"), completed("treatment")],
            "coding": [completed()],
        }
    )
    _create_run(engine, "run_parallel", research_request(), proposal)

    assert engine.ready_task_ids("run_parallel") == [
        "task_baseline",
        "task_treatment",
    ]
    assert engine.ready_task_ids("run_parallel") == [
        "task_baseline",
        "task_treatment",
    ]
    engine.execute_task("run_parallel", "task_baseline")
    assert engine.ready_task_ids("run_parallel") == ["task_treatment"]


def test_question_pauses_and_answer_resumes_same_task_context() -> None:
    question_result = AgentResult(
        status=ModuleStatus.NEEDS_USER_INPUT,
        report="Dataset required",
        artifacts=[], control=question_signal("session_child", 'Which dataset?', ['dataset']),
        session=SessionRef(
            id="session_child",
            module=AgentOwner.EXECUTOR,
            state_uri="memory://child",
            status=SessionStatus.PAUSED,
            created_at=NOW,
            updated_at=NOW,
        ),
    )
    resumed_result = completed().model_copy(update={
        "session": question_result.session.model_copy(update={"status": SessionStatus.COMPLETED}),
    })
    port = ScopedScriptedModulePort([question_result, resumed_result])
    engine = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=port,
            )
        }),
        store=InMemoryRunStore(),
    )
    proposal = WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[task("task_experiment", "experiment")],
    )
    _create_run(engine, "run_question", research_request(), proposal)

    paused = engine.run_until_stable("run_question")
    assert paused.status == RunStatus.PAUSED
    assert paused.pending_question is not None
    # The paused Attempt is non-terminal: no finished_at, and it resumes on the
    # same Attempt number (ADR-0011 §2).
    attempt = paused.workflow.tasks[0].attempts[0]
    assert attempt.status.value == "needs_user_input"
    assert attempt.finished_at is None

    answer = RecordedAnswer(
        question_id=paused.pending_question.id,
        question_text=paused.pending_question.text,
        requested_fields=paused.pending_question.requested_fields,
        run_id=paused.run_id, task_id="task_experiment", attempt_number=1, session_id="session_child",
        values={"dataset": "demo"},
        answered_at=NOW,
    )
    # The controller owns run-level answer state; the scheduler only moves the
    # task back into the ready set.
    run = engine.store.load("run_question")
    run.answers.append(answer)
    run.pending_question = None
    run.status = RunStatus.RUNNING
    engine.resume_task_in_place(run, "task_experiment")
    engine.store.save(run)
    final = engine.run_until_stable("run_question")

    assert final.workflow.tasks[0].status == TaskStatus.COMPLETED
    # The same Attempt resumed, not a new one.
    assert [attempt.number for attempt in final.workflow.tasks[0].attempts] == [1]
    assert port.requests[1].resume_answers == [answer]
    assert port.requests[1].parent_session_id == "session_child"
    assert port.requests[1].attempt_number == 1


def test_long_task_id_question_stays_within_id_cap() -> None:
    paused = _pause_with_question("task_" + "a" * 128, "run_long_question")

    assert paused.status == RunStatus.PAUSED
    assert paused.pending_question is not None
    assert len(paused.pending_question.id) <= 137
    assert paused.pending_question.id.startswith("question_")


def test_question_id_keeps_owning_task_scope() -> None:
    paused = _pause_with_question("task_experiment", "run_short_question")

    assert paused.pending_question is not None
    assert paused.pending_question.id.startswith("question_")
    assert paused.pending_question.task_id == "task_experiment"


def test_successive_questions_in_one_attempt_reject_the_previous_answer() -> None:
    from resagent2_orchestrator.scheduler import _validate_answer

    port = ScopedScriptedModulePort([_ask_result(), _ask_result()])
    engine = WorkflowScheduler(
        bindings=executor_bindings({"experiment": ModuleBinding(
            owner=AgentOwner.EXECUTOR, port=port,
        )}),
        store=InMemoryRunStore(),
    )
    proposal = WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[task("task_experiment", "experiment")],
    )
    _create_run(engine, "run_two_questions", research_request(), proposal)
    first = engine.run_until_stable("run_two_questions")
    answer = RecordedAnswer(
        question_id=first.pending_question.id, values={"x": "first"},
        question_text=first.pending_question.text,
        requested_fields=first.pending_question.requested_fields,
        run_id=first.run_id, task_id="task_experiment", attempt_number=1, session_id="session_child",
        answered_at=NOW,
    )
    run = engine.store.load(first.run_id)
    run.answers.append(answer)
    run.pending_question = None
    run.status = RunStatus.RUNNING
    engine.resume_task_in_place(run, "task_experiment")
    engine.store.save(run)

    # A new scheduler reads the persisted answer/Attempt; no in-memory counter.
    restarted = WorkflowScheduler(bindings=executor_bindings(engine.bindings), store=engine.store)
    second = restarted.run_until_stable(run.run_id)
    assert second.pending_question.id != answer.question_id
    assert len(second.workflow.tasks[0].attempts) == 1
    assert port.requests[1].attempt_number == 1
    assert port.requests[1].parent_session_id == "session_child"
    with pytest.raises(OrchestrationError, match="does not match"):
        _validate_answer(second.pending_question, answer)
    _validate_answer(second.pending_question, UserAnswer(
        question_id=second.pending_question.id, values={"x": "second"},
        answered_at=NOW,
    ))


def test_scheduler_preserves_the_tool_allocated_question_identity() -> None:
    from pydantic import TypeAdapter

    from resagent2_contracts import QuestionId
    signal = question_signal("session_child", "Choose metric", ["metric"])
    result = _ask_result().model_copy(update={"control": signal})
    engine = scheduler({"experiment": [result]})
    _create_run(engine, "run_question_identity", research_request(), WorkflowProposal(
        work_request_id="work_legacy_initial", tasks=[task("task_experiment", "experiment")],
    ))
    paused = engine.run_until_stable("run_question_identity")
    assert paused.pending_question.id == signal.question.id
    assert TypeAdapter(QuestionId).validate_python(paused.pending_question.id) == signal.question.id


def test_failed_attempt_cannot_be_promoted_to_completed() -> None:
    failed = AgentResult(
        status=ModuleStatus.FAILED,
        report="Measurement completed successfully",

        error=ModuleError(
            code=ErrorCode.TOOL_FAILED,
            message="Actual execution failed",
            retryable=False,
        ),
    )
    engine = scheduler({"experiment": [failed]})
    _create_run(engine,
        "run_failed_report",
        research_request(),
        WorkflowProposal(
            work_request_id="work_legacy_initial",
            tasks=[task("task_experiment", "experiment")],
        ),
    )

    run = engine.run_until_stable("run_failed_report")

    assert run.workflow.tasks[0].status == TaskStatus.FAILED


def test_retryable_failure_creates_a_new_attempt_automatically() -> None:
    retryable = AgentResult(
        status=ModuleStatus.FAILED,
        report="temporary failure",
        error=ModuleError(
            code=ErrorCode.TIMEOUT,
            message="temporary timeout",
            retryable=True,
        ),
    )
    engine = scheduler(
        {"experiment": [retryable, completed("retry worked")]}
    )
    _create_run(engine,
        "run_auto_retry",
        research_request(),
        WorkflowProposal(
            work_request_id="work_legacy_initial",
            tasks=[task("task_experiment", "experiment")],
        ),
    )

    run = engine.run_until_stable("run_auto_retry")

    assert run.workflow.tasks[0].status == TaskStatus.COMPLETED
    assert [item.number for item in run.workflow.tasks[0].attempts] == [1, 2]


def test_recovery_closes_interrupted_attempt_and_uses_normal_retry_budget() -> None:
    engine = scheduler({"experiment": [completed("retry worked")]})
    _create_run(engine, "run_interrupted", research_request(), WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[task("task_experiment", "experiment")],
    ))
    run = engine.store.load("run_interrupted")
    workflow_task = run.workflow.tasks[0]
    workflow_task.status = TaskStatus.RUNNING
    workflow_task.attempts.append(
        Attempt(
            number=1,
            status=AttemptStatus.RUNNING,
            started_at=NOW,
        )
    )
    engine.store.save(run)

    recovered = engine.store.load("run_interrupted")
    assert engine._recover_interrupted_attempts_in_place(recovered)
    engine.store.save(recovered)
    interrupted = recovered.workflow.tasks[0].attempts[0]
    assert interrupted.status == AttemptStatus.FAILED
    assert interrupted.error is not None
    assert interrupted.error.code == ErrorCode.INTERRUPTED
    assert recovered.workflow.tasks[0].status == TaskStatus.PENDING

    final = engine.run_until_stable("run_interrupted")
    assert final.workflow.tasks[0].status == TaskStatus.COMPLETED
    assert [item.number for item in final.workflow.tasks[0].attempts] == [1, 2]


def test_budget_exhaustion_does_not_persist_a_running_attempt() -> None:
    engine = scheduler({"experiment": [completed()]})
    _create_run(engine, "run_no_calls", research_request(), WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[task("task_experiment", "experiment")],
    ))
    run = engine.store.load("run_no_calls")
    run.usage.requests = {f"fixture_{i}:0": "succeeded" for i in range(run.request.budget.max_llm_calls)}
    engine.store.save(run)

    with pytest.raises(OrchestrationError, match="LLM-call budget"):
        engine.execute_task("run_no_calls", "task_experiment")

    unchanged = engine.store.load("run_no_calls").workflow.tasks[0]
    assert unchanged.status == TaskStatus.PENDING
    assert unchanged.attempts == []


def test_task_request_receives_full_remaining_run_budget(monkeypatch) -> None:
    from resagent2_orchestrator import scheduler as scheduler_module

    class FixedClock:
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(scheduler_module, "datetime", FixedClock)
    engine = scheduler({"experiment": [completed()]})
    request = ResearchRequest(goal='Use the full remaining budget', budget=RunBudget(max_llm_calls=120, timeout_seconds=3600), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1))
    _create_run(engine, "run_remaining_budget", request, WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[task("task_experiment", "experiment")],
    ))
    run = engine.store.load("run_remaining_budget")
    run.created_at = NOW
    run.updated_at = NOW
    run.usage.requests = {f"fixture_{i}:0": "succeeded" for i in range(17)}

    module_request = engine._module_request(
        run,
        run.workflow.tasks[0],
        1,
        parent_session_id=None,
    )

    assert module_request.budget.max_llm_calls == 103
    assert module_request.budget.timeout_seconds == 3600


def test_malformed_task_delegation_is_a_contract_failure_not_a_question() -> None:
    request_work = AgentResult.model_construct(
        status=ModuleStatus.REQUEST_WORK,
        report="invalid task result",
        artifacts=[ArtifactCandidate(kind="work_request", path="work.json", media_type="application/json", summary="Invalid control", content="{}")],
        control=ControlSignal.model_construct(action="request_work", source_session_id="session_invalid_task", source_turn=1, source_call_id="call_invalid"),
        session=SessionRef(
            id="session_invalid_task",
            module=AgentOwner.EXECUTOR,
            state_uri="session://session_invalid_task",
            status=SessionStatus.PAUSED,
            created_at=NOW,
            updated_at=NOW,
        ),
    )
    engine = scheduler({"experiment": [request_work]})
    _create_run(engine, "run_invalid_request_work", research_request(), WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[task("task_experiment", "experiment")],
    ))

    run = engine.run_until_stable("run_invalid_request_work")
    attempt = run.workflow.tasks[0].attempts[0]
    assert run.pending_question is None
    assert attempt.status == AttemptStatus.FAILED
    assert attempt.error is not None
    assert attempt.error.code == ErrorCode.CONTRACT_ERROR


def test_invalid_module_port_result_becomes_contract_failure() -> None:
    class InvalidPort:
        def invoke(self, request):
            return {"status": "completed"}

    engine = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=InvalidPort(),
            )
        }),
        store=InMemoryRunStore(),
    )
    _create_run(engine,
        "run_invalid_port",
        research_request(),
        WorkflowProposal(
            work_request_id="work_legacy_initial",
            tasks=[task("task_experiment", "experiment")],
        ),
    )

    run = engine.run_until_stable("run_invalid_port")

    error = run.workflow.tasks[0].attempts[0].error
    assert run.workflow.tasks[0].status == TaskStatus.FAILED
    assert error is not None
    assert error.code == ErrorCode.CONTRACT_ERROR


def test_ready_work_keeps_run_running_until_it_is_executed() -> None:
    engine = scheduler({"experiment": [completed()]})
    created = _create_run(engine,
        "run_ready_gate",
        research_request(),
        WorkflowProposal(
            work_request_id="work_legacy_initial",
            tasks=[task("task_experiment", "experiment")],
        ),
    )

    assert created.status == RunStatus.RUNNING
    assert engine.ready_task_ids("run_ready_gate") == ["task_experiment"]
    assert created.workflow.tasks[0].attempts == []


def test_patch_is_append_only() -> None:
    engine = scheduler({"experiment": [completed()]})
    _create_run(engine,
        "run_isolate",
        research_request(),
        WorkflowProposal(
            work_request_id="work_a",
            tasks=[task("task_a", "experiment", work_request_id="work_a")],
        ),
    )
    patched = engine.apply_patch(
        "run_isolate",
        WorkflowPatch(
            work_request_id="work_b",
            based_on_revision=1,
            add_tasks=[task("task_b", "experiment", work_request_id="work_b")],
        ),
    )
    assert [t.id for t in patched.workflow.tasks] == ["task_a", "task_b"]


@pytest.mark.parametrize("kind", ["proposal", "patch"])
def test_empty_candidate_is_rejected_without_mutation(tmp_path, kind) -> None:
    engine = WorkflowScheduler(
        bindings=executor_bindings({}), store=InMemoryRunStore(),
        artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data",
    )
    now = datetime.now(UTC)
    run = ResearchRun(
        run_id="run_candidate", request=research_request(), status=RunStatus.RUNNING,
        created_at=now, updated_at=now,
    )
    if kind == "patch":
        from resagent2_contracts import (Workflow, ArtifactCandidate, ControlSignal)
        run.workflow = Workflow(
            run_id=run.run_id, revision=1, tasks=[],
            created_from="work_legacy_initial",
        )
    engine.store.save(run)
    before = engine.store.load(run.run_id).model_dump()
    with pytest.raises(ValueError, match="empty task graph"):
        if kind == "proposal":
            accept_proposal(engine, run.run_id, WorkflowProposal(
                work_request_id="work_legacy_initial",
                tasks=[],
            ))
        else:
            engine.apply_patch(run.run_id, WorkflowPatch(
                work_request_id="work_next", based_on_revision=1,
                add_tasks=[],
            ))
    assert engine.store.load(run.run_id).model_dump() == before
    assert not engine.run_layout.run_dir(run.run_id).exists()


@pytest.mark.parametrize("old_status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
def test_patch_rejects_prior_round_dependency_without_mutation(tmp_path, old_status) -> None:
    engine = WorkflowScheduler(
        bindings=executor_bindings({"experiment": ModuleBinding(
            owner=AgentOwner.EXECUTOR, port=ScopedScriptedModulePort([]),
        )}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
    )
    run = _create_run(engine, "run_candidate", research_request(), WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[task("task_old", "experiment")],
    ))
    run.workflow.tasks[0].status = old_status
    engine.store.save(run)
    before = engine.store.load(run.run_id).model_dump()
    with pytest.raises(ValueError, match="unknown task"):
        engine.apply_patch(run.run_id, WorkflowPatch(
            work_request_id="work_next", based_on_revision=1,
            add_tasks=[task(
                "task_new", "experiment", ["task_old"],
                work_request_id="work_next",
            )],
        ))
    assert engine.store.load(run.run_id).model_dump() == before

    patched = engine.apply_patch(run.run_id, WorkflowPatch(
        work_request_id="work_next", based_on_revision=1,
        add_tasks=[
            task("task_fix", "experiment", work_request_id="work_next"),
            task("task_rerun", "experiment", ["task_fix"], work_request_id="work_next"),
        ],
    ))
    assert patched.workflow.revision == 2
    assert patched.workflow.tasks[0].status == old_status
    assert patched.workflow.tasks[-1].depends_on == ["task_fix"]
