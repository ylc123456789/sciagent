"""Tests for the Phase 7.5 ResearchController (DEVELOPMENT_PLAN §7.5)."""
from sciagent_executor import default_execution_registry
from e2e.scheduler_fixtures import executor_bindings
from control_fixtures import question_signal, ScopedScriptedModulePort, ParentQuestionClient

from e2e.native_fixtures import tool_turn, tool_turns

from sciagent_contracts import RunPermissions, ExecutionLimits

from datetime import UTC, datetime
import json

import pytest

from sciagent_contracts import (
    AgentOwner,
    AgentResult,
    ArtifactCandidate,
    ConclusionRequirements,
    ControlSignal,
    DatasetRef,
    ErrorCode,
    ModuleError,
    ModuleStatus,
    QuestionDraft,
    RecordedAnswer,
    ResearchRequest,
    RunBudget,
    RunStatus,
    ScientificOpinion,
    ScientificVerdict,
    SessionRef,
    SessionStatus,
    TaskProposal,
    task_session_id,
    UserAnswer,
    WorkRequest,
    WorkRequestDraft,
    WorkRequestStatus,
    WorkflowAgentKind,
    WorkflowAgentRegistry,
    WorkflowPatch,
    WorkflowProposal,
)
from sciagent_orchestrator import (
    DeterministicWorkInterpreter,
    CompilationError,
    CompilationResult,
    DeterministicWorkflowCompiler,
    FixedWorkflowCompiler,
    InMemoryRunStore,
    JsonRunStore,
    ModuleBinding,
    ResearchController,
    ResearchRun,
    ScriptedModulePort,
    WorkflowScheduler,
)
from sciagent_runtime import InMemorySessionStore, JsonSessionStore, PromptLLMClient, ScriptedLLMClient
from sciagent_scientific import ScientificAgent
from sciagent_orchestrator.handoffs import read_json, system_artifact

NOW = datetime(2026, 8, 28, tzinfo=UTC)


def _context_answer(context, answer):
    assert context.included_sections.count("task") == 1
    return next(value for value in [e["record"] for e in json.loads(context.task_context.split("\n", 1)[1])["business_dialogue"] if e["type"] == "answer"]
                if value["question_id"] == answer.question_id)


def _answer_projection(answer):
    return answer.model_dump(mode="json", exclude_none=True)


@pytest.fixture
def run_clock(monkeypatch):
    from datetime import timedelta
    from sciagent_orchestrator import controller as controller_module
    from sciagent_orchestrator import scheduler as scheduler_module

    class Clock:
        value = datetime.now(UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.value

        @classmethod
        def advance(cls, seconds):
            cls.value += timedelta(seconds=seconds)

    monkeypatch.setattr(controller_module, "datetime", Clock)
    monkeypatch.setattr(scheduler_module, "datetime", Clock)
    monkeypatch.setattr("sciagent_runtime.loop.datetime", Clock)
    return Clock


def test_user_wait_is_excluded_after_restart_without_resetting_budget(tmp_path, run_clock):
    from datetime import timedelta
    from sciagent_orchestrator import OrchestrationError

    ask = {
        "tool": "ask_user",
        "arguments": {"assessment": {"statement": "Need a user decision"}, 'text': 'Prepare the resources, then confirm.', 'requested_fields': ['ready']},
    }

    class TimedClient(ScriptedLLMClient):
        tool_session_key = "test-native-tools/v1"

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            run_clock.advance(10)  # Real work before each pause/finish must still count.
            return super().next_tool_call(context, schemas, turns, max_input_tokens=max_input_tokens)

    def rebuild(actions):
        controller = _build_recoverable_controller(
            JsonRunStore(tmp_path / "runs"), JsonSessionStore(tmp_path / "sessions"), actions,
        )
        controller.scientific_port.llm_client = TimedClient(tool_turns(actions))
        return controller

    first = rebuild([ask])
    paused = first.create_run("run_wait", research_request())
    assert paused.remaining_timeout_seconds(run_clock.now()) == 50
    assert paused.llm_calls_used == 1
    run_clock.advance(2 * 86400)
    assert paused.remaining_timeout_seconds(run_clock.now()) == 50

    second = rebuild([ask])
    budgets = []
    original_run = second.scientific_port.invoke

    def capture(request):
        budgets.append((
            request.budget.max_llm_calls,
            request.budget.timeout_seconds,
        ))
        return original_run(request)

    second.scientific_port.invoke = capture
    answer = UserAnswer(
        question_id=paused.pending_question.id, values={"ready": "yes"},
        answered_at=run_clock.now() + timedelta(days=100),  # Not the trusted clock.
    )
    again = second.answer_question(paused.run_id, answer)
    assert again.status == RunStatus.PAUSED
    assert budgets == [(49, 50)]
    assert again.user_wait_seconds == 2 * 86400
    assert again.remaining_timeout_seconds(run_clock.now()) == 40
    assert again.llm_calls_used == 2
    assert again.scientific_session.id == paused.scientific_session.id
    before = second.scheduler.store.load(paused.run_id).model_dump_json()
    with pytest.raises(OrchestrationError, match="does not match"):
        second.answer_question(paused.run_id, answer)
    assert second.scheduler.store.load(paused.run_id).model_dump_json() == before

    run_clock.advance(3600)
    third = rebuild([finish_action()])
    completed = third.answer_question(again.run_id, UserAnswer(
        question_id=again.pending_question.id, values={"ready": "yes"},
        answered_at=run_clock.now(),
    ))
    assert completed.status == RunStatus.COMPLETED
    assert completed.user_wait_seconds == 2 * 86400 + 3600
    assert completed.remaining_timeout_seconds(run_clock.now()) == 30
    assert completed.llm_calls_used == 3


def test_repeated_interrupt_preserves_manual_wait_after_restart(tmp_path, run_clock):
    from sciagent_runtime.cancellation import current_cancellation
    from sciagent_runtime.controls import RunControlInbox

    class TimedClient(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            run_clock.advance(5)
            return super().next_tool_call(*args, **kwargs)

    class InterruptFirstClient(TimedClient):
        def next_tool_call(self, *args, **kwargs):
            reply = super().next_tool_call(*args, **kwargs)
            current_cancellation().cancel("Pause during the first model call")
            return reply

    def rebuild(client):
        controller = _build_recoverable_controller(
            JsonRunStore(tmp_path / "runs"), JsonSessionStore(tmp_path / "sessions"), [],
        )
        controller.control_inbox = RunControlInbox(tmp_path / "controls")
        controller.scientific_port.llm_client = client
        return controller

    first = rebuild(InterruptFirstClient(tool_turns([finish_action()])))
    paused = first.create_run("run_manual_wait", research_request())
    assert paused.status == RunStatus.PAUSED
    assert paused.remaining_timeout_seconds(run_clock.now()) == 55
    original_pause = paused.manually_paused_at
    run_clock.advance(120)
    assert paused.remaining_timeout_seconds(run_clock.now()) == 55

    second = rebuild(TimedClient(tool_turns([finish_action()])))
    second.control_inbox.submit(paused.run_id, "interrupt")
    paused_again = second.resume_run(paused.run_id)
    assert paused_again.status == RunStatus.PAUSED
    assert paused_again.manually_paused_at == original_pause
    assert paused_again.remaining_timeout_seconds(run_clock.now()) == 55
    run_clock.advance(30)
    completed = second.resume_run(paused.run_id)
    assert completed.status == RunStatus.COMPLETED, completed.terminal_error
    assert completed.user_wait_seconds == 150
    assert completed.manually_paused_at is None
    assert completed.remaining_timeout_seconds(run_clock.now()) == 50
    assert completed.llm_calls_used == 2


def test_interrupt_waiting_root_question_counts_overlap_once(tmp_path, run_clock):
    from sciagent_runtime.controls import RunControlInbox

    class TimedClient(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            run_clock.advance(5)
            return super().next_tool_call(*args, **kwargs)

    ask = {"tool": "ask_user", "arguments": {
        "assessment": {"statement": "Need a user decision"},
        "text": "Choose the direction", "requested_fields": ["direction"],
    }}
    controller = _build_recoverable_controller(
        JsonRunStore(tmp_path / "runs"), JsonSessionStore(tmp_path / "sessions"), [],
    )
    controller.control_inbox = RunControlInbox(tmp_path / "controls")
    controller.scientific_port.llm_client = TimedClient(tool_turns([ask, finish_action()]))
    waiting = controller.create_run("run_question_manual_wait", research_request())
    original_question = waiting.pending_question
    run_clock.advance(120)
    controller.control_inbox.submit(waiting.run_id, "interrupt")
    stopped = controller.resume_run(waiting.run_id)
    assert stopped.status == RunStatus.PAUSED
    assert stopped.pending_question == original_question
    run_clock.advance(30)
    assert stopped.remaining_timeout_seconds(run_clock.now()) == 55
    completed = controller.answer_question(waiting.run_id, UserAnswer(
        question_id=original_question.id, values={"direction": "Continue"},
        answered_at=run_clock.now(),
    ))
    assert completed.status == RunStatus.COMPLETED, completed.terminal_error
    assert completed.user_wait_seconds == 150
    assert completed.manually_paused_at is None
    assert completed.pending_question is None
    assert completed.remaining_timeout_seconds(run_clock.now()) == 50
    assert completed.answers[0].question_id == original_question.id


def test_interrupt_cancelled_child_question_settles_wait_before_manual_pause(tmp_path, run_clock):
    from sciagent_runtime.controls import RunControlInbox

    controller = build_controller(actions=[
        request_work_action(), finish_action(limitations=["The interrupted task remains incomplete"]),
    ])
    controller.control_inbox = RunControlInbox(tmp_path / "controls")
    controller.scheduler.bindings["executor"] = ModuleBinding(
        owner=AgentOwner.EXECUTOR,
        port=ScopedScriptedModulePort([_task_question_result()]),
    )
    waiting = controller.create_run("run_cancel_question_wait", research_request())
    assert waiting.pending_question.related_question_id
    run_clock.advance(120)
    controller.control_inbox.submit(waiting.run_id, "interrupt")
    stopped = controller.resume_run(waiting.run_id)
    assert stopped.status == RunStatus.PAUSED
    assert stopped.pending_question is None
    assert not stopped.child_questions
    assert child_work(stopped)[0].status == WorkRequestStatus.CANCELLED
    assert stopped.user_wait_seconds == 120
    run_clock.advance(30)
    assert stopped.remaining_timeout_seconds(run_clock.now()) == 60
    completed = controller.resume_run(waiting.run_id)
    assert completed.status == RunStatus.COMPLETED, completed.terminal_error
    assert completed.user_wait_seconds == 150
    assert completed.manually_paused_at is None
    assert completed.remaining_timeout_seconds(run_clock.now()) == 60


def test_task_wait_uses_same_clock_and_refreshes_resources(run_clock):
    class Source:
        refs = []

        def references(self):
            return list(self.refs)

    source = Source()
    controller = build_controller(
        actions=[request_work_action(), finish_action()], dataset_ref_source=source,
    )
    port = ScopedScriptedModulePort([_task_question_result(), completed_result()])
    controller.scheduler.bindings["executor"] = ModuleBinding(
        owner=AgentOwner.EXECUTOR, port=port,
    )
    paused = controller.create_run("run_task_wait", research_request())
    assert paused.status == RunStatus.PAUSED
    attempt = execution_tasks(paused)[0].attempts[0]
    run_clock.advance(86400)
    source.refs = [DatasetRef(dataset_id="new_data", relative_path="new-data")]
    completed = controller.answer_question(paused.run_id, UserAnswer(
        question_id=paused.pending_question.id,
        values={"dataset": "ready"}, answered_at=run_clock.now(),
    ))
    assert completed.status == RunStatus.COMPLETED
    assert completed.user_wait_seconds == 86400
    assert port.requests[-1].budget.max_llm_calls == 47
    assert port.requests[-1].budget.timeout_seconds == 60
    from sciagent_orchestrator.handoffs import read_json
    catalog = next(ref for ref in port.requests[-1].input_artifacts if ref.kind == "dataset_catalog")
    assert read_json(catalog)["datasets"] == [ref.model_dump(mode="json") for ref in source.refs]
    assert port.requests[-1].parent_session_id == attempt.session.id
    assert port.requests[-1].attempt_number == attempt.number
    assert port.requests[-1].output_dir == port.requests[0].output_dir
    assert len(execution_tasks(completed)[0].attempts) == 1


def test_ordinary_downtime_is_not_user_wait(run_clock):
    controller = build_controller(actions=[])
    now = run_clock.now()
    run = ResearchRun(
        run_id="run_downtime", request=research_request(), status=RunStatus.RUNNING,
        created_at=now, updated_at=now,
    )
    controller.scheduler.store.save(run)
    run_clock.advance(61)
    assert run.remaining_timeout_seconds(run_clock.now()) == 0
    failed = controller.run_until_stable(run.run_id)
    assert failed.status == RunStatus.FAILED
    assert failed.terminal_error.code == ErrorCode.TIMEOUT
    assert failed.user_wait_seconds == 0


@pytest.mark.parametrize("invalid", [-1, float("inf"), float("nan")])
def test_run_rejects_invalid_user_wait(invalid):
    from pydantic import ValidationError

    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        ResearchRun(
            run_id="run_wait_invalid", request=research_request(), status=RunStatus.RUNNING,
            created_at=now, updated_at=now, user_wait_seconds=invalid,
        )



def child_work(run):
    return [work for work in run.work_requests if work.owner_task_id is not None]


def execution_tasks(run):
    return [task for task in run.workflow.tasks if task.id != "task_root"] if run.workflow else []


def root_request(controller, run):
    task = run.task("task_root")
    return controller.scheduler._module_request(run, task, task.attempts[-1].number,
        parent_session_id=task.attempts[-1].session.id)


def research_request() -> ResearchRequest:
    return ResearchRequest(goal='Evaluate the method', budget=RunBudget(max_llm_calls=50, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=5, max_attempts_per_task=2))


def registry() -> WorkflowAgentRegistry:
    return default_execution_registry()


def proposal(work_request_id: str) -> WorkflowProposal:
    return WorkflowProposal(
        work_request_id=work_request_id,
        tasks=[
            TaskProposal(execution_ref="experiment@1",
                id="task_experiment",
                work_request_id=work_request_id,
                workflow_agent_kind="executor",
                instruction="Run the experiment",

            )
        ],
    )


def completed_result() -> AgentResult:
    return AgentResult(status=ModuleStatus.COMPLETED, report="done", session=SessionRef(id="session_task_child", module=AgentOwner.EXECUTOR, state_uri="memory://session_task_child", status=SessionStatus.COMPLETED, created_at=NOW, updated_at=NOW))


def build_controller(
    *,
    actions: list[dict],
    store=None,
    dataset_ref_source=None,
) -> ResearchController:
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([completed_result()]),
            )
        }),
        store=store or InMemoryRunStore(),
    )
    compiler = DeterministicWorkflowCompiler(proposal("work_1"), patch=None)
    scientific = ScientificAgent(
        ParentQuestionClient(tool_turns(actions)),
        store=InMemorySessionStore(),
    )
    return ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=compiler,
        scheduler=scheduler,
        registry=registry(),
        dataset_ref_source=dataset_ref_source,
    )


def request_work_action() -> dict:
    return {
        "tool": "request_work",
        "arguments": {
            "assessment": {"statement": "need evidence"},
            "work_request": {'objective': 'Run the experiment', 'tasks': [{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Run the experiment'}]},
        },
    }


def finish_action(verdict=ScientificVerdict.INCONCLUSIVE, limitations=()) -> dict:
    opinion = {"verdict": verdict.value, "statement": "done"}
    if limitations:
        opinion["limitations"] = list(limitations)
    return {"tool": "finish", "arguments": {"report": "Scientific conclusion", "artifacts": [{'kind': "scientific_opinion", 'path': "scientific_opinion.json", 'media_type': "application/json", 'summary': "Scientific conclusion", 'content': opinion}]}}


def test_direct_conclusion_without_work() -> None:
    controller = build_controller(actions=[finish_action(ScientificVerdict.INCONCLUSIVE)])

    run = controller.create_run("run_direct", research_request())

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert run.final_opinion is not None
    assert child_work(run) == []
    assert run.final_report_artifact_id in run.artifacts
    report = run.artifacts[run.final_report_artifact_id]
    assert report.kind == "final_report"
    assert report.producer == AgentOwner.ORCHESTRATOR
    assert report.metadata == {"source_type": "final_report"}


def test_run_creation_freezes_grants_before_any_work_request(tmp_path):
    from sciagent_contracts import WorkspaceAccess, WorkspaceSpec
    controller = build_controller(actions=[{"tool": "ask_user", "arguments": {
        "assessment": {"statement": "Need readiness"}, "text": "Ready?", "requested_fields": ["ready"]}}, request_work_action(), finish_action()])
    original = WorkspaceSpec(workspace_id="ws_main", source_kind="local", location=str(tmp_path / "original"),
        access=WorkspaceAccess(read_paths=["src"], denied_paths=["src/private"]))
    controller.scheduler.workspace_specs = {"ws_main": original}
    request = research_request().model_copy(update={"permissions": RunPermissions()})
    created = controller.create_run("run_frozen", request)
    assert not child_work(created)
    assert created.workspaces["ws_main"].source.access == original.access
    request.permissions.execute_commands = True
    original.access.read_paths[:] = ["."]
    original.access.write_paths[:] = ["."]
    original.access.denied_paths.clear()
    controller.scheduler.workspace_specs = {"ws_main": WorkspaceSpec(workspace_id="ws_main", source_kind="local",
        location=str(tmp_path / "other"), access=WorkspaceAccess(read_paths=["."], write_paths=["."]))}
    completed = controller.answer_question(created.run_id, UserAnswer(question_id=created.pending_question.id,
        values={"ready": "yes"}, answered_at=datetime.now(UTC)))
    assert completed.status == RunStatus.COMPLETED, completed.terminal_error
    received = controller.scheduler.bindings["executor"].port.requests[0]
    assert received.workspace.root == str(tmp_path / "original")
    assert received.workspace.access.read_paths == ["src"]
    assert received.workspace.access.write_paths == []
    assert received.workspace.access.denied_paths == ["src/private"]
    assert not received.permissions.execute_commands


def test_first_scientific_turn_recovers_from_bound_session_checkpoint() -> None:
    """A crash after runtime checkpointing cannot orphan the first session."""
    store = InMemoryRunStore()
    scientific = ScientificAgent(
        ParentQuestionClient(tool_turns([finish_action(ScientificVerdict.INCONCLUSIVE)])),
        store=InMemorySessionStore(),
    )

    class ProcessCrash(BaseException):
        pass

    class _CrashAfterScientificCheckpoint:
        calls = 0

        def read_terminal_result(self, **kwargs):
            return scientific.read_terminal_result(**kwargs)

        def invoke(self, request):
            result = scientific.invoke(request)
            self.calls += 1
            if self.calls == 1:
                raise ProcessCrash("simulated controller crash")
            return result

    scheduler = WorkflowScheduler(bindings=executor_bindings({}), store=store)
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=_CrashAfterScientificCheckpoint(),
        compiler=DeterministicWorkflowCompiler(proposal("work_1"), patch=None),
        scheduler=scheduler,
        registry=registry(),
    )

    with pytest.raises(ProcessCrash, match="simulated controller crash"):
        controller.create_run("run_first_turn_restart", research_request())

    persisted = store.load("run_first_turn_restart")
    assert persisted.scientific_session is not None
    assert persisted.scientific_session.id == task_session_id(persisted.run_id, "task_root", 1)
    assert persisted.scientific_session.status == SessionStatus.ACTIVE

    recovered = controller.run_until_stable("run_first_turn_restart")
    assert recovered.status == RunStatus.COMPLETED
    assert recovered.scientific_session is not None
    assert recovered.scientific_session.id == persisted.scientific_session.id


def test_completion_gate_violations_are_persisted() -> None:
    class _InvalidCompletionPort:
        def invoke(self, request):
            return AgentResult(status="completed", report="Conclusion without its session or evidence")

    scheduler = WorkflowScheduler(bindings=executor_bindings({}), store=InMemoryRunStore())
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=_InvalidCompletionPort(),
        compiler=DeterministicWorkflowCompiler(proposal("work_1"), patch=None),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_invalid_gate", research_request())

    assert run.status == RunStatus.FAILED
    assert run.terminal_error.code == ErrorCode.CONTRACT_ERROR
    assert "bound session" in run.terminal_error.message
    assert scheduler.store.load(run.run_id).terminal_error == run.terminal_error


def test_single_work_cycle_completes() -> None:
    controller = build_controller(
        actions=[request_work_action(), finish_action()]
    )

    run = controller.create_run("run_cycle", research_request())

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert run.final_opinion is not None
    assert len(child_work(run)) == 1
    assert child_work(run)[0].status.value == "consumed"
    # The WorkOutcome must carry the real execution summary, not the task goal.
    report_id = child_work(run)[0].outcome.tasks[0].attempts[0].report_artifact_id
    from sciagent_components import RegisteredArtifactReader
    assert RegisteredArtifactReader(list(run.artifacts.values()), run_id=run.run_id).read_text(report_id)["content"] == "done"


def _cycle_compiler():
    """A compiler that emits a proposal for the first request and a patch (adding
    a fresh task) for every subsequent request on the same workflow."""

    class _CycleCompiler:
        def compile(self, request, *, current, registry, limits, workspaces=None):
            if current is None:
                return CompilationResult(proposal(request.id))
            if not any(task.workflow_agent_kind != WorkflowAgentKind.SCIENTIFIC for task in current.tasks):
                return CompilationResult(WorkflowPatch(work_request_id=request.id, based_on_revision=current.revision, add_tasks=proposal(request.id).tasks))
            # A new request on an existing workflow becomes a patch adding one task.
            return CompilationResult(
                WorkflowPatch(
                    work_request_id=request.id,
                    based_on_revision=current.revision,
                    add_tasks=[
                        TaskProposal(execution_ref="experiment@1",
                            id=f"task_{request.id}",
                            work_request_id=request.id,
                            workflow_agent_kind="executor",
                            instruction=f"Run for {request.id}",

                        )
                    ],
                )
            )

    return _CycleCompiler()


def test_multiple_serial_work_cycles(tmp_path) -> None:
    actions = [
        request_work_action(),
        request_work_action(),
        finish_action(),
    ]
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([completed_result(), completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    scientific = ScientificAgent(ParentQuestionClient(tool_turns(actions)), store=InMemorySessionStore())
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=_cycle_compiler(),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_multi", research_request())

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(child_work(run)) == 2
    assert [item.status.value for item in child_work(run)] == ["consumed", "consumed"]


def test_task_failure_then_request_alternative_work(tmp_path) -> None:
    failed = AgentResult(
        status=ModuleStatus.FAILED,
        report="crashed",
        error=ModuleError(
            code=ErrorCode.TOOL_FAILED,
            message="crashed",
            retryable=False,
        ),
    )
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([failed, completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    actions = [
        request_work_action(),
        request_work_action(),
        finish_action(limitations=["a task failed"]),
    ]
    scientific = ScientificAgent(ParentQuestionClient(tool_turns(actions)), store=InMemorySessionStore())
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=_cycle_compiler(),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_failure", research_request())

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(child_work(run)) == 2


def test_paused_question_then_answer_resumes() -> None:
    ask_action = {
        "tool": "ask_user",
        "arguments": {"assessment": {"statement": "Need a user decision"}, 'text': 'Which dataset?', 'requested_fields': ['dataset']},
    }
    controller = build_controller(
        actions=[ask_action, finish_action()],
    )

    paused = controller.create_run("run_paused", research_request())
    assert paused.status == RunStatus.PAUSED
    assert paused.pending_question is not None

    answer = UserAnswer(
        question_id=paused.pending_question.id,
        values={"dataset": "demo"},
        answered_at=NOW,
    )
    run = controller.answer_question("run_paused", answer)

    assert run.status == RunStatus.COMPLETED, run.terminal_error


def test_newly_registered_dataset_is_added_when_answer_resumes_run() -> None:
    class _DatasetSource:
        refs = []

        def references(self):
            return list(self.refs)

    source = _DatasetSource()
    ask_action = {
        "tool": "ask_user",
        "arguments": {"assessment": {"statement": "Need a user decision"}, 'text': 'Provision cifar10 and confirm when it is ready.', 'requested_fields': ['dataset_ready']},
    }
    controller = build_controller(
        actions=[ask_action, finish_action()],
        dataset_ref_source=source,
    )
    request = research_request()
    paused = controller.create_run("run_dataset_refresh", request)
    assert paused.dataset_refs == []
    assert paused.request == request

    source.refs = [DatasetRef(dataset_id="cifar10", relative_path="cifar10")]
    completed = controller.answer_question(
        paused.run_id,
        UserAnswer(
            question_id=paused.pending_question.id,
            values={"dataset_ready": "yes"},
            answered_at=NOW,
        ),
    )

    assert completed.status == RunStatus.COMPLETED
    assert completed.dataset_refs == source.refs
    assert completed.request == request




def test_catalog_refresh_survives_controller_restart_and_verbal_confirmation(tmp_path):
    from sciagent_components import (
        DatasetCatalog,
        ResourceLayout,
    )

    layout = ResourceLayout(resource_root=tmp_path / "resources")
    layout.dataset_root.mkdir(parents=True)
    ask = {
        "tool": "ask_user",
        "arguments": {"assessment": {"statement": "Need a user decision"}, 'text': 'Place and register demo, then confirm', 'requested_fields': ['dataset_ready']},
    }

    def rebuild(actions):
        controller = _build_recoverable_controller(
            JsonRunStore(tmp_path / "runs"),
            JsonSessionStore(tmp_path / "sessions"), actions,
        )
        controller.dataset_ref_source = DatasetCatalog(layout.dataset_root)
        controller.scientific_port.resource_layout = layout
        return controller

    original = research_request()
    first = rebuild([ask])
    paused = first.create_run("run_catalog", original)
    session_id = paused.scientific_session.id
    (layout.dataset_root / "catalog.json").write_text('{"demo": "demo"}', encoding="utf-8")

    second = rebuild([ask])
    still_missing = second.answer_question(paused.run_id, UserAnswer(
        question_id=paused.pending_question.id,
        values={"dataset_ready": "yes"}, answered_at=datetime.now(UTC),
    ))
    text = second.scientific_port.llm_client.contexts[0].text
    assert '"available_dataset_ids": []' in text
    assert '"unavailable_dataset_ids": ["demo"]' in text
    assert still_missing.status == RunStatus.PAUSED
    assert still_missing.pending_question.id != paused.pending_question.id

    (layout.dataset_root / "demo").mkdir()
    third = rebuild([finish_action()])
    completed = third.answer_question(paused.run_id, UserAnswer(
        question_id=still_missing.pending_question.id,
        values={"dataset_ready": "yes"}, answered_at=datetime.now(UTC),
    ))
    assert completed.status == RunStatus.COMPLETED
    assert completed.scientific_session.id == session_id
    assert completed.request == original
    assert completed.dataset_refs == DatasetCatalog(layout.dataset_root).references()
    assert '"available_dataset_ids": ["demo"]' in third.scientific_port.llm_client.contexts[0].text


def test_dataset_binding_cannot_be_remapped_during_run() -> None:
    class _DatasetSource:
        def references(self):
            return [DatasetRef(dataset_id="cifar10", relative_path="other")]

    controller = build_controller(
        actions=[finish_action()],
        dataset_ref_source=_DatasetSource(),
    )
    now = datetime.now(UTC)
    controller.scheduler.store.save(ResearchRun(
        run_id="run_dataset_remap", request=research_request(),
        dataset_refs=[DatasetRef(dataset_id="cifar10", relative_path="cifar10")],
        status=RunStatus.RUNNING, created_at=now, updated_at=now,
    ))
    with pytest.raises(ValueError, match="remapped during Run"):
        controller.run_until_stable("run_dataset_remap")


def _task_question_result() -> AgentResult:
    return AgentResult(
        status=ModuleStatus.NEEDS_USER_INPUT,
        report="Which dataset?",
        artifacts=[], control=question_signal("session_task_child", 'Which dataset?', ['dataset']),
        session=SessionRef(
            id="session_task_child",
            module=AgentOwner.EXECUTOR,
            state_uri="memory://session_task_child",
            status=SessionStatus.PAUSED,
            created_at=NOW,
            updated_at=NOW,
        ),
    )


def test_task_question_resumes_same_attempt_via_controller() -> None:
    """A task-level question pauses the run; the answer resumes the same Attempt.

    This is the cross-layer fix for P1-1: the controller (not a separate
    scheduler answer path) routes the answer back to the paused task, which
    resumes on the same Attempt number instead of starting a new one.
    """
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([_task_question_result(), completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    scientific = ScientificAgent(
        ParentQuestionClient(tool_turns([request_work_action(), finish_action()])),
        store=InMemorySessionStore(),
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=DeterministicWorkflowCompiler(proposal("work_1"), patch=None),
        scheduler=scheduler,
        registry=registry(),
    )

    paused = controller.create_run("run_task_question", research_request())

    assert paused.status == RunStatus.PAUSED
    assert paused.pending_question is not None
    assert paused.pending_question.task_id == "task_root"
    assert next(iter(paused.child_questions.values())).task_id == "task_experiment"
    task = execution_tasks(paused)[0]
    assert task.status.value == "needs_user_input"
    assert task.attempts[0].status.value == "needs_user_input"
    assert task.attempts[0].finished_at is None

    answer = UserAnswer(
        question_id=paused.pending_question.id,
        values={"dataset": "demo"},
        answered_at=NOW,
    )
    run = controller.answer_question("run_task_question", answer)

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    task = execution_tasks(run)[0]
    # The same Attempt resumed, not a new one.
    assert [attempt.number for attempt in task.attempts] == [1]
    assert task.attempts[0].status.value == "completed"
    assert child_work(run)[0].status.value == "consumed"
    resumed_request = scheduler.bindings["executor"].port.requests[-1]
    from sciagent_orchestrator.handoffs import read_json
    answers = resumed_request.user_answers
    assert len(answers) == 1 and answers[0].source == "parent"
    assert answers[0].question_text == paused.pending_question.text
    assert answers[0].question_id != paused.pending_question.id
    assert answers[0].values == answer.values
    assert len(run.answers) == 2 and run.answers[0].source == "user"
    final_context = scientific.llm_client.contexts[-1]
    task_answer_ids = [ref.id for ref in resumed_request.input_artifacts if ref.kind == "answer"]
    assert all(f"material_{key}" not in final_context.included_sections for key in task_answer_ids)
    assert all(key in final_context.text for key in task_answer_ids)  # Discoverable, not replayed as a resume answer.


def test_task_question_resume_does_not_consume_attempt_budget() -> None:
    """max_attempts_per_task=1 must still allow a pause/resume round-trip."""
    request = ResearchRequest(goal='Evaluate', budget=RunBudget(max_llm_calls=50, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=5, max_attempts_per_task=1))
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([_task_question_result(), completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    scientific = ScientificAgent(
        ParentQuestionClient(tool_turns([request_work_action(), finish_action()])),
        store=InMemorySessionStore(),
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=DeterministicWorkflowCompiler(proposal("work_1"), patch=None),
        scheduler=scheduler,
        registry=registry(),
    )

    paused = controller.create_run("run_task_question_1", request)
    assert paused.status == RunStatus.PAUSED

    answer = UserAnswer(
        question_id=paused.pending_question.id,
        values={"dataset": "demo"},
        answered_at=NOW,
    )
    run = controller.answer_question("run_task_question_1", answer)

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert [attempt.number for attempt in execution_tasks(run)[0].attempts] == [1]


def test_json_store_recovers_run_boundary(tmp_path) -> None:
    store = JsonRunStore(tmp_path / "state")
    controller = build_controller(
        actions=[request_work_action(), finish_action()],
        store=store,
    )
    controller.create_run("run_recover", research_request())

    recovered = JsonRunStore(tmp_path / "state").load("run_recover")
    assert recovered.status == RunStatus.COMPLETED
    assert recovered.final_opinion is not None


@pytest.mark.parametrize("rejected_first", [False, True])
def test_no_match_returns_whole_work_request_to_scientific_without_executing(tmp_path, rejected_first):
    work = request_work_action()
    work["arguments"]["work_request"]["tasks"] = [
        {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "measure", "instruction": "Measure the requested outcome"},
        {"selection":{"kind":"no_match","reason":"No available preset supplies the required external instrument"}, "key": "mixed", "instruction": "Implement a method and run its scientific evaluation"},
    ]
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({"experiment": ModuleBinding(
            owner=AgentOwner.EXECUTOR, port=ScopedScriptedModulePort([completed_result()]))}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data",
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(), compiler=None, scheduler=scheduler, registry=registry(),
        scientific_port=ScientificAgent(ParentQuestionClient(tool_turns([work, {
            "tool": "ask_user", "arguments": {
                "assessment": {"statement": "The mixed task must be planned again"},
                "text": "Which part should be delegated first?", "requested_fields": ["first_part"],
            },
        }])), store=InMemorySessionStore()),
    )
    if rejected_first:
        work["arguments"]["work_request"]["tasks"].reverse()
    controller.compiler = FixedWorkflowCompiler()
    requests = []
    invoke = controller.scientific_port.invoke

    def capture(request):
        requests.append(request)
        return invoke(request)

    controller.scientific_port.invoke = capture
    run = controller.create_run("run_no_match", research_request())

    assert run.status == RunStatus.PAUSED, run.terminal_error
    assert execution_tasks(run) == []
    assert run.pending_compilation is None
    assert run.llm_calls_used == 2  # Scientific submission and reconsideration; fixed compilation is free.
    for kind, binding in controller.scheduler.bindings.items():
        if kind != WorkflowAgentKind.SCIENTIFIC:
            assert binding.port.requests == []
    rejected, = child_work(run)
    assert rejected.status == WorkRequestStatus.FAILED
    assert rejected.outcome.tasks == []
    assert "mixed" in rejected.error.message
    assert "No available preset supplies the required external instrument" in rejected.error.message
    feedback, = requests[-1].work_results
    assert feedback.work_request_id == rejected.id
    assert feedback.tasks == []
    assert feedback.error == rejected.error


def test_compilation_failure_fails_run() -> None:
    class _FailingCompiler:
        def compile(self, request, *, current, registry, limits, workspaces=None):
            raise ValueError("compiler exploded")

    scheduler = WorkflowScheduler(
        bindings=executor_bindings({}),
        store=InMemoryRunStore(),
    )
    scientific = ScientificAgent(
        ParentQuestionClient(tool_turns([request_work_action()])),
        store=InMemorySessionStore(),
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=_FailingCompiler(),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_compile_fail", research_request())

    assert run.status == RunStatus.FAILED
    assert child_work(run)[0].status.value == "failed"


@pytest.mark.parametrize("accepted_status", [WorkRequestStatus.COMPILING, WorkRequestStatus.EXECUTING])
def test_compiling_restart_resumes_an_already_accepted_workflow(tmp_path, accepted_status) -> None:
    class ProcessCrash(BaseException):
        pass
    class CrashCompiler:
        def compile(self, *args, **kwargs):
            raise ProcessCrash()
    class MustNotCompile:
        def compile(self, *args, **kwargs):
            raise AssertionError("accepted child tasks must not be recompiled")
    controller = build_controller(actions=[request_work_action(), finish_action()])
    controller.compiler = CrashCompiler()
    with pytest.raises(ProcessCrash):
        controller.create_run("run_accept_recovery", research_request())
    saved = controller.scheduler.store.load("run_accept_recovery")
    work = child_work(saved)[0]
    assert work.status == WorkRequestStatus.COMPILING
    controller.scheduler.apply_patch(saved.run_id, WorkflowPatch(work_request_id=work.id,
        based_on_revision=saved.workflow.revision, add_tasks=proposal(work.id).tasks))
    accepted = controller.scheduler.store.load(saved.run_id)
    work = child_work(accepted)[0]
    work.status = accepted_status
    if accepted_status == WorkRequestStatus.EXECUTING:
        work.workflow_revision = accepted.workflow.revision
    controller.scheduler.store.save(accepted)
    controller.compiler = MustNotCompile()
    run = controller.run_until_stable(saved.run_id)
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert child_work(run)[0].status == WorkRequestStatus.CONSUMED
    assert len(execution_tasks(run)) == 1


@pytest.mark.parametrize("previous_failed", [False, True])
def test_zero_task_slots_fail_before_compiler_and_preserve_history(tmp_path, previous_failed) -> None:
    class CountingCompiler:
        calls = 0

        def compile(self, request, **kwargs):
            self.calls += 1
            current = kwargs["current"]
            return CompilationResult(WorkflowPatch(work_request_id=request.id, based_on_revision=current.revision, add_tasks=proposal(request.id).tasks))

    first_result = (
        AgentResult(
            status=ModuleStatus.FAILED, report="Execution failed",
            error=ModuleError(code=ErrorCode.TOOL_FAILED, message="Original failure", retryable=False),
        )
        if previous_failed else completed_result()
    )
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({"experiment": ModuleBinding(
            owner=AgentOwner.EXECUTOR, port=ScopedScriptedModulePort([first_result]),
        )}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
    )
    compiler = CountingCompiler()
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=ScientificAgent(
            ParentQuestionClient(tool_turns([request_work_action(), request_work_action(),
                                        finish_action(limitations=["The subsequent work request could not run because its task slots were exhausted"])])),
            store=InMemorySessionStore(),
        ),
        compiler=compiler, scheduler=scheduler, registry=registry(),
    )
    request = research_request()
    request.execution_limits.max_tasks = 2
    run = controller.create_run("run_task_slots", request)

    assert run.status == RunStatus.COMPLETED
    assert run.terminal_error is None
    assert child_work(run)[0].status == WorkRequestStatus.CONSUMED
    assert child_work(run)[-1].status == WorkRequestStatus.FAILED
    assert child_work(run)[-1].error.code == ErrorCode.BUDGET_EXHAUSTED
    assert compiler.calls == 1
    assert run.llm_calls_used == 3  # two requests plus the post-failure Scientific conclusion
    assert len(execution_tasks(run)) == 1
    assert execution_tasks(run)[0].status.value == ("failed" if previous_failed else "completed")
    history = run.workflow.model_dump_json()

    assert child_work(run)[-1].outcome.tasks == []
    assert child_work(run)[-1].outcome.error.message == "No remaining task slots"
    assert controller.run_until_stable(run.run_id) == run


def test_second_work_outcome_contains_only_second_round_tasks() -> None:
    actions = [
        request_work_action(),
        request_work_action(),
        finish_action(),
    ]
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([completed_result(), completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    scientific = ScientificAgent(ParentQuestionClient(tool_turns(actions)), store=InMemorySessionStore())
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=_cycle_compiler(),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_outcome_isolate", research_request())

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    first = child_work(run)[0]
    second = child_work(run)[1]
    assert first.outcome.work_request_id == first.id
    assert [t.task_id for t in first.outcome.tasks] == ["task_experiment"]
    assert second.outcome.work_request_id == second.id
    assert [t.task_id for t in second.outcome.tasks] == [f"task_{second.id}"]


def test_failed_task_appears_in_unresolved_then_is_reported() -> None:
    failed = AgentResult(
        status=ModuleStatus.FAILED,
        report="crashed",
        error=ModuleError(
            code=ErrorCode.TOOL_FAILED, message="crashed", retryable=False
        ),
    )
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([failed, completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    actions = [
        request_work_action(),
        request_work_action(),
        finish_action(limitations=["a task failed"]),
    ]
    scientific = ScientificAgent(ParentQuestionClient(tool_turns(actions)), store=InMemorySessionStore())
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=_cycle_compiler(),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_unresolved", research_request())

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    first_outcome = child_work(run)[0].outcome
    assert [t.task_id for t in first_outcome.tasks] == ["task_experiment"]
    assert first_outcome.tasks[0].status == "failed"


def test_untrusted_reading_log_does_not_control_run_completion() -> None:
    class _ForgingPort:
        """A ScientificPort that reports an observed id that was never registered."""

        def __init__(self):
            from sciagent_scientific import ScientificAgent

            self.agent = ScientificAgent(
                ParentQuestionClient(tool_turns([request_work_action(), finish_action()])),
                store=InMemorySessionStore(),
            )

        def invoke(self, request):
            result = self.agent.invoke(request)
            # Forge an extra observed id into the result.
            for artifact in result.artifacts:
                if artifact.kind == "observation_trace":
                    artifact.content = json.dumps({"observed_artifact_ids": ["artifact_fake"]})
            return result

    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=_ForgingPort(),
        compiler=DeterministicWorkflowCompiler(proposal("work_1"), patch=None),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_forged", research_request())

    assert run.status == RunStatus.COMPLETED
    assert "artifact_fake" not in run.artifacts
    assert run.final_opinion is not None
    assert "artifact_fake" not in run.final_opinion.evidence_artifact_ids


def test_run_total_llm_budget_exhaustion() -> None:
    # A tiny run budget so the accumulated Scientific turns exceed it.
    tiny = ResearchRequest(goal='Evaluate', budget=RunBudget(max_llm_calls=1, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=5, max_attempts_per_task=2))
    actions = [request_work_action(), request_work_action()]
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([completed_result(), completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    scientific = ScientificAgent(ParentQuestionClient(tool_turns(actions)), store=InMemorySessionStore())
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=_cycle_compiler(),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_budget", tiny)

    assert run.status == RunStatus.FAILED


def test_answer_then_request_work_then_outcome_completes() -> None:
    ask_action = {
        "tool": "ask_user",
        "arguments": {"assessment": {"statement": "Need a user decision"}, 'text': 'Which dataset?', 'requested_fields': ['dataset']},
    }
    actions = [ask_action, request_work_action(), finish_action()]
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    scientific = ScientificAgent(ParentQuestionClient(tool_turns(actions)), store=InMemorySessionStore())
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=DeterministicWorkflowCompiler(proposal("work_1"), patch=None),
        scheduler=scheduler,
        registry=registry(),
    )

    paused = controller.create_run("run_answer_work", research_request())
    assert paused.status == RunStatus.PAUSED

    answer = UserAnswer(
        question_id=paused.pending_question.id,
        values={"dataset": "demo"},
        answered_at=NOW,
    )
    run = controller.answer_question("run_answer_work", answer)

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert run.final_opinion is not None


def _build_recoverable_controller(
    run_store, session_store, actions
) -> ResearchController:
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([completed_result()]),
            )
        }),
        store=run_store,
    )
    scientific = ScientificAgent(ParentQuestionClient(tool_turns(actions)), store=session_store)
    return ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=DeterministicWorkflowCompiler(proposal("work_1"), patch=None),
        scheduler=scheduler,
        registry=registry(),
    )


def test_real_restart_recovers_paused_scientific_session(tmp_path) -> None:
    run_store = JsonRunStore(tmp_path / "runs")
    session_store = JsonSessionStore(tmp_path / "sessions")
    ask_action = {
        "tool": "ask_user",
        "arguments": {"assessment": {"statement": "Need a user decision"}, 'text': 'Which dataset?', 'requested_fields': ['dataset']},
    }
    controller = _build_recoverable_controller(
        run_store, session_store, [ask_action, finish_action()]
    )
    paused = controller.create_run("run_restart", research_request())
    assert paused.status == RunStatus.PAUSED

    # Rebuild everything from disk: a fresh controller and a fresh Scientific
    # Agent that share the same persistent stores.
    rebuilt = _build_recoverable_controller(
        JsonRunStore(tmp_path / "runs"),
        JsonSessionStore(tmp_path / "sessions"),
        [finish_action()],
    )
    answer = UserAnswer(
        question_id=paused.pending_question.id,
        values={"dataset": "demo"},
        answered_at=NOW,
    )
    run = rebuilt.answer_question("run_restart", answer)

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert run.final_opinion is not None
    restored = JsonRunStore(tmp_path / "runs").load(run.run_id)
    assert restored.answers[0].question_text == ask_action["arguments"]["text"]
    assert restored.answers[0].values == answer.values
    assert restored.scientific_session.id == paused.scientific_session.id
    context = rebuilt.scientific_port.llm_client.contexts[0]
    answer_ref = restored.answers[0]
    assert _context_answer(context, answer_ref) == _answer_projection(restored.answers[0])


@pytest.mark.parametrize("comment", ["", " 业务反馈必须放在字段答案中\n "])
def test_business_operation_comment_is_rejected_before_persisting_or_resuming(tmp_path, comment):
    from sciagent_orchestrator import OrchestrationError

    store = JsonRunStore(tmp_path / "runs")
    sessions = JsonSessionStore(tmp_path / "sessions")
    ask = {"tool": "ask_user", "arguments": {
        "assessment": {"statement": "Need a user decision"},
        "text": "Which dataset?", "requested_fields": ["dataset"],
    }}
    controller = _build_recoverable_controller(store, sessions, [ask])
    run = controller.create_run("run_business_comment", research_request())
    assert run.status == "paused" and run.pending_question.action is None
    files = [*store.root.glob("*.json"), *(tmp_path / "sessions").glob("*.json")]
    before = {path: path.read_bytes() for path in files}
    rebuilt = _build_recoverable_controller(store, sessions, [finish_action()])
    with pytest.raises(OrchestrationError, match="answer business questions in the requested fields"):
        rebuilt.answer_question(run.run_id, UserAnswer(
            question_id=run.pending_question.id, values={"dataset": "new user choice"},
            answered_at=NOW, comment=comment,
        ))
    assert {path: path.read_bytes() for path in files} == before
    assert rebuilt.scientific_port.llm_client.contexts == []


def test_restarted_generic_answers_stay_paired_with_their_own_questions(tmp_path):
    from sciagent_orchestrator import OrchestrationError

    def ask(text):
        return {"tool": "ask_user", "arguments": {"assessment": {"statement": "Need a user decision"}, 'text': text, 'requested_fields': ['answer']}}

    def rebuild(actions):
        return _build_recoverable_controller(
            JsonRunStore(tmp_path / "runs"), JsonSessionStore(tmp_path / "sessions"), actions,
        )

    first_text = "第一个选择 helper_a.py，第二个选择 helper_b.py。你选哪个？"
    second_text = "第一个模式是 add，第二个模式是 mul。你选哪个？"
    first = rebuild([ask(first_text)])
    paused = first.create_run("run_paired_answers", research_request())
    first_session = JsonSessionStore(tmp_path / "sessions").load(paused.scientific_session.id)
    # Even an in-process caller passing a subclass cannot supply the authoritative question.
    forged = RecordedAnswer(
        question_id=paused.pending_question.id, question_text="Forged question: choose helper_a.py",
        values={"answer": "第二个"}, answered_at=NOW, requested_fields=["answer"],
        run_id=paused.run_id, session_id=paused.scientific_session.id,
    )
    second = rebuild([ask(second_text)])
    again = second.answer_question(paused.run_id, forged)
    assert again.status == RunStatus.PAUSED
    assert again.answers[0].question_text == first_text
    assert again.pending_question.text == second_text
    assert again.pending_question.id != forged.question_id
    first_context = second.scientific_port.llm_client.contexts[0]
    first_ref = again.answers[0]
    assert _context_answer(first_context, first_ref) == _answer_projection(again.answers[0])
    assert "Forged question" not in first_context.text

    before = JsonRunStore(tmp_path / "runs").load(paused.run_id).model_dump_json()
    with pytest.raises(OrchestrationError, match="does not match"):
        second.answer_question(paused.run_id, forged)
    assert JsonRunStore(tmp_path / "runs").load(paused.run_id).model_dump_json() == before

    third = rebuild([finish_action()])
    completed = third.answer_question(paused.run_id, UserAnswer(
        question_id=again.pending_question.id, values={"answer": "第二个"}, answered_at=NOW,
    ))
    assert completed.status == RunStatus.COMPLETED
    restored = JsonRunStore(tmp_path / "runs").load(paused.run_id)
    assert [answer.question_text for answer in restored.answers] == [first_text, second_text]
    assert [answer.values for answer in restored.answers] == [{"answer": "第二个"}] * 2
    last_context = third.scientific_port.llm_client.contexts[0]
    last_ref = restored.answers[1]
    assert _context_answer(last_context, last_ref) == _answer_projection(restored.answers[1])
    assert _context_answer(last_context, first_ref) == _answer_projection(restored.answers[0])
    # Both verified answers remain task dialogue without separate material sections.
    assert f"material_{first_ref.question_id}" not in last_context.included_sections
    assert restored.delivered_answer_ids == [answer.question_id for answer in restored.answers]
    final_session = JsonSessionStore(tmp_path / "sessions").load(restored.scientific_session.id)
    assert final_session.session_id == first_session.session_id
    assert final_session.created_at == first_session.created_at
    assert len(final_session.events) > len(first_session.events)


def test_paired_answers_keep_task_and_scientific_scopes():
    controller = build_controller(actions=[request_work_action(), finish_action()])
    completed = controller.create_run("run_answer_scopes", research_request())
    task = execution_tasks(completed)[0]
    answers = [RecordedAnswer(question_id=f"question_{i}", question_text=f"Question {i}", requested_fields=["answer"],
                              values={"answer":"yes"}, answered_at=NOW, run_id=completed.run_id,
                              session_id=completed.scientific_session.id if i == 0 else "session_task_child",
                              task_id="task_root" if i == 0 else task.id, attempt_number=1 if i == 0 else i)
               for i in range(3)]
    completed.answers = answers
    request = controller.scheduler._module_request(completed, task, 1, parent_session_id="session_task_child")
    assert request.user_answers == request.resume_answers == [answers[1]]
    scientific = root_request(controller, completed)
    assert scientific.user_answers == [answers[0]]
    assert not any(ref.kind == "answer" for ref in scientific.input_artifacts)
    completed.delivered_answer_ids = [answers[0].question_id]
    assert root_request(controller, completed).resume_answers == []


def test_budget_overrun_does_not_complete(tmp_path) -> None:
    """A final turn that pushes llm_calls past the budget must not complete."""
    tiny = ResearchRequest(goal='Evaluate', budget=RunBudget(max_llm_calls=1, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=5, max_attempts_per_task=2))
    # First request_work consumes 1 LLM call; a second turn would overrun.
    actions = [request_work_action(), finish_action()]
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    scientific = ScientificAgent(ParentQuestionClient(tool_turns(actions)), store=InMemorySessionStore())
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=scientific,
        compiler=DeterministicWorkflowCompiler(proposal("work_1"), patch=None),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_overrun", tiny)

    assert run.status == RunStatus.FAILED
    assert run.llm_calls_used >= tiny.budget.max_llm_calls


def test_compiling_restart_recompiles_without_workflow() -> None:
    """The root graph exists, but a compiling child has no accepted Tasks yet."""
    class ProcessCrash(BaseException):
        pass
    class CrashCompiler:
        def compile(self, *args, **kwargs):
            raise ProcessCrash()
    controller = build_controller(actions=[request_work_action(), finish_action()])
    controller.compiler = CrashCompiler()
    with pytest.raises(ProcessCrash):
        controller.create_run("run_compile_restart", research_request())
    saved = controller.scheduler.store.load("run_compile_restart")
    assert execution_tasks(saved) == []
    assert child_work(saved)[0].status == WorkRequestStatus.COMPILING
    controller.compiler = DeterministicWorkflowCompiler(proposal("work_unused_fixture_identity"))
    run = controller.run_until_stable(saved.run_id)
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(execution_tasks(run)) == 1


def test_compiler_llm_calls_enter_the_run_ledger() -> None:
    from sciagent_runtime.budget import current_budget
    class _CountingCompiler:
        def compile(self, request, *, current, registry, limits, workspaces=None):
            scope = current_budget()
            for index in range(7):
                scope.charge("test-compiler", index)
                scope.usage.complete("test-compiler", index, "succeeded")
            return CompilationResult(WorkflowPatch(work_request_id=request.id, based_on_revision=current.revision, add_tasks=proposal(request.id).tasks), llm_calls=7)

    scheduler = WorkflowScheduler(
        bindings=executor_bindings({
            "experiment": ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=ScopedScriptedModulePort([completed_result()]),
            )
        }),
        store=InMemoryRunStore(),
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=ScientificAgent(
            ParentQuestionClient(tool_turns([request_work_action(), finish_action()])),
            store=InMemorySessionStore(),
        ),
        compiler=_CountingCompiler(),
        scheduler=scheduler,
        registry=registry(),
    )

    run = controller.create_run("run_compiler_calls", research_request())

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    # 7 compiler calls + 2 Scientific calls (request_work + finish).
    assert run.llm_calls_used == 9


@pytest.mark.parametrize("usage_known", [True, False])
def test_failed_compiler_usage_comes_from_reservations(tmp_path, usage_known) -> None:
    from sciagent_runtime.budget import current_budget
    class _FailingCompiler:
        # A stale implementation attribute must never override invocation usage.
        llm_calls = 99
        invocations = 0

        def compile(
            self, request, *, current, registry, limits,
            workspaces=None,
        ):
            self.invocations += 1
            if usage_known:
                for index in range(3):
                    current_budget().charge("failed-compiler", index)
                raise CompilationError("compiler failed", llm_calls=3)
            raise RuntimeError("compiler failed")

    compiler = _FailingCompiler()
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({}),
        store=InMemoryRunStore(),
        artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "runs",
    )
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=ScientificAgent(
            ParentQuestionClient(tool_turns([request_work_action()])),
            store=InMemorySessionStore(),
        ),
        compiler=compiler,
        scheduler=scheduler,
        registry=registry(),
    )
    run = controller.create_run("run_compiler_usage_failure", research_request())

    assert compiler.invocations == 1
    assert run.status == RunStatus.FAILED
    assert child_work(run)[0].status.value == "failed"
    assert run.llm_calls_used == (5 if usage_known else 2)
    assert run.terminal_error is not None
    assert child_work(run)[0].error.details["compiler_usage_known"] is usage_known
    assert child_work(run)[0].outcome.error == child_work(run)[0].error
    assert child_work(run)[0].outcome.tasks == []


def test_required_outputs_are_frozen_in_the_conclusion_requirement_artifact():
    ask = {
        "tool": "ask_user",
        "arguments": {
            "assessment": {"statement": "Need the metric source"},
            "text": "Which source should supply the metrics?", "requested_fields": ["source"],
        },
    }
    controller = build_controller(actions=[ask])
    request = research_request().model_copy(update={"required_artifacts": ["metrics", "analysis"]})
    run = controller.create_run("run_frozen_outputs", request)
    assert run.status == RunStatus.PAUSED
    snapshot = run.conclusion_requirements_ref
    assert read_json(snapshot, ConclusionRequirements).required_artifacts == ["metrics", "analysis"]

    request.required_artifacts[:] = ["replacement"]
    persisted = controller.scheduler.store.load(run.run_id)
    assert persisted.conclusion_requirements_ref == snapshot
    assert read_json(persisted.conclusion_requirements_ref, ConclusionRequirements).required_artifacts == [
        "metrics", "analysis",
    ]
    forwarded = root_request(controller, persisted)
    assert snapshot in forwarded.input_artifacts
    assert forwarded.task_context["conclusion_requirements"] == read_json(snapshot)
    assert "required_artifacts" not in forwarded.task_context


def test_controller_registers_required_scientific_candidate_before_final_acceptance():
    action = finish_action()
    action["arguments"]["artifacts"].append(ArtifactCandidate(
        kind="module_report", path="analysis.md", media_type="text/markdown",
        summary="Requested analysis", content="The analysis", output_name="analysis",
    ).model_dump(mode="json"))
    controller = build_controller(actions=[action])
    request = research_request().model_copy(update={"required_artifacts": ["analysis"]})
    run = controller.create_run("run_scientific_delivery", request)

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    deliveries = [ref for ref in run.artifacts.values() if ref.output_name == "analysis"]
    assert len(deliveries) == 1
    assert deliveries[0].producer == AgentOwner.SCIENTIFIC
    assert deliveries[0].session_id == run.scientific_session.id
    assert run.final_opinion.evidence_artifact_ids == []
    assert run.final_report_artifact_id in run.artifacts
    assert controller.scheduler.artifact_registry.missing_required_artifacts(
        ["analysis"], run_id=run.run_id, artifacts=run.artifacts,
    ) == []


def test_missing_output_feedback_can_execute_work_and_resume_without_citation():
    controller = build_controller(actions=[finish_action(), request_work_action(), finish_action()])
    produced = completed_result().model_copy(update={"artifacts": [
        ArtifactCandidate(
            kind="experiment_result", path="result.json", media_type="application/json",
            summary="Measured output", content='{"accuracy": 0.8}', output_name="metrics",
        ),
    ]})
    controller.scheduler.bindings["executor"] = ModuleBinding(
        owner=AgentOwner.EXECUTOR, port=ScopedScriptedModulePort([produced]),
    )
    request = research_request().model_copy(update={"required_artifacts": ["metrics"]})
    run = controller.create_run("run_required_work", request)

    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(child_work(run)) == 1
    assert child_work(run)[0].status == WorkRequestStatus.CONSUMED
    assert child_work(run)[0].owner_session_id == run.scientific_session.id
    assert any(ref.output_name == "metrics" for ref in run.artifacts.values())
    assert run.final_opinion.evidence_artifact_ids == []
    contexts = controller.scientific_port.llm_client.contexts
    assert len(contexts) == 3
    assert "required_artifact_missing" in json.dumps(controller.scientific_port.store.load(run.scientific_session.id).tool_turns[0].model_dump(mode="json"))
    assert "metrics" in json.dumps(controller.scientific_port.store.load(run.scientific_session.id).tool_turns[0].model_dump(mode="json"))
    persisted = controller.scientific_port.store.load(run.scientific_session.id)
    assert persisted.session_id == run.scientific_session.id
    assert persisted.llm_calls_used == run.llm_calls_used == 3


def test_invalid_work_input_is_corrected_in_same_scientific_session():
    invalid = request_work_action()
    invalid["arguments"]["work_request"]["tasks"][0]["input_artifact_ids"] = ["artifact_typo"]
    controller = build_controller(actions=[
        invalid, request_work_action(), finish_action(),
    ])
    run = controller.create_run("run_correct_work_input", research_request())
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert run.llm_calls_used == 3
    assert len(child_work(run)) == 1
    assert child_work(run)[0].status == WorkRequestStatus.CONSUMED
    client = controller.scientific_port.llm_client
    assert any("Work input artifact is not authorized" in receipt
               for turn in client.histories[1] for receipt in turn.tool_results.values())
    state = controller.scientific_port.store.load(run.scientific_session.id)
    assert state.run_id == run.run_id
    assert state.llm_calls_used == 3


@pytest.mark.parametrize("execute_commands,approval_mode", [(True, "auto_review"), (True, "manual"), (False, "auto_review")])
def test_scientific_sees_role_and_real_run_authority_without_rewriting_business_requirements(execute_commands, approval_mode):
    constraints = ["Preserve the requested method and supplied source data"]
    action = request_work_action()
    controller = build_controller(actions=[action, finish_action()])
    request = research_request().model_copy(update={
        "permissions": RunPermissions(execute_commands=execute_commands, approval_mode=approval_mode),
        "constraints": list(constraints),
    })
    run = controller.create_run("run_scoped_authority", request)
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    context = controller.scientific_port.llm_client.contexts[0]
    task = json.loads(context.task_context.split("\n", 1)[1])
    assert "permissions" not in task
    assert "run_permissions" not in task["task_context"]
    assert request.goal in task["instruction"] and constraints[0] in task["instruction"]
    authority = json.loads(context.runtime_context.split("\n", 1)[1])
    assert authority["source"] == "runtime" and authority["current_agent"] == "scientific"
    assert authority["current_agent_permissions"] == {
        "schema_version": "32.0", "execute_commands": execute_commands,
        "approval_mode": approval_mode, "request_work": True}
    assert authority["run_execution_permissions"] == {
        "execute_commands": execute_commands, "approval_mode": approval_mode}
    assert authority["work_depth"] == 1 and authority["max_work_depth"] == 3
    child = controller.scheduler.bindings["executor"].port.requests[0]
    assert child.permissions.execute_commands is execute_commands
    assert child.permissions.approval_mode == approval_mode
    assert child.instruction == "Run the experiment"
    assert "shared_constraints" not in child.task_context
    assert constraints[0] not in child.instruction
    assert run.task("task_root").instruction == "\n\n".join([request.goal, *constraints])
    assert run.request.constraints == request.constraints == constraints
    assert run.request.permissions == request.permissions
