"""Crash recovery must finish lifecycle handoffs before dispatching more work."""

from datetime import UTC, datetime
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from sciagent_contracts import (
    ActionSnapshot, AgentResult, ControlSignal, ErrorCode, ModuleError, ModuleStatus,
    PendingQuestion, RunStatus, SessionStatus, TaskProposal, TaskStatus,
    UserAnswer, WorkRequestDraft, WorkRequestStatus, WorkflowAgentKind, WorkflowPatch,
)
from sciagent_orchestrator import CompilationResult, JsonRunStore
from sciagent_runtime.cancellation import OperationCancelled

from test_recursive_work import (
    Port, complete, conclude, controller, delegate, research_request, session,
)


class ProcessCrash(BaseException):
    pass


class CrashAfterSave(JsonRunStore):
    """Fail after an actual atomic disk replacement, never only an in-memory edit."""

    def __init__(self, root, crash_when):
        super().__init__(root)
        self.crash_when = crash_when

    def save(self, run):
        super().save(run)
        if self.crash_when is not None and self.crash_when(run):
            self.crash_when = None
            raise ProcessCrash()


class NamedTaskCompiler:
    """Allow separate tasks of the same Agent kind in a nested workflow."""

    def compile(self, work, *, current, **kwargs):
        return CompilationResult(WorkflowPatch(work_request_id=work.id,
            based_on_revision=current.revision, add_tasks=[TaskProposal(execution_ref=(None if item.key.split("_", 1)[0] == "scientific" else item.key.split("_", 1)[0] + "@1"),
                id="task_" + item.key, work_request_id=work.id,
                workflow_agent_kind=("scientific" if item.key.split("_", 1)[0] == "scientific" else "executor"),
                instruction=item.instruction) for item in work.request.tasks]))


class SessionPort(Port):
    def __init__(self, handler):
        super().__init__(handler)
        self.terminated = []
        self.cancelled_questions = []

    def terminate_session(self, session_id, reason):
        self.terminated.append((session_id, reason))

    def cancel_question(self, session_id, question_id, reason):
        self.cancelled_questions.append((session_id, question_id, reason))


def disk_controller(tmp_path, root, coding=None, experiment=None, *, crash_when=None):
    engine = controller(tmp_path, root, coding, experiment)
    engine.compiler = NamedTaskCompiler()
    store = (CrashAfterSave(tmp_path / "state", crash_when) if crash_when is not None
             else JsonRunStore(tmp_path / "state"))
    engine.scheduler.store = engine.scheduler.artifact_registry.store = store
    return engine


def ask(request):
    ref = session(request)
    question = PendingQuestion(id="question_leaf", run_id=request.run_id,
        task_id=request.task_id, attempt_number=request.attempt_number, session_id=ref.id,
        recipient_session_id=request.upstream_session_id, text="Which metric?",
        requested_fields=["metric"], created_at=datetime.now(UTC))
    return AgentResult(status=ModuleStatus.NEEDS_USER_INPUT, report="Original child question.  \n",
        session=ref, control=ControlSignal(action="ask_user", source_session_id=ref.id,
            source_turn=0, source_call_id="ask_metric", question=question))


def final_report(request):
    result = conclude(request)
    result.artifacts[0].content = json.dumps({"verdict": "inconclusive", "statement": "done",
        "limitations": ["Unfinished child work was cancelled and supplies no validated evidence"]})
    return result


def reports(run):
    return {ref.task_id: ref.id for ref in run.artifacts.values() if ref.kind == "module_report"}


def assert_single_reports(run):
    refs = [ref for ref in run.artifacts.values() if ref.kind == "module_report"]
    assert len(refs) == len({(ref.task_id, ref.attempt_number) for ref in refs})
    for task in run.workflow.tasks:
        for attempt in task.attempts:
            if attempt.report_artifact_id is not None:
                ref = run.artifacts[attempt.report_artifact_id]
                assert Path(url2pathname(urlparse(ref.uri).path)).read_text() == attempt.report


def cancellation_cut(boundary, task_id):
    def reached(run):
        pending = run.pending_cancellation
        if pending is None:
            return False
        task = run.task(task_id)
        attempt = task.attempts[-1]
        if boundary == "consumed_interrupt":
            return bool(run.consumed_control_ids) and task.status != TaskStatus.CANCELLED
        if task.status != TaskStatus.CANCELLED or attempt.session.status != SessionStatus.CANCELLED:
            return False
        if boundary == "cancelled_session":
            return attempt.report_artifact_id is None
        if attempt.report_artifact_id is None:
            return False
        report_pending = f"task:{task.id}:attempt:{attempt.number}:report" in run.pending_registrations
        return report_pending if boundary == "bound_report" else not report_pending
    return reached


@pytest.mark.parametrize("boundary", [
    "consumed_interrupt", "cancelled_session", "bound_report", "acknowledged_report",
])
def test_interrupt_recovery_finishes_cancellation_before_dispatch(tmp_path, boundary):
    holder = {}

    def root_handler(request, turn):
        if turn == 0:
            result = delegate(request, "coding", "First child")
            result.control.work_request.request = WorkRequestDraft(objective="Two independent jobs", tasks=[
                {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "coding", "instruction": "First child"},
                {"selection":{"kind":"preset","preset_ref":"experiment@1","reason":"Explicit test preset"}, "key": "experiment", "instruction": "Undispatched sibling"}])
            return result
        holder["engine"].control_inbox.submit(request.run_id, "interrupt")
        raise OperationCancelled("User interrupted current execution")

    root, coding, sibling = Port(root_handler), SessionPort(lambda r, n: ask(r)), Port(lambda r, n: complete(r))
    engine = holder["engine"] = disk_controller(tmp_path, root, coding, sibling,
        crash_when=cancellation_cut(boundary, "task_coding"))
    run_id = "run_interrupt_lifecycle"
    with pytest.raises(ProcessCrash):
        engine.create_run(run_id, research_request(goal="Two independent jobs"))
    saved = JsonRunStore(tmp_path / "state").load(run_id)
    assert saved.pending_cancellation is not None
    assert len(saved.consumed_control_ids) == 1
    assert len(sibling.requests) == 0

    # A second process can crash after the final cancellation/pause checkpoint.
    fresh = disk_controller(tmp_path, root, coding, sibling, crash_when=lambda run:
        run.status == RunStatus.PAUSED and run.pending_cancellation is None)
    with pytest.raises(ProcessCrash):
        fresh.resume_run(run_id)
    paused = JsonRunStore(tmp_path / "state").load(run_id)
    before = (reports(paused), paused.runtime_notices)
    recovered = disk_controller(tmp_path, root, coding, sibling).run_until_stable(run_id)
    assert recovered.status == RunStatus.PAUSED
    assert recovered.work("work_coding").status == WorkRequestStatus.CANCELLED
    assert {recovered.task(key).status for key in ("task_coding", "task_experiment")} == {TaskStatus.CANCELLED}
    assert recovered.task("task_experiment").attempts == []
    assert (len(root.requests), len(coding.requests), len(sibling.requests)) == (2, 1, 0)
    assert len(coding.terminated) == 1
    assert recovered.child_questions == {} and recovered.pending_question is None
    assert recovered.pending_module_results == recovered.pending_registrations == {}
    assert (reports(recovered), recovered.runtime_notices) == before
    assert [notice["type"] for notice in recovered.runtime_notices].count("work_interrupted") == 1
    assert [notice["type"] for notice in recovered.runtime_notices].count("user_interrupted") == 1
    assert_single_reports(recovered)


def test_nested_interrupt_preserves_completed_sibling_and_original_reports(tmp_path):
    holder = {}

    def root_handler(request, turn):
        assert turn == 0
        result = delegate(request, "coding_owner", "Nested owner")
        result.control.work_request.request = WorkRequestDraft(objective="Completed and unfinished work", tasks=[
            {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "experiment_done", "instruction": "Complete before interrupt"},
            {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "coding_owner", "instruction": "Nested owner"}])
        return result

    def owner(request, turn):
        if turn == 0:
            return delegate(request, "experiment_leaf", "Nested question")
        holder["engine"].control_inbox.submit(request.run_id, "interrupt")
        raise OperationCancelled("User interrupted nested work")

    root, coding = Port(root_handler), SessionPort(owner)
    experiment = SessionPort(lambda r, n: complete(r) if r.task_id == "task_experiment_done" else ask(r))
    engine = holder["engine"] = disk_controller(tmp_path, root, coding, experiment,
        crash_when=cancellation_cut("bound_report", "task_experiment_leaf"))
    run_id = "run_nested_interrupt"
    with pytest.raises(ProcessCrash):
        engine.create_run(run_id, research_request(goal="Keep actual completed work"))
    saved = JsonRunStore(tmp_path / "state").load(run_id)
    completed = saved.task("task_experiment_done").model_dump(mode="json")
    report_id = saved.task("task_experiment_done").attempts[0].report_artifact_id
    recovered = disk_controller(tmp_path, root, coding, experiment).resume_run(run_id)
    assert recovered.status == RunStatus.PAUSED
    assert recovered.task("task_experiment_done").model_dump(mode="json") == completed
    assert reports(recovered)["task_experiment_done"] == report_id
    assert recovered.task("task_coding_owner").status == recovered.task("task_experiment_leaf").status == TaskStatus.CANCELLED
    assert recovered.work("work_experiment_leaf").result_delivered
    assert recovered.child_questions == {} and recovered.pending_question is None
    assert len(coding.terminated) == len(experiment.terminated) == 1
    assert experiment.terminated[0][0] != recovered.task("task_experiment_done").attempts[0].session.id
    assert (len(root.requests), len(coding.requests), len(experiment.requests)) == (1, 2, 2)
    assert recovered.pending_module_results == recovered.pending_registrations == {}
    assert_single_reports(recovered)
    again = disk_controller(tmp_path, root, coding, experiment).run_until_stable(run_id)
    assert (reports(again), again.runtime_notices) == (reports(recovered), recovered.runtime_notices)


@pytest.mark.parametrize("second_boundary", ["cancelled_session", "bound_report", "acknowledged_report"])
@pytest.mark.parametrize("owner_outcome", ["failed", "rejected_completion"])
def test_failed_owner_report_replays_cleanup_until_child_question_is_removed(tmp_path, second_boundary, owner_outcome):
    def owner(request, turn):
        if turn == 0:
            return delegate(request, "experiment", "Nested question")
        if owner_outcome == "rejected_completion":
            return complete(request)
        return AgentResult(status=ModuleStatus.FAILED, report="Original owner failure.  \n",
            session=session(request, SessionStatus.FAILED), error=ModuleError(
                code=ErrorCode.TOOL_FAILED, message="Owner stopped", retryable=False))

    root = Port(lambda r, n: delegate(r, "coding", "Owner") if n == 0 else final_report(r))
    coding, experiment = SessionPort(owner), SessionPort(lambda r, n: ask(r))

    def failed_report_bound(run):
        task = run.task("task_coding") if any(t.id == "task_coding" for t in run.workflow.tasks) else None
        return (task is not None and task.status == TaskStatus.FAILED and task.attempts[-1].report_artifact_id is not None
            and "task:task_coding:attempt:1" in run.pending_module_results and run.pending_cancellation is None)

    engine = disk_controller(tmp_path, root, coding, experiment, crash_when=failed_report_bound)
    run_id = "run_failed_owner_lifecycle"
    with pytest.raises(ProcessCrash):
        engine.create_run(run_id, research_request(goal="Report the actual failed work"))
    saved = JsonRunStore(tmp_path / "state").load(run_id)
    owner_report = saved.task("task_coding").attempts[0].report_artifact_id
    assert saved.child_questions and saved.task("task_experiment").status == TaskStatus.NEEDS_USER_INPUT
    receiving = saved.pending_module_results["task:task_coding:attempt:1"]
    assert receiving.result.status == (ModuleStatus.COMPLETED if owner_outcome == "rejected_completion" else ModuleStatus.FAILED)
    assert receiving.received_result.status == ModuleStatus.FAILED

    fresh = disk_controller(tmp_path, root, coding, experiment,
        crash_when=cancellation_cut(second_boundary, "task_experiment"))
    with pytest.raises(ProcessCrash):
        fresh.run_until_stable(run_id)
    recovered = disk_controller(tmp_path, root, coding, experiment).run_until_stable(run_id)
    assert recovered.status == RunStatus.COMPLETED, recovered.terminal_error
    assert recovered.task("task_coding").status == TaskStatus.FAILED
    assert recovered.task("task_coding").attempts[0].error.code == (
        ErrorCode.CONTRACT_ERROR if owner_outcome == "rejected_completion" else ErrorCode.TOOL_FAILED)
    assert reports(recovered)["task_coding"] == owner_report
    assert recovered.task("task_experiment").status == TaskStatus.CANCELLED
    assert recovered.work("work_experiment").status == WorkRequestStatus.CANCELLED
    assert recovered.work("work_experiment").result_delivered
    assert recovered.child_questions == {} and recovered.pending_question is None
    assert (len(root.requests), len(coding.requests), len(experiment.requests)) == (2, 2, 1)
    assert len(experiment.terminated) == 1 and coding.terminated == []
    assert recovered.pending_module_results == recovered.pending_registrations == {}
    assert [notice["type"] for notice in recovered.runtime_notices].count("work_interrupted") == 1
    assert_single_reports(recovered)
    again = disk_controller(tmp_path, root, coding, experiment).run_until_stable(run_id)
    assert (reports(again), again.runtime_notices) == (reports(recovered), recovered.runtime_notices)


@pytest.mark.parametrize("outcome", ["completed", "failed"])
@pytest.mark.parametrize("expired", [False, True])
def test_root_terminal_commit_recovers_only_remaining_acknowledgements(tmp_path, monkeypatch, outcome, expired):
    report = "Original root terminal report.  \n"

    def result(request, turn):
        assert turn == 0
        if outcome == "failed":
            return AgentResult(status=ModuleStatus.FAILED, report=report,
                session=session(request, SessionStatus.FAILED), error=ModuleError(
                    code=ErrorCode.TOOL_FAILED, message="Known root failure", retryable=False))
        return conclude(request).model_copy(update={"report": report})

    def committed(run):
        return (run.status.value == outcome and run.scientific_report == report
            and run.task("task_root").attempts[0].report_artifact_id is not None
            and (outcome == "failed" or run.final_report_artifact_id is not None))

    root = Port(result)
    engine = disk_controller(tmp_path, root, crash_when=committed)
    run_id = "run_root_terminal_ack"
    with pytest.raises(ProcessCrash):
        engine.create_run(run_id, research_request(goal="Recover the committed terminal result"))
    saved = JsonRunStore(tmp_path / "state").load(run_id)
    assert saved.pending_module_results and saved.pending_registrations
    allocated = set(saved.artifacts)
    final_report_id = saved.final_report_artifact_id

    def forbidden(*args, **kwargs):
        raise AssertionError("a committed terminal result needs only acknowledgement")

    fresh = disk_controller(tmp_path, root)
    if expired:
        from datetime import timedelta
        from sciagent_orchestrator import controller as controller_module, scheduler as scheduler_module

        class Clock:
            @classmethod
            def now(cls, tz=None):
                return datetime.now(UTC) + timedelta(days=1)

        monkeypatch.setattr(controller_module, "datetime", Clock)
        monkeypatch.setattr(scheduler_module, "datetime", Clock)
    monkeypatch.setattr(root, "invoke", forbidden)
    monkeypatch.setattr(fresh.gate, "validate", forbidden)
    monkeypatch.setattr(fresh.report_renderer, "render", forbidden)
    recovered = fresh.run_until_stable(run_id)
    assert recovered.status.value == outcome
    assert recovered.scientific_report == report
    assert recovered.final_report_artifact_id == final_report_id
    assert set(recovered.artifacts) == allocated
    assert recovered.pending_module_results == recovered.pending_registrations == {}
    assert len(root.requests) == len(recovered.task("task_root").attempts) == 1
    assert_single_reports(recovered)
    again = disk_controller(tmp_path, root).run_until_stable(run_id)
    assert (reports(again), again.runtime_notices) == (reports(recovered), recovered.runtime_notices)


def test_abandon_recovery_finishes_saved_control_without_repeating_parent_call(tmp_path):
    holder = {}

    def root_handler(request, turn):
        if turn == 0:
            return delegate(request, "coding", "Old work")
        if turn == 1:
            ref = session(request)
            return AgentResult(status=ModuleStatus.ABANDON_WORK, report="End old work", session=ref,
                control=ControlSignal(action="abandon_work", source_session_id=ref.id,
                    source_turn=1, source_call_id="abandon_original",
                    abandon_work_id=request.task_context["active_work_request_id"], reason="Reassess"))
        return final_report(request)

    def child(request, turn):
        holder["engine"].control_inbox.submit(request.run_id, "input", text="Reassess old work")
        return AgentResult(status=ModuleStatus.YIELDED, report="Original partial child report.  \n", session=session(request))

    root, coding = Port(root_handler), SessionPort(child)
    engine = holder["engine"] = disk_controller(tmp_path, root, coding,
        crash_when=cancellation_cut("cancelled_session", "task_coding"))
    run_id = "run_abandon_lifecycle"
    with pytest.raises(ProcessCrash):
        engine.create_run(run_id, research_request(goal="End old work and conclude"))
    saved = JsonRunStore(tmp_path / "state").load(run_id)
    assert saved.pending_cancellation.source == "abandon"
    assert "task:task_root:attempt:1" in saved.pending_module_results

    def received_abandon(run):
        pending = run.pending_module_results.get("task:task_root:attempt:1")
        return pending is not None and pending.received_result is not None and pending.received_result.status == ModuleStatus.ABANDON_WORK

    fresh = disk_controller(tmp_path, root, coding, crash_when=received_abandon)
    with pytest.raises(ProcessCrash):
        fresh.run_until_stable(run_id)
    recovered = disk_controller(tmp_path, root, coding).run_until_stable(run_id)
    assert recovered.status == RunStatus.COMPLETED, recovered.terminal_error
    assert recovered.work("work_coding").status == WorkRequestStatus.CANCELLED
    assert recovered.task("task_coding").status == TaskStatus.CANCELLED
    assert (len(root.requests), len(coding.requests), len(coding.terminated)) == (3, 1, 1)
    assert len([key for key in recovered.received_controls if "abandon_original" in key]) == 1
    assert [notice["type"] for notice in recovered.runtime_notices].count("work_interrupted") == 1
    assert recovered.pending_module_results == recovered.pending_registrations == {}
    assert_single_reports(recovered)
    again = disk_controller(tmp_path, root, coding).run_until_stable(run_id)
    assert (reports(again), again.runtime_notices) == (reports(recovered), recovered.runtime_notices)


def waiting_with_child_question(tmp_path, question_kind):
    """Use public execution to pause on a permission or escalated business question."""
    def root_handler(request, turn):
        if turn == 0:
            result = delegate(request, "coding", "Questioned child")
            result.control.work_request.request = WorkRequestDraft(objective="Two jobs", tasks=[
                {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "coding", "instruction": "Questioned child"},
                {"selection":{"kind":"preset","preset_ref":"experiment@1","reason":"Explicit test preset"}, "key": "experiment", "instruction": "Undispatched sibling"}])
            return result
        if question_kind == "escalated" and turn == 1:
            ref = session(request)
            child_question, = request.child_questions
            question = PendingQuestion(id="question_escalated", run_id=request.run_id,
                task_id=request.task_id, attempt_number=request.attempt_number, session_id=ref.id,
                related_question_id=child_question.id, text="Please choose the child's metric",
                requested_fields=["metric"], created_at=datetime.now(UTC))
            return AgentResult(status=ModuleStatus.NEEDS_USER_INPUT, report="Escalated original question",
                session=ref, control=ControlSignal(action="ask_user", source_session_id=ref.id,
                    source_turn=1, source_call_id="escalate_metric", question=question))
        return final_report(request)

    def child(request, turn):
        result = ask(request)
        if question_kind == "permission":
            result.control.question.action = ActionSnapshot(action_id="action_original", tool="run_shell",
                arguments={"command": "original-command"}, run_id=request.run_id,
                task_id=request.task_id, attempt_number=request.attempt_number, session_id=result.session.id)
        return result

    root, coding, sibling = SessionPort(root_handler), SessionPort(child), Port(lambda r, n: complete(r))
    engine = disk_controller(tmp_path, root, coding, sibling)
    paused = engine.create_run("run_question_lifecycle", research_request(goal="Keep original question scope"))
    assert paused.status == RunStatus.PAUSED and paused.pending_question is not None
    return engine, paused, root, coding, sibling


def crash_with_consumed_interrupt(tmp_path, paused, root, coding, sibling):
    engine = disk_controller(tmp_path, root, coding, sibling,
        crash_when=cancellation_cut("consumed_interrupt", "task_coding"))
    engine.control_inbox.submit(paused.run_id, "interrupt")
    with pytest.raises(ProcessCrash):
        engine.resume_run(paused.run_id)
    saved = JsonRunStore(tmp_path / "state").load(paused.run_id)
    assert saved.pending_cancellation is not None and saved.pending_question is not None


def test_new_user_input_survives_pending_cancellation_and_is_processed_after_pause(tmp_path):
    _, paused, root, coding, sibling = waiting_with_child_question(tmp_path, "escalated")
    crash_with_consumed_interrupt(tmp_path, paused, root, coding, sibling)
    fresh = disk_controller(tmp_path, root, coding, sibling)
    event = fresh.control_inbox.submit(paused.run_id, "input", text="New constraint received during cancellation")
    stopped = fresh.resume_run(paused.run_id)
    assert stopped.status == RunStatus.PAUSED and stopped.pending_cancellation is None
    assert event.event_id not in stopped.consumed_control_ids
    assert [entry.event_id for entry in fresh.control_inbox.pending(paused.run_id, stopped.consumed_control_ids)] == [event.event_id]
    assert (len(root.requests), len(coding.requests), len(sibling.requests)) == (2, 1, 0)

    # Observing stability may consume the input but must preserve the explicit pause.
    waiting = disk_controller(tmp_path, root, coding, sibling).run_until_stable(paused.run_id)
    assert waiting.status == RunStatus.PAUSED
    assert [entry.text for entry in waiting.task_inputs["task_root"]] == [event.text]
    assert waiting.consumed_control_ids.count(event.event_id) == 1
    assert (len(root.requests), len(coding.requests), len(sibling.requests)) == (2, 1, 0)
    completed = disk_controller(tmp_path, root, coding, sibling).resume_run(paused.run_id)
    assert completed.status == RunStatus.COMPLETED, completed.terminal_error
    assert [entry.text for entry in root.requests[-1].upstream_inputs] == [event.text]
    assert completed.task("task_coding").status == completed.task("task_experiment").status == TaskStatus.CANCELLED
    assert len(coding.terminated) == 1 and len(sibling.requests) == 0
    assert_single_reports(completed)


@pytest.mark.parametrize("question_kind", ["permission", "escalated"])
def test_old_answer_cannot_revive_child_while_cancellation_is_pending(tmp_path, question_kind):
    _, paused, root, coding, sibling = waiting_with_child_question(tmp_path, question_kind)
    old_question = paused.pending_question
    crash_with_consumed_interrupt(tmp_path, paused, root, coding, sibling)
    fresh = disk_controller(tmp_path, root, coding, sibling)
    with pytest.raises(ValueError, match="answer does not match pending question"):
        fresh.answer_question(paused.run_id, UserAnswer(question_id=old_question.id,
            values={"metric": "mse"}, answered_at=datetime.now(UTC)))
    stopped = JsonRunStore(tmp_path / "state").load(paused.run_id)
    assert stopped.status == RunStatus.PAUSED and stopped.pending_cancellation is None
    assert stopped.pending_question is None and stopped.child_questions == {} and stopped.answers == []
    assert stopped.task("task_coding").status == stopped.task("task_experiment").status == TaskStatus.CANCELLED
    assert len(root.requests) == (2 if question_kind == "escalated" else 1)
    assert (len(coding.requests), len(sibling.requests), len(coding.terminated)) == (1, 0, 1)
    assert len(root.cancelled_questions) == (1 if question_kind == "escalated" else 0)
    assert stopped.pending_module_results == stopped.pending_registrations == {}
    again = disk_controller(tmp_path, root, coding, sibling).run_until_stable(paused.run_id)
    assert (reports(again), again.runtime_notices) == (reports(stopped), stopped.runtime_notices)


@pytest.mark.parametrize("outcome, stop_confirmed", [
    (ModuleStatus.FAILED, None), (ModuleStatus.BLOCKED, None), (ModuleStatus.FAILED, False),
])
def test_root_owner_failure_recovers_child_cleanup_after_report_checkpoint(tmp_path, monkeypatch, outcome, stop_confirmed):
    error = ModuleError(code=ErrorCode.INTERRUPTED if stop_confirmed is False else ErrorCode.TOOL_FAILED,
        message="Original root stop failure", retryable=False,
        details={"stop_confirmed": False, "pid": 123} if stop_confirmed is False else {})

    def owner(request, turn):
        if turn == 0:
            return delegate(request, "coding", "Child asks a question")
        return AgentResult(status=outcome, report="Original root failure.  \n",
            session=session(request, SessionStatus(outcome.value)), error=error)

    root, coding = SessionPort(owner), SessionPort(lambda r, n: ask(r))
    engine = disk_controller(tmp_path, root, coding, crash_when=lambda run:
        run.task("task_root").attempts and run.task("task_root").attempts[0].report_artifact_id is not None
        and "task:task_root:attempt:1" in run.pending_module_results)
    run_id = "run_failed_root_owner"
    with pytest.raises(ProcessCrash):
        engine.create_run(run_id, research_request(goal="Preserve root failure and close descendants"))
    saved = JsonRunStore(tmp_path / "state").load(run_id)
    original_report = saved.task("task_root").attempts[0].report_artifact_id
    assert saved.child_questions and saved.task("task_coding").status == TaskStatus.NEEDS_USER_INPUT

    def forbidden(*args, **kwargs):
        raise AssertionError("root failure recovery must not invoke another root decision")

    monkeypatch.setattr(root, "invoke", forbidden)
    recovered = disk_controller(tmp_path, root, coding).run_until_stable(run_id)
    assert recovered.status == RunStatus.FAILED and recovered.terminal_error == error
    assert recovered.task("task_root").status == (TaskStatus.BLOCKED if outcome == ModuleStatus.BLOCKED else TaskStatus.FAILED)
    assert reports(recovered)["task_root"] == original_report
    assert recovered.task("task_coding").status == TaskStatus.CANCELLED
    assert recovered.work("work_coding").status == WorkRequestStatus.CANCELLED
    assert recovered.work("work_coding").result_delivered
    assert recovered.pending_question is None and recovered.child_questions == {}
    assert len(root.requests) == 2 and len(coding.requests) == len(coding.terminated) == 1
    assert recovered.pending_module_results == recovered.pending_registrations == {}
    assert_single_reports(recovered)


@pytest.mark.parametrize("expired", [False, True])
def test_interrupt_after_saved_unconfirmed_stop_preserves_terminal_failure(tmp_path, monkeypatch, expired):
    error = ModuleError(code=ErrorCode.INTERRUPTED, message="Original process stop is unconfirmed",
        retryable=False, details={"stop_confirmed": False, "pid": 123, "command": "original-running-command"})

    def owner(request, turn):
        assert turn == 0
        result = delegate(request, "coding", "Running child")
        result.control.work_request.request = WorkRequestDraft(objective="Two jobs", tasks=[
            {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "coding", "instruction": "Running child"},
            {"selection":{"kind":"preset","preset_ref":"experiment@1","reason":"Explicit test preset"}, "key": "experiment", "instruction": "Undispatched sibling"}])
        return result

    root = Port(owner)
    coding = SessionPort(lambda r, n: AgentResult(status=ModuleStatus.FAILED, report="Original stop failure.  \n",
        session=session(r, SessionStatus.FAILED), error=error))
    sibling = Port(lambda r, n: complete(r))

    def received_failure(run):
        pending = run.pending_module_results.get("task:task_coding:attempt:1")
        return pending is not None and pending.received_result is not None

    engine = disk_controller(tmp_path, root, coding, sibling, crash_when=received_failure)
    run_id = "run_stop_failure_interrupt"
    with pytest.raises(ProcessCrash):
        engine.create_run(run_id, research_request(goal="Keep original process stop facts"))
    fresh = disk_controller(tmp_path, root, coding, sibling)
    if expired:
        from datetime import timedelta
        from sciagent_orchestrator import controller as controller_module, scheduler as scheduler_module

        class Clock:
            @classmethod
            def now(cls, tz=None):
                return datetime.now(UTC) + timedelta(days=1)

        monkeypatch.setattr(controller_module, "datetime", Clock)
        monkeypatch.setattr(scheduler_module, "datetime", Clock)
    fresh.control_inbox.submit(run_id, "interrupt")
    recovered = fresh.resume_run(run_id)
    assert recovered.status == RunStatus.FAILED and recovered.terminal_error == error
    assert recovered.task("task_coding").status == TaskStatus.FAILED
    assert recovered.task("task_coding").attempts[0].error == error
    assert recovered.task("task_experiment").status == TaskStatus.CANCELLED
    assert recovered.task("task_experiment").attempts == []
    assert (len(root.requests), len(coding.requests), len(sibling.requests)) == (1, 1, 0)
    notices = [entry for entry in recovered.runtime_notices if entry["type"] in {"work_interrupted", "user_interrupted"}]
    assert notices and all(entry["stop_confirmed"] is False for entry in notices)
    assert recovered.pending_module_results == recovered.pending_registrations == {}
    with pytest.raises(ValueError, match="terminal Run cannot resume"):
        disk_controller(tmp_path, root, coding, sibling).resume_run(run_id)
    assert_single_reports(recovered)
