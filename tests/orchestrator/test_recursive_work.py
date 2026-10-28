"""Uniform Task ownership and ordered, direct-parent update boundaries."""
from resagent2_executor import default_execution_registry
from e2e.scheduler_fixtures import executor_bindings
from datetime import UTC, datetime, timedelta
import json
import pytest

from resagent2_contracts import (
    AgentOwner, AgentResult, ArtifactCandidate, AttemptStatus, BusinessInput,
    ControlSignal, ExecutionLimits, ModuleStatus, ModuleError, ErrorCode, ResearchRequest, RunStatus, RunBudget, RunPermissions,
    SessionRef, SessionStatus, PendingQuestion, RecordedAnswer, TaskProposal, TaskStatus, WorkRequest, WorkRequestDraft,
    WorkRequestStatus, WorkflowAgentKind, WorkflowAgentRegistry, WorkflowPatch,
    task_session_id,
)
from resagent2_orchestrator import (
    CompilationResult, DeterministicWorkInterpreter, InMemoryRunStore, ModuleBinding,
    ResearchController, WorkflowScheduler,
)
from resagent2_runtime.controls import RunControlInbox


class DirectCompiler:
    def compile(self, work, *, current, **kwargs):
        return CompilationResult(WorkflowPatch(work_request_id=work.id,
            based_on_revision=current.revision, add_tasks=[TaskProposal(execution_ref=(None if item.key == "scientific" else item.key + "@1"),
                id="task_" + item.key, work_request_id=work.id,
                workflow_agent_kind=("scientific" if item.key == "scientific" else "executor"),
                instruction=item.instruction) for item in work.request.tasks]))


class Port:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        return self.handler(request, len(self.requests) - 1)


def session(request, status=SessionStatus.PAUSED):
    now = datetime.now(UTC)
    sid = task_session_id(request.run_id, request.task_id, request.attempt_number)
    return SessionRef(id=sid, module=request.agent, state_uri="memory://" + sid,
        status=status, created_at=now, updated_at=now)


def delegate(request, key, instruction):
    ref = session(request)
    now = datetime.now(UTC)
    work = WorkRequest(id="work_" + key, run_id=request.run_id,
        owner_session_id=ref.id, owner_task_id=request.task_id,
        owner_attempt_number=request.attempt_number, depth=request.work_depth + 1,
        processed_input_ids=request.input_ids,
        request=WorkRequestDraft(objective="private parent management objective",
            tasks=[{"selection":{"kind":"preset","preset_ref":key.split("_", 1)[0] + "@1","reason":"Explicit test preset"}, "key": key, "instruction": instruction}]),
        created_at=now, updated_at=now)
    return AgentResult(status=ModuleStatus.REQUEST_WORK, report="delegated", session=ref,
        control=ControlSignal(action="request_work", source_session_id=ref.id,
            source_turn=0, source_call_id="delegate_" + key, work_request=work))


def complete(request):
    return AgentResult(status=ModuleStatus.COMPLETED, report="original " + request.task_id,
        session=session(request, SessionStatus.COMPLETED))


def conclude(request):
    return AgentResult(status=ModuleStatus.COMPLETED, report="root original report",
        session=session(request, SessionStatus.COMPLETED), artifacts=[ArtifactCandidate(
            kind="scientific_opinion", path="opinion.json", media_type="application/json",
            summary="Conclusion", content=json.dumps({"verdict": "inconclusive", "statement": "done"}))])


def update(request, text, *, targets=None):
    ref = session(request)
    if targets is None:
        targets = [item["task_id"] for item in request.task_context["child_tasks"]]
    inputs = [{"task_ids": targets, "text": text}] if targets else []
    return AgentResult(status=ModuleStatus.UPDATE_WORK, report="updated", session=ref,
        control=ControlSignal(action="update_work", source_session_id=ref.id,
            source_turn=1, source_call_id="update", work_update={
                "work_request_id": request.task_context["active_work_request_id"],
                "provided_input_ids": request.input_ids, "inputs": inputs, "answers": []}))


def controller(tmp_path, root, coding=None, experiment=None):
    bindings = {}
    for kind, port in (("coding", coding), ("experiment", experiment)):
        if port:
            bindings[kind] = ModuleBinding(AgentOwner.EXECUTOR, port)
    scheduler = WorkflowScheduler(bindings=executor_bindings(bindings), store=InMemoryRunStore(),
        artifact_root=tmp_path / "artifacts", data_root=tmp_path / "runs")
    return ResearchController(scientific_port=root, compiler=DirectCompiler(), scheduler=scheduler,
        registry=default_execution_registry(), interpreter=DeterministicWorkInterpreter(),
        control_inbox=RunControlInbox(tmp_path / "controls"))


def research_request(**kwargs):
    return ResearchRequest(budget=RunBudget(max_llm_calls=30, timeout_seconds=60), permissions=RunPermissions(), **kwargs)


def test_root_only_uses_one_task_attempt_and_session_and_one_task_budget(tmp_path):
    root = Port(lambda request, turn: conclude(request))
    engine = controller(tmp_path, root)
    run = engine.create_run("run_root_only", research_request(goal="Original root requirement",
        execution_limits=ExecutionLimits(max_tasks=1)))
    assert run.status == RunStatus.COMPLETED, (run.terminal_error, run.completion_violations)
    assert len(run.workflow.tasks) == len(run.work_requests) == 1
    task = run.workflow.tasks[0]
    assert task.workflow_agent_kind == WorkflowAgentKind.SCIENTIFIC
    assert task.attempts[0].session.id == task_session_id(run.run_id, task.id, 1)
    assert run.work_requests[0].status == WorkRequestStatus.CONSUMED
    assert run.work_requests[0].outcome.tasks[0].task_id == task.id
    opinion = next(ref for ref in run.artifacts.values() if ref.kind == "scientific_opinion")
    assert (opinion.task_id, opinion.attempt_number, opinion.session_id) == (task.id, 1, task.attempts[0].session.id)


def test_recursive_work_resumes_each_owner_in_its_original_attempt_and_session(tmp_path):
    root = Port(lambda r, n: delegate(r, "coding", "Coding initial instruction") if n == 0 else conclude(r))
    coding = Port(lambda r, n: delegate(r, "experiment", "Experiment initial instruction") if n == 0 else complete(r))
    experiment = Port(lambda r, n: complete(r))
    engine = controller(tmp_path, root, coding, experiment)
    run = engine.create_run("run_recursive", research_request(goal="Root private initial requirement"))
    assert run.status == RunStatus.COMPLETED, (run.terminal_error, run.completion_violations)
    assert [work.depth for work in run.work_requests] == [1, 2, 3]
    for port in (root, coding):
        first, resumed = port.requests
        assert resumed.parent_session_id == task_session_id(run.run_id, first.task_id, 1)
        assert resumed.attempt_number == first.attempt_number == 1
        assert len(resumed.work_results) == 1
    expected_catalog = [preset.preset_ref for preset in run.execution_presets]
    assert [item.preset_ref for item in root.requests[0].execution_catalog] == expected_catalog
    assert [item.preset_ref for item in coding.requests[0].execution_catalog] == expected_catalog
    assert experiment.requests[0].execution_catalog == []
    assert experiment.requests[0].permissions.request_work is False
    assert coding.requests[0].instruction == "Coding initial instruction"
    assert "Root private" not in json.dumps(coding.requests[0].task_context)
    assert "private parent management objective" not in json.dumps(experiment.requests[0].task_context)
    assert all(len(task.attempts) == 1 for task in run.workflow.tasks)


def test_user_update_gates_owner_chain_before_leaf_continues(tmp_path):
    calls = []
    holder = {}
    def root_handler(request, turn):
        calls.append("root_" + str(turn))
        if turn == 0:
            return delegate(request, "coding", "Coding original")
        if turn == 1:
            assert [event.text for event in request.upstream_inputs] == ["New evaluation constraint"]
            return update(request, "Coding: use the revised evaluation constraint")
        return conclude(request)
    def coding_handler(request, turn):
        calls.append("coding_" + str(turn))
        if turn == 0:
            return delegate(request, "experiment", "Experiment original")
        if turn == 1:
            assert [event.text for event in request.upstream_inputs] == ["Coding: use the revised evaluation constraint"]
            return update(request, "Experiment: apply the revised evaluation constraint")
        return complete(request)
    def leaf_handler(request, turn):
        calls.append("leaf_" + str(turn))
        if turn == 0:
            holder["engine"].control_inbox.submit(request.run_id, "input", text="New evaluation constraint")
            return AgentResult(status=ModuleStatus.YIELDED, report="yield for input", session=session(request))
        assert [event.text for event in request.upstream_inputs] == ["Experiment: apply the revised evaluation constraint"]
        assert request.instruction == "Experiment original"
        return complete(request)
    ports = [Port(root_handler), Port(coding_handler), Port(leaf_handler)]
    engine = holder["engine"] = controller(tmp_path, *ports)
    run = engine.create_run("run_update_chain", research_request(goal="Root original"))
    assert run.status == RunStatus.COMPLETED, (run.terminal_error, run.completion_violations)
    assert calls == ["root_0", "coding_0", "leaf_0", "root_1", "coding_1", "leaf_1", "coding_2", "root_2"]
    for port in ports:
        assert {request.attempt_number for request in port.requests} == {1}
        assert len({session(request).id for request in port.requests}) == 1
    assert [run.input_ids(task.id) for task in run.workflow.tasks] == [port.requests[-1].input_ids for port in ports]
    for port in (ports[1], ports[2]):
        frozen = port.requests[0].execution_definition
        assert all(request.execution_definition == frozen for request in port.requests)
        assert run.task(port.requests[0].task_id).execution_ref == frozen.ref
        assert run.execution_definitions[frozen.ref] == frozen


def test_received_input_order_does_not_depend_on_source_timestamps(tmp_path):
    engine = controller(tmp_path, Port(lambda r, n: conclude(r)))
    run = engine.create_run("run_order", research_request(goal="Order"))
    now = datetime.now(UTC)
    run.append_input("task_root", BusinessInput(event_id="first", text="first", source="user", created_at=now))
    run.append_input("task_root", BusinessInput(event_id="second", text="second", source="user", created_at=now-timedelta(days=1)))
    run.append_answer(RecordedAnswer(question_id="question_old_timestamp", question_text="Later answer?",
        requested_fields=["answer"], values={"answer": "yes"}, answered_at=now-timedelta(days=2),
        run_id=run.run_id, task_id="task_root", attempt_number=1, session_id=run.scientific_session.id))
    assert run.input_ids("task_root") == ["first", "second", "question_old_timestamp"]
    request = engine.scheduler._module_request(run, run.task("task_root"), 1,
        parent_session_id=run.scientific_session.id)
    from resagent2_components.materials import request_task_context
    dialogue = request_task_context(request)["business_dialogue"]
    assert [entry["record"].get("event_id", entry["record"].get("question_id")) for entry in dialogue] == [
        "first", "second", "question_old_timestamp"]


class ProcessCrash(BaseException):
    pass


def paused_parent_with_two_children(tmp_path):
    """Reach the owner boundary through a real direct-child question."""
    def parent(request, turn):
        if turn:
            raise ProcessCrash()
        result = delegate(request, "coding", "Coding original")
        work = result.control.work_request
        work.request = WorkRequestDraft(objective="Parent-only objective", tasks=[
            {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "coding", "instruction": "Coding original"},
            {"selection":{"kind":"preset","preset_ref":"experiment@1","reason":"Explicit test preset"}, "key": "experiment", "instruction": "Experiment original"}])
        return result
    def asking_child(request, turn):
        ref = session(request)
        question = PendingQuestion(id="question_metric", run_id=request.run_id,
            task_id=request.task_id, attempt_number=request.attempt_number, session_id=ref.id,
            recipient_session_id=request.upstream_session_id, text="Which metric?",
            requested_fields=["metric"], created_at=datetime.now(UTC))
        return AgentResult(status=ModuleStatus.NEEDS_USER_INPUT, report="Need metric", session=ref,
            control=ControlSignal(action="ask_user", source_session_id=ref.id,
                source_turn=0, source_call_id="question", question=question))
    engine = controller(tmp_path, Port(parent), Port(asking_child), Port(lambda r, n: complete(r)))
    with pytest.raises(ProcessCrash):
        engine.create_run("run_update_batch", research_request(goal="Original root"))
    run = engine.scheduler.store.load("run_update_batch")
    owner = run.task("task_root")
    request = engine.scheduler._module_request(run, owner, 1, parent_session_id=owner.attempts[0].session.id)
    return engine, run, request


def update_signal(request, *, inputs=(), answers=(), identity="batch"):
    return ControlSignal(action="update_work", source_session_id=request.parent_session_id,
        source_turn=5, source_call_id=identity, work_update={
            "work_request_id": request.task_context["active_work_request_id"],
            "provided_input_ids": request.input_ids, "inputs": list(inputs), "answers": list(answers)})


def test_update_batch_is_atomic_and_guidance_does_not_answer_a_question(tmp_path):
    engine, run, request = paused_parent_with_two_children(tmp_path)
    baseline = run.model_dump_json()
    bad = update_signal(request, inputs=[{"task_ids": ["task_coding", "task_experiment"], "text": "Use MSE"}],
        answers=[{"task_id": "task_experiment", "question_id": "question_metric", "values": {"metric": "mse"}}])
    with pytest.raises(ValueError, match="live direct child business question"):
        engine._apply_work_update(run, request, bad)
    assert run.model_dump_json() == baseline
    good = update_signal(request, inputs=[{"task_ids": ["task_coding", "task_experiment"], "text": "Use MSE"}])
    engine._apply_work_update(run, request, good)
    assert "question_metric" in run.child_questions
    assert run.task("task_coding").status == TaskStatus.NEEDS_USER_INPUT
    assert [event.text for event in run.task_inputs["task_coding"]] == ["Use MSE"]
    assert [event.text for event in run.task_inputs["task_experiment"]] == ["Use MSE"]
    before = run.model_dump_json()
    engine._apply_work_update(run, request, good)
    assert run.model_dump_json() == before


def test_exact_business_answer_resumes_only_its_question_and_preserves_original(tmp_path):
    engine, run, request = paused_parent_with_two_children(tmp_path)
    bad = update_signal(request, answers=[{"task_id": "task_coding", "question_id": "question_metric", "values": {"wrong": "mse"}}])
    with pytest.raises(ValueError):
        engine._apply_work_update(run, request, bad)
    assert run.answers == [] and "question_metric" in run.child_questions
    good = update_signal(request, answers=[{"task_id": "task_coding", "question_id": "question_metric", "values": {"metric": "mse"}}])
    engine._apply_work_update(run, request, good)
    answer, = run.answers
    assert (answer.task_id, answer.question_id, answer.question_text, answer.values) == (
        "task_coding", "question_metric", "Which metric?", {"metric": "mse"})
    assert answer.source == "parent" and answer.session_id == run.task("task_coding").attempts[0].session.id
    assert run.task("task_coding").status == TaskStatus.PENDING
    assert run.child_questions == {}


def test_completed_target_rejects_entire_update_batch(tmp_path):
    engine, run, request = paused_parent_with_two_children(tmp_path)
    run.task("task_experiment").status = TaskStatus.COMPLETED
    before = run.model_dump_json()
    signal = update_signal(request, inputs=[{"task_ids": ["task_coding"], "text": "First"},
        {"task_ids": ["task_experiment"], "text": "Cannot change completed work"}])
    with pytest.raises(ValueError, match="live direct child"):
        engine._apply_work_update(run, request, signal)
    assert run.model_dump_json() == before


def test_selected_child_update_preserves_sibling_context_and_exact_question_wait(tmp_path):
    from resagent2_components.materials import request_task_context

    engine, run, _ = paused_parent_with_two_children(tmp_path)
    run.append_input("task_root", BusinessInput(event_id="input_selective",
        text="Private upstream direction for the parent to assess", source="user",
        created_at=datetime.now(UTC)))
    work = run.active_child("task_root")
    assert run.needs_parent_processing(work)
    assert "task_experiment" not in engine.scheduler._ready_task_ids(run)
    owner = run.task("task_root")
    request = engine.scheduler._module_request(run, owner, 1,
        parent_session_id=owner.attempts[0].session.id)
    sibling = run.task("task_experiment")
    sibling_before = engine.scheduler._module_request(run, sibling, 1, parent_session_id=None)
    sibling_context = request_task_context(sibling_before)
    selected_text = "Coding-only amendment: use MSE for the original implementation"

    signal = update_signal(request, identity="selected_input", inputs=[{
        "task_ids": ["task_coding"], "text": selected_text}])
    engine._apply_work_update(run, request, signal)

    assert work.processed_input_ids == request.input_ids == ["input_selective"]
    assert not run.needs_parent_processing(work)
    assert "task_experiment" in engine.scheduler._ready_task_ids(run)
    sibling_after = engine.scheduler._module_request(run, sibling, 1, parent_session_id=None)
    assert sibling_after.input_ids == sibling_before.input_ids == []
    assert sibling_after.upstream_inputs == sibling_before.upstream_inputs == []
    assert sibling_after.user_answers == sibling_before.user_answers == []
    assert request_task_context(sibling_after) == sibling_context
    assert sibling_after.instruction == "Experiment original"
    assert run.task_inputs.get("task_experiment", []) == []
    assert selected_text not in json.dumps(sibling_after.model_dump(mode="json"))
    assert "Private upstream direction" not in json.dumps(sibling_after.model_dump(mode="json"))

    selected = run.task("task_coding")
    selected_request = engine.scheduler._module_request(run, selected, 1,
        parent_session_id=selected.attempts[0].session.id)
    assert selected_request.instruction == "Coding original"
    assert [event.text for event in selected_request.upstream_inputs] == [selected_text]
    assert len(selected_request.input_ids) == 1
    assert selected_request.user_answers == [] and run.answers == []
    assert "question_metric" in run.child_questions
    assert selected.status == TaskStatus.NEEDS_USER_INPUT
    assert "task_coding" not in engine.scheduler._ready_task_ids(run)

    answer = update_signal(request, identity="selected_answer", answers=[{
        "task_id": "task_coding", "question_id": "question_metric", "values": {"metric": "mse"}}])
    engine._apply_work_update(run, request, answer)
    assert selected.status == TaskStatus.PENDING and run.child_questions == {}
    assert {"task_coding", "task_experiment"} <= set(engine.scheduler._ready_task_ids(run))
    resumed = engine.scheduler._module_request(run, selected, 1,
        parent_session_id=selected.attempts[0].session.id)
    assert resumed.instruction == "Coding original"
    assert resumed.input_ids == [*selected_request.input_ids, "question_metric"]
    assert resumed.user_answers[0].values == {"metric": "mse"}
    assert [item["type"] for item in request_task_context(resumed)["business_dialogue"]] == ["input", "answer"]
    final_sibling = engine.scheduler._module_request(run, sibling, 1, parent_session_id=None)
    assert final_sibling.input_ids == [] and final_sibling.upstream_inputs == [] and final_sibling.user_answers == []
    assert request_task_context(final_sibling) == sibling_context
    assert run.task_inputs.get("task_experiment", []) == []


def test_stale_parent_update_cannot_release_children_and_empty_update_can_ack(tmp_path):
    engine, run, old_request = paused_parent_with_two_children(tmp_path)
    run.append_input("task_root", BusinessInput(event_id="new_input", text="New user direction",
        source="user", created_at=datetime.now(UTC)))
    work = run.active_child("task_root")
    assert run.needs_parent_processing(work)
    assert "task_experiment" not in engine.scheduler._ready_task_ids(run)
    before = run.model_dump_json()
    with pytest.raises(ValueError, match="before new direct input"):
        engine._apply_work_update(run, old_request, update_signal(old_request))
    assert run.model_dump_json() == before
    owner = run.task("task_root")
    fresh = engine.scheduler._module_request(run, owner, 1, parent_session_id=owner.attempts[0].session.id)
    engine._apply_work_update(run, fresh, update_signal(fresh))
    assert not run.needs_parent_processing(work)
    assert run.task_inputs.get("task_experiment", []) == []
    assert "question_metric" in run.child_questions
    assert "task_experiment" in engine.scheduler._ready_task_ids(run)


@pytest.mark.parametrize("restriction", ["capability", "depth"])
def test_controller_rechecks_authoritative_delegation_limits(tmp_path, restriction):
    engine, run, _ = paused_parent_with_two_children(tmp_path)
    task = run.task("task_experiment")
    from resagent2_contracts import Attempt
    attempt = Attempt(number=1, status=AttemptStatus.RUNNING, started_at=datetime.now(UTC))
    task.attempts = [attempt]
    request = engine.scheduler._module_request(run, task, 1, parent_session_id=None)
    # The Port cannot grant itself a capability by claiming broader request metadata.
    forged = request.model_copy(update={"permissions": request.permissions.model_copy(update={"request_work": True})})
    if restriction == "capability":
        engine.scheduler.bindings["executor"] = ModuleBinding(
            AgentOwner.EXECUTOR, Port(lambda r, n: complete(r)), can_delegate=False)
    else:
        run.request.execution_limits.max_work_depth = 2
    result = delegate(forged, "coding", "Nested task")
    with pytest.raises(ValueError, match="delegation"):
        engine._receive_task_control(run, task, attempt, forged, result)
    assert len(run.work_requests) == 2


def test_root_completion_yields_to_new_input_before_freezing_final_report(tmp_path):
    holder = {}
    def handler(request, turn):
        result = conclude(request)
        if turn == 0:
            holder["engine"].control_inbox.submit(request.run_id, "input", text="Review the latest constraint")
            return result.model_copy(update={"report": "Outdated provisional report"})
        assert [entry.text for entry in request.upstream_inputs] == ["Review the latest constraint"]
        result.artifacts[0].content = json.dumps({"verdict": "inconclusive", "statement": "Revised conclusion"})
        return result.model_copy(update={"report": "Current original report"})
    root = Port(handler)
    deferred = []
    root.defer_completion = lambda sid, reason: deferred.append(sid)
    engine = holder["engine"] = controller(tmp_path, root)
    run = engine.create_run("run_defer_completion", research_request(goal="Root requirement"))
    assert run.status == RunStatus.COMPLETED, (run.terminal_error, run.completion_violations)
    assert deferred == [run.scientific_session.id]
    assert len(run.task("task_root").attempts) == 1
    attempt = run.task("task_root").attempts[0]
    assert attempt.report == "Current original report"
    from urllib.parse import urlparse
    from urllib.request import url2pathname
    from pathlib import Path
    assert Path(url2pathname(urlparse(run.artifacts[attempt.report_artifact_id].uri).path)).read_text() == "Current original report"
    assert run.final_opinion.statement == "Revised conclusion"
    assert run.pending_module_results == {} and run.pending_registrations == {}


def test_root_receiving_recovers_after_report_binding_without_reinvoking_agent(tmp_path, monkeypatch):
    root = Port(lambda r, n: conclude(r))
    engine = controller(tmp_path, root)
    original = engine.scheduler.artifact_registry.register_final_report
    def crash(*args, **kwargs):
        raise ProcessCrash()
    monkeypatch.setattr(engine.scheduler.artifact_registry, "register_final_report", crash)
    with pytest.raises(ProcessCrash):
        engine.create_run("run_root_receiving", research_request(goal="Recover original result"))
    saved = engine.scheduler.store.load("run_root_receiving")
    assert saved.task("task_root").attempts[0].report_artifact_id is not None
    assert saved.pending_module_results
    monkeypatch.setattr(engine.scheduler.artifact_registry, "register_final_report", original)
    run = engine.run_until_stable(saved.run_id)
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(root.requests) == 1
    assert len(run.task("task_root").attempts) == 1
    assert run.pending_module_results == {} and run.pending_registrations == {}


def test_failed_nested_owner_cancels_its_child_subtree_and_exact_questions(tmp_path):
    def root_handler(request, turn):
        if turn == 0:
            return delegate(request, "coding", "Coding task")
        result = conclude(request)
        result.artifacts[0].content = json.dumps({"verdict": "inconclusive", "statement": "Work failed",
            "limitations": ["Coding failed and the unfinished experiment was cancelled"]})
        return result
    def coding_handler(request, turn):
        if turn == 0:
            return delegate(request, "experiment", "Experiment task")
        return AgentResult(status=ModuleStatus.FAILED, report="Unable to continue",
            session=session(request, SessionStatus.FAILED), error=ModuleError(
                code=ErrorCode.TOOL_FAILED, message="Owner stopped", retryable=False))
    def experiment_handler(request, turn):
        ref = session(request)
        return AgentResult(status=ModuleStatus.NEEDS_USER_INPUT, report="Need a value", session=ref,
            control=ControlSignal(action="ask_user", source_session_id=ref.id, source_turn=0,
                source_call_id="ask", question=PendingQuestion(id="question_nested", run_id=request.run_id,
                    task_id=request.task_id, attempt_number=1, session_id=ref.id,
                    recipient_session_id=request.upstream_session_id, text="Which value?",
                    requested_fields=["value"], created_at=datetime.now(UTC))))
    root, coding, experiment = Port(root_handler), Port(coding_handler), Port(experiment_handler)
    engine = controller(tmp_path, root, coding, experiment)
    run = engine.create_run("run_failed_owner", research_request(goal="Report the actual result"))
    assert run.status == RunStatus.COMPLETED, (run.terminal_error, run.completion_violations)
    nested = run.work("work_experiment")
    assert nested.status == WorkRequestStatus.CANCELLED and nested.result_delivered
    assert run.task("task_experiment").status == TaskStatus.CANCELLED
    assert run.child_questions == {} and run.pending_question is None
    assert len(experiment.requests) == 1
    assert all(work.status not in {WorkRequestStatus.REQUESTED, WorkRequestStatus.COMPILING,
        WorkRequestStatus.EXECUTING, WorkRequestStatus.STABLE} for work in run.work_requests)


def test_abandon_replay_after_saved_control_does_not_cancel_twice(tmp_path, monkeypatch):
    holder = {}
    def root_handler(request, turn):
        if turn == 0:
            return delegate(request, "coding", "original")
        if turn == 1:
            return AgentResult(status=ModuleStatus.ABANDON_WORK, report="End old work", session=session(request),
                control=ControlSignal(action="abandon_work", source_session_id=session(request).id,
                    source_turn=1, source_call_id="abandon_call",
                    abandon_work_id=request.task_context["active_work_request_id"], reason="Reassess"))
        result = conclude(request)
        result.artifacts[0].content = json.dumps({"verdict": "inconclusive", "statement": "done",
            "limitations": ["Cancelled work supplies no complete experimental evidence"]})
        return result
    def child(request, turn):
        holder["engine"].control_inbox.submit(request.run_id, "input", text="Reassess the work")
        return AgentResult(status=ModuleStatus.YIELDED, report="Input arrived", session=session(request))
    root_port = Port(root_handler)
    engine = holder["engine"] = controller(tmp_path, root_port, Port(child))
    original = engine.scheduler._save
    crashed = False
    def save(run):
        nonlocal crashed
        original(run)
        if not crashed and "task:task_root:attempt:1" in run.pending_module_results and any(
                "abandon_call" in key for key in run.received_controls):
            crashed = True
            raise ProcessCrash()
    monkeypatch.setattr(engine.scheduler, "_save", save)
    with pytest.raises(ProcessCrash):
        engine.create_run("run_abandon_replay", research_request(goal="Work"))
    run = engine.run_until_stable("run_abandon_replay")
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert run.task("task_coding").status == TaskStatus.CANCELLED
    assert len(root_port.requests) == 3
    assert sum(notice["type"] == "work_interrupted" for notice in run.runtime_notices) == 1


@pytest.mark.parametrize("reception_failure", [False, True])
def test_saved_root_failure_remains_terminal_after_recovery(tmp_path, monkeypatch, reception_failure):
    def root_handler(request, turn):
        if reception_failure:
            return AgentResult(status=ModuleStatus.COMPLETED, report="Foreign session",
                session=session(request, SessionStatus.COMPLETED).model_copy(update={"id": "session_foreign"}))
        return AgentResult(status=ModuleStatus.FAILED, report="Known failure", session=session(request, SessionStatus.FAILED),
            error=ModuleError(code=ErrorCode.TOOL_FAILED, message="Known failure", retryable=False))
    root_port = Port(root_handler)
    engine = controller(tmp_path, root_port)
    original = engine.scheduler._save
    crashed = False
    def save(run):
        nonlocal crashed
        original(run)
        if not crashed and "task:task_root:attempt:1" in run.pending_module_results:
            crashed = True
            raise ProcessCrash()
    monkeypatch.setattr(engine.scheduler, "_save", save)
    with pytest.raises(ProcessCrash):
        engine.create_run("run_failure_replay", research_request(goal="Work"))
    run = engine.run_until_stable("run_failure_replay")
    assert run.status == RunStatus.FAILED
    assert run.terminal_error.code == (ErrorCode.CONTRACT_ERROR if reception_failure else ErrorCode.TOOL_FAILED)
    assert len(root_port.requests) == 1


def test_received_control_cannot_reuse_native_identity_for_a_different_target(tmp_path):
    engine, run, request = paused_parent_with_two_children(tmp_path)
    task = run.task(request.task_id)
    attempt = task.attempts[-1]
    signal = update_signal(request)
    result = AgentResult(status=ModuleStatus.UPDATE_WORK, report="update", session=session(request), control=signal)
    key = f"session:{signal.source_session_id}:turn:{signal.source_turn}:call:{signal.source_call_id}"
    run.received_controls[key] = "work_different"
    before = run.model_dump_json()
    with pytest.raises(ValueError, match="different target"):
        engine._receive_task_control(run, task, attempt, request, result)
    assert run.model_dump_json() == before
