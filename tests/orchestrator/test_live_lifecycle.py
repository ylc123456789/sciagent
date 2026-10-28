"""Behavioral acceptance for hierarchy, ordinary input and explicit interruption."""
from dataclasses import replace
from sciagent_executor import default_execution_registry
from e2e.scheduler_fixtures import executor_bindings
from datetime import UTC, datetime
import json
from pydantic import BaseModel
import pytest

from e2e.native_fixtures import tool_turns
from sciagent_contracts import (AgentOwner, AgentPermissions, AgentResult, AttemptStatus,
    ModuleStatus, ResearchRequest, RunBudget, RunPermissions, RunStatus, SessionStatus,
    TaskProposal, TaskStatus, WorkflowAgentKind,
    WorkflowAgentRegistry, WorkflowProposal, WorkRequestStatus, task_session_id)
from sciagent_runtime import (AgentLoop, AgentDefinition, AskUserTool, FinishTool,
    AllowListPermissionPolicy, CompletionDecision, InMemorySessionStore, ScriptedLLMClient)
from sciagent_runtime.cancellation import current_cancellation
from sciagent_runtime.controls import RunControlInbox
from sciagent_components.materials import request_task_context
from sciagent_runtime.models import ContextSection
from sciagent_orchestrator import (ResearchController, WorkflowScheduler, ModuleBinding,
    InMemoryRunStore, DeterministicWorkflowCompiler, DeterministicWorkInterpreter)
from sciagent_scientific import ScientificAgent
from sciagent_capabilities.delegation import UpdateWorkInput, UpdateWorkTool, WorkAnswer

def action(tool, **arguments):
    return {"tool": tool, "arguments": arguments}

def scientific_finish():
    return action("finish", report="Observed actual retained results and interruption facts",
        artifacts=[{"kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json",
            "summary": "Conclusion", "content": {"verdict": "inconclusive", "statement": "No scientific claim",
            "limitations": ["Interrupted work does not supply complete experimental evidence"]}}])

def work():
    return action("request_work", assessment={"statement": "Need execution"}, work_request={
        "objective": "Implement assigned task",
        "tasks": [{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "implement", "instruction": "Do the assigned implementation. Preserve original input exactly"}]})

def active_work_id(context):
    authority = json.loads(context.runtime_context.split("\n", 1)[1])
    return authority["child_work"]["active_work_request_id"]

class LiveParent(ScriptedLLMClient):
    def next_tool_call(self, context, *args, **kwargs):
        if self._turns:
            for call in self._turns[0].tool_calls:
                if call.name == "update_work":
                    arguments = json.loads(call.arguments)
                    if arguments.get("work_request_id") == "active":
                        arguments["work_request_id"] = active_work_id(context)
                        call.arguments = json.dumps(arguments)
        return super().next_tool_call(context, *args, **kwargs)

class Check:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None, report="Task completed")

class MarkInput(BaseModel):
    value: str

class MarkTool:
    """Write one observable marker."""
    name = "mark"
    input_model = MarkInput
    def __init__(self, effects): self.effects = effects
    def execute(self, state, arguments):
        from sciagent_runtime import ToolObservation
        self.effects.append(arguments.value)
        return ToolObservation(summary="Marker written", value=arguments.value)

class Child:
    def __init__(self, client, effects):
        self.requests = []
        self.store = InMemorySessionStore()
        self.loop = AgentLoop(store=self.store)
        tools = (MarkTool(effects), AskUserTool(), FinishTool())
        self.definition = AgentDefinition(name="coding", owner=AgentOwner.EXECUTOR,
            system_prompt="Follow upstream requirements", tools=tools, llm_client=client,
            context_builder=lambda request, state, limit: [ContextSection(name="task", content=json.dumps(request_task_context(request))),
                ContextSection(name="material_index", content='{"artifacts":[]}')],
            permission_policy=AllowListPermissionPolicy({t.name for t in tools}), completion_check=Check())
    def invoke(self, request):
        self.requests.append(request)
        definition = replace(self.definition,
            execution_ref=request.execution_definition.ref,
            execution_config_hash=request.execution_definition.config_hash)
        return self.loop.run(definition, request, session_id=task_session_id(request.run_id, request.task_id, request.attempt_number))
    def terminate_session(self, session_id, reason): self.loop.terminate_session(session_id, reason)

def build(tmp_path, root_client, child):
    scheduler = WorkflowScheduler(bindings=executor_bindings({"coding": ModuleBinding(owner=AgentOwner.EXECUTOR, port=child)}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    scientific = ScientificAgent(root_client, store=InMemorySessionStore())
    return ResearchController(scientific_port=scientific, scheduler=scheduler,
        registry=default_execution_registry(),
        compiler=DeterministicWorkflowCompiler(WorkflowProposal(work_request_id="work_fixture", tasks=[TaskProposal(execution_ref="coding@1",
            id="task_live", work_request_id="work_fixture", workflow_agent_kind="executor",
            instruction="Do the assigned implementation. Preserve original input exactly")]), patch=None),
        interpreter=DeterministicWorkInterpreter(), control_inbox=RunControlInbox(tmp_path / "controls"))

def request():
    return ResearchRequest(goal="Complete assigned work", permissions=RunPermissions(),
        budget=RunBudget(max_llm_calls=30, timeout_seconds=60))

def test_ordinary_input_pairs_unstarted_batch_then_parent_clarifies_same_child(tmp_path):
    effects = []
    inbox = RunControlInbox(tmp_path / "controls")
    class ChildClient(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            if not self.contexts:
                inbox.submit("run_live", "input", "New user requirement")
            return super().next_tool_call(*args, **kwargs)
    from sciagent_runtime.models import ToolCallTurn
    batch = ToolCallTurn(tool_calls=[{"id": "call_"+value, "name": "mark", "arguments": json.dumps({"value":value})}
                                   for value in ("first", "second")])
    child_client = ChildClient([batch, *tool_turns([action("finish", report="Done")])])
    parent_client = LiveParent(tool_turns([work(), action("update_work", work_request_id="active", inputs=[{"task_ids": ["task_live"], "text": "Apply the new requirement"}]), scientific_finish()]))
    child = Child(child_client, effects)
    controller = build(tmp_path, parent_client, child)
    run = controller.create_run("run_live", request())
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert effects == []
    accepted = child.store.load(task_session_id(run.run_id, "task_live", 1)).tool_turns[0]
    assert len(accepted.tool_results) == 2
    assert all(json.loads(result)["value"]["execution_status"] == "not_executed" for result in accepted.tool_results.values())
    assert len(child.requests) == 2 and child.requests[1].parent_session_id == task_session_id(child.requests[0].run_id, child.requests[0].task_id, 1)
    assert len(run.task("task_live").attempts) == 1
    assert child.requests[1].upstream_inputs[0].source == "parent"
    assert "New user requirement" in parent_client.contexts[1].task_context
    assert "Apply the new requirement" in child_client.contexts[1].task_context
    assert "Preserve original input exactly" in child_client.contexts[0].task_context
    assert "shared_constraints" not in child_client.contexts[0].task_context
    assert not any(ref.kind in {"question", "answer"} for ref in run.artifacts.values())

def test_explicit_interrupt_invalidates_late_child_reply_and_terminates_work(tmp_path):
    effects = []
    inbox = RunControlInbox(tmp_path / "controls")
    class ChildClient(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            inbox.submit("run_stop", "interrupt", "Stop old work and reassess")
            current_cancellation().cancel("Explicit test interruption")
            return super().next_tool_call(*args, **kwargs)
    child_client = ChildClient(tool_turns([action("mark", value="late side effect")]))
    parent_client = LiveParent(tool_turns([work(), scientific_finish()]))
    child = Child(child_client, effects)
    controller = build(tmp_path, parent_client, child)
    paused = controller.create_run("run_stop", request())
    assert paused.status == RunStatus.PAUSED and effects == []
    old = paused.work(paused.task("task_live").work_request_id)
    assert old.status == WorkRequestStatus.CANCELLED and old.outcome.status == WorkRequestStatus.CANCELLED
    task = paused.task("task_live")
    assert task.status == TaskStatus.CANCELLED and task.attempts[0].status == AttemptStatus.CANCELLED
    assert child.store.load(task_session_id(child.requests[0].run_id, child.requests[0].task_id, 1)).status == SessionStatus.CANCELLED
    assert paused.runtime_notices[-1]["stop_confirmed"]
    resumed = controller.resume_run("run_stop")
    assert resumed.status == RunStatus.COMPLETED, resumed.terminal_error
    assert len(child.requests) == 1
    assert resumed.scientific_session.id == paused.scientific_session.id
    assert "Stop old work and reassess" in parent_client.contexts[1].task_context
    assert any(turn.runtime_notice and "work_interrupted" in turn.runtime_notice for turn in controller.scientific_port.store.load(resumed.scientific_session.id).tool_turns)

def test_parent_can_read_interrupted_command_through_same_task_artifact_handoff(tmp_path):
    from dataclasses import replace
    from sciagent_components import RegisteredArtifactReader, preserve_execution_record
    from sciagent_orchestrator.handoffs import read_json
    from sciagent_runtime.cancellation import OperationCancelled

    facts = {"operation": "process", "command": "python train.py", "pid": 321,
             "exit_code": -15, "stdout_path": "partial.stdout", "stderr_path": "partial.stderr",
             "duration_seconds": 1.25, "files_retained": True}
    partial = tmp_path / "partial.checkpoint"

    class InterruptedCommand:
        name = "run_shell"
        input_model = MarkInput
        def execute(self, state, arguments):
            partial.write_bytes(b"actual incomplete checkpoint")
            controller.control_inbox.submit("run_stop_record", "interrupt", "Stop and inspect what happened")
            current_cancellation().cancel("User interrupted command")
            raise OperationCancelled("User interrupted command", details=facts)

    class CommandChild(Child):
        def __init__(self, client):
            super().__init__(client, [])
            self.definition = replace(self.definition, tools=(InterruptedCommand(), FinishTool()),
                permission_policy=AllowListPermissionPolicy({"run_shell", "finish"}))
        def invoke(self, request):
            return preserve_execution_record(super().invoke(request), self.store)

    child = CommandChild(ScriptedLLMClient(tool_turns([action("run_shell", value="run command")])))
    parent_client = LiveParent(tool_turns([work(), scientific_finish()]))
    controller = build(tmp_path, parent_client, child)
    paused = controller.create_run("run_stop_record", request())
    assert paused.status == RunStatus.PAUSED
    task = paused.task("task_live")
    attempt = task.attempts[0]
    assert task.status == TaskStatus.CANCELLED
    record = next(paused.artifacts[key] for key in attempt.artifact_ids
                  if paused.artifacts[key].kind == "execution_record")
    assert read_json(record) == {"results": [{**facts, "execution_status": "interrupted", "stop_confirmed": True}]}
    report = paused.artifacts[attempt.report_artifact_id]
    assert record.id in report.metadata["output_artifact_ids"]
    assert record.id in paused.scientific_artifact_ids
    notice = next(item for item in paused.runtime_notices if item["type"] == "work_interrupted")
    assert notice["stop_confirmed"] is True and record.id in notice["artifact_ids"]
    assert "not validated deliveries" in notice["file_retention_note"]
    assert not any(item.kind == "code_change" for item in paused.artifacts.values())
    assert partial.read_bytes() == b"actual incomplete checkpoint"
    child_state = child.store.load(task_session_id(child.requests[0].run_id, child.requests[0].task_id, 1))
    assert len(child_state.tool_turns[0].tool_results) == 1

    # The parent uses only public refs; details do not enter its requirements.
    resumed = controller.resume_run(paused.run_id)
    assert resumed.status == RunStatus.COMPLETED, resumed.terminal_error
    context = parent_client.contexts[1]
    assert record.id in context.material_index
    assert facts["command"] not in context.task_context
    directory = json.loads(context.material_index.split("\n", 1)[1])
    registered = next(node for node in directory["children"] if node["name"] == "registered_materials")
    works = next(node for node in registered["children"] if node["name"] == "work_requests")
    indexed_work = next(node for node in works["children"] if node["name"] == task.work_request_id)
    indexed_task = next(node for node in indexed_work["children"] if node["name"] == task.id)
    assert any(item["location"]["artifact_id"] == record.id for item in indexed_task["children"])
    reader = RegisteredArtifactReader(list(resumed.artifacts.values()), run_id=resumed.run_id)
    assert json.loads(reader.read_text(record.id)["content"])["results"][0]["command"] == "python train.py"
    assert len(child.requests) == 1
    assert sum(item.kind == "execution_record" for item in resumed.artifacts.values()) == 1
    assert partial.read_bytes() == b"actual incomplete checkpoint"


def test_user_forward_requires_received_source_and_preserves_source_reference(tmp_path):
    from sciagent_contracts import AgentRequest, BusinessInput, TaskBudget
    event = BusinessInput(event_id="original", text="Exact original words", source="user", created_at=datetime(2020,1,1,tzinfo=UTC))
    req = AgentRequest(run_id="run_forward", agent=AgentOwner.SCIENTIFIC, work_request_id="work_root",
        task_id="task_root", attempt_number=1, instruction="Do work", budget=TaskBudget(max_llm_calls=10,timeout_seconds=60),
        permissions=AgentPermissions(request_work=True), upstream_inputs=[event], input_ids=[event.event_id],
        task_context={"active_work_request_id": "work_child", "child_tasks": [{"task_id": "task_live", "status": "pending"}]})
    tool = UpdateWorkTool(req)
    state = type("State", (), {"session_id": "session_parent"})()
    with pytest.raises(PermissionError, match="reference"):
        tool.execute(state, UpdateWorkInput(work_request_id="work_child", inputs=[{
            "task_ids": ["task_live"], "source_input_id": "invented"}]))
    observation = tool.execute(state, UpdateWorkInput(work_request_id="work_child", inputs=[{
        "task_ids": ["task_live"], "source_input_id": "original"}]))
    assert observation.work_update["inputs"][0]["source_input_id"] == "original"
    assert event.text == "Exact original words" and event.created_at == datetime(2020,1,1,tzinfo=UTC)


def test_interrupt_while_waiting_keeps_prior_command_record_readable_to_parent(tmp_path):
    from dataclasses import replace
    from e2e.native_fixtures import ParentQuestionClient
    from sciagent_components import preserve_execution_record
    from sciagent_runtime import ToolObservation
    from sciagent_orchestrator.handoffs import read_json
    from sciagent_contracts import VerificationResult

    facts = dict(command="python prepare.py", exit_code=0, timed_out=False,
        stdout_path="prepared.stdout", stderr_path="prepared.stderr", duration_seconds=0.1)
    class Command:
        name = "run_shell"
        input_model = MarkInput
        def execute(self, state, arguments):
            return ToolObservation(summary="Preparation completed", value=facts)
    class CommandChild(Child):
        def __init__(self, client):
            super().__init__(client, [])
            self.definition = replace(self.definition, tools=(Command(), AskUserTool(), FinishTool()),
                permission_policy=AllowListPermissionPolicy({"run_shell", "ask_user", "finish"}))
        def invoke(self, request):
            return preserve_execution_record(super().invoke(request), self.store)

    parent = ParentQuestionClient(tool_turns([work(), scientific_finish()]))
    child = CommandChild(ScriptedLLMClient(tool_turns([
        action("run_shell", value="prepare"),
        action("ask_user", text="Choose how to continue", requested_fields=["choice"]),
    ])))
    controller = build(tmp_path, parent, child)
    waiting = controller.create_run("run_wait_record", request())
    assert waiting.pending_question and waiting.task("task_live").status == TaskStatus.NEEDS_USER_INPUT
    prior = next(ref for ref in waiting.artifacts.values() if ref.kind == "execution_record")
    assert read_json(prior) == {"results": [VerificationResult(**facts).model_dump(mode="json")]}
    controller.control_inbox.submit(waiting.run_id, "interrupt", "Stop while awaiting a decision")
    paused = controller.resume_run(waiting.run_id)
    assert paused.status == RunStatus.PAUSED and paused.pending_question is None
    assert prior.id in paused.scientific_artifact_ids
    assert paused.artifacts[prior.id] == prior
    assert child.store.load(task_session_id(child.requests[0].run_id, child.requests[0].task_id, 1)).status == SessionStatus.CANCELLED
    resumed = controller.resume_run(paused.run_id)
    assert resumed.status == RunStatus.COMPLETED, resumed.terminal_error
    assert prior.id in parent.contexts[-1].material_index
    assert len(child.requests) == 1


def test_input_between_last_agent_return_and_owner_receiving_is_not_swallowed(tmp_path):
    client = ScriptedLLMClient(tool_turns([scientific_finish(), scientific_finish()]))
    child = Child(ScriptedLLMClient([]), [])
    controller = build(tmp_path, client, child)
    original = controller.scientific_port.invoke
    def invoke(request):
        result = original(request)
        if len(client.contexts) == 1:
            controller.control_inbox.submit("run_final_input", "input", "Clarify the final conclusion")
        return result
    controller.scientific_port.invoke = invoke
    run = controller.create_run("run_final_input", request())
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(client.contexts) == 2
    assert "Clarify the final conclusion" in client.contexts[1].task_context
    assert sum(ref.kind == "scientific_opinion" for ref in run.artifacts.values()) == 2
    assert len(run.consumed_control_ids) == 1


def test_escalation_reference_forwards_real_user_answer_and_child_history_stays_part_three(tmp_path):
    from e2e.native_fixtures import ParentQuestionClient, tool_turn
    from sciagent_contracts import UserAnswer
    class Parent(ParentQuestionClient):
        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            questions = json.loads(context.pending_notifications or "[]")
            if questions and questions[0]["id"] in self._asked_child_ids:
                data = json.loads(context.task_context.split("\n",1)[1])
                user = [event["record"] for event in data["business_dialogue"] if event["type"] == "answer"][-1]
                self.contexts.append(context)
                self.schemas.append(schemas)
                self.histories.append([t.model_copy(deep=True) for t in turns])
                return tool_turn("update_work", {"work_request_id": active_work_id(context), "answers": [{
                    "task_id": "task_live", "question_id": questions[0]["id"], "source_answer_id": user["question_id"]}]})
            return super().next_tool_call(context, schemas, turns, max_input_tokens=max_input_tokens)
    child_client = ScriptedLLMClient(tool_turns([action("ask_user", text="Choose a method", requested_fields=["method"]), action("finish", report="Done")]))
    parent_client = Parent(tool_turns([work(), scientific_finish()]))
    child = Child(child_client, [])
    controller = build(tmp_path, parent_client, child)
    paused = controller.create_run("run_escalate", request())
    child_question = next(iter(paused.child_questions.values()))
    assert paused.pending_question.related_question_id == child_question.id
    assert child_question.text not in [e["record"] for e in json.loads(parent_client.contexts[1].task_context.split("\n",1)[1])["business_dialogue"] if e["type"] == "answer"]
    run = controller.answer_question(paused.run_id, UserAnswer(question_id=paused.pending_question.id,
        values={"method": "custom free text, exactly"}, answered_at=datetime.now(UTC)))
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    child_answer = child.requests[1].user_answers[0]
    assert child_answer.source == "user" and child_answer.original_question_id == paused.pending_question.id
    assert child_answer.values == {"method": "custom free text, exactly"}
    history = controller.scientific_port.store.load(run.scientific_session.id).tool_turns
    assert any(t.runtime_notice and '"type": "child_question"' in t.runtime_notice for t in history)
    assert not run.child_questions


def test_interrupt_waiting_escalation_cancels_only_related_upper_question(tmp_path):
    from e2e.native_fixtures import ParentQuestionClient
    parent = ParentQuestionClient(tool_turns([work(), scientific_finish()]))
    child = Child(ScriptedLLMClient(tool_turns([action("ask_user", text="Need user choice", requested_fields=["choice"])])), [])
    controller = build(tmp_path, parent, child)
    waiting = controller.create_run("run_escalated_stop", request())
    assert waiting.pending_question.related_question_id
    controller.control_inbox.submit(waiting.run_id, "interrupt")
    stopped = controller.resume_run(waiting.run_id)
    assert stopped.status == RunStatus.PAUSED and stopped.pending_question is None and not stopped.child_questions
    assert stopped.work(stopped.task("task_live").work_request_id).status == WorkRequestStatus.CANCELLED
    assert controller.scientific_port.store.load(stopped.scientific_session.id).pending_control is None
    assert stopped.user_wait_seconds >= 0
    resumed = controller.resume_run(stopped.run_id)
    assert resumed.status == RunStatus.COMPLETED, resumed.terminal_error


def test_parent_clarifies_all_affected_tasks_before_releasing_child_work(tmp_path):
    from dataclasses import replace

    effects = []
    inbox = RunControlInbox(tmp_path / "controls")
    class FirstClient(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            if not self.contexts:
                inbox.submit("run_multi_input", "input", "New requirement applies to both Tasks")
            return super().next_tool_call(*args, **kwargs)
    first = FirstClient(tool_turns([action("mark", value="initial"), action("finish", report="First done")]))
    second = ScriptedLLMClient(tool_turns([action("finish", report="Second done")]))
    class Children(Child):
        def invoke(self, request):
            if self.requests:
                published = controller.scheduler.store.load("run_multi_input")
                assert published.task_inputs["task_first"][0].text == "Apply requirement to first Task"
                assert published.task_inputs["task_second"][0].text == "Apply requirement to second Task"
            self.requests.append(request)
            definition = replace(self.definition, llm_client=first if request.task_id == "task_first" else second,
                execution_ref=request.execution_definition.ref,
                execution_config_hash=request.execution_definition.config_hash)
            return self.loop.run(definition, request, session_id=task_session_id(request.run_id, request.task_id, request.attempt_number))
    child = Children(first, effects)
    work_call = action("request_work", assessment={"statement": "Need both Tasks"}, work_request={
        "objective": "Implement assigned task",
        "tasks": [{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "first", "instruction": "Do first assigned implementation"},
                  {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "second", "instruction": "Do second assigned implementation"}]})
    parent_client = LiveParent(tool_turns([work_call,
        action("update_work", work_request_id="active", inputs=[
            {"task_ids": ["task_first"], "text": "Apply requirement to first Task"},
            {"task_ids": ["task_second"], "text": "Apply requirement to second Task"}]),
        scientific_finish()]))
    controller = build(tmp_path, parent_client, child)
    controller.compiler = DeterministicWorkflowCompiler(WorkflowProposal(work_request_id="work_fixture", tasks=[
        TaskProposal(execution_ref="coding@1", id="task_" + key, work_request_id="work_fixture", workflow_agent_kind="executor",
            instruction="Do " + key + " assigned implementation") for key in ("first", "second")]), patch=None)
    run = controller.create_run("run_multi_input", request())
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert [r.task_id for r in child.requests] == ["task_first", "task_first", "task_second"]
    assert child.requests[1].parent_session_id == task_session_id(child.requests[0].run_id, "task_first", 1)
    assert [len(run.task("task_" + key).attempts) for key in ("first", "second")] == [1, 1]
    assert "Apply requirement to first Task" in first.contexts[1].task_context
    assert "Apply requirement to second Task" in second.contexts[0].task_context
    assert len(parent_client.contexts) == 3
    assert effects == []


def test_parent_corrects_invalid_update_before_answering_exact_child_question(tmp_path):
    from e2e.native_fixtures import tool_turn
    class ParentClient(ScriptedLLMClient):
        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            index = len(self.contexts)
            if index in {1, 2}:
                waiting = controller.scheduler.store.load("run_answer_pairing")
                assert waiting.child_questions and waiting.task("task_live").status == TaskStatus.NEEDS_USER_INPUT
                assert len(child.requests) == 1
                self.contexts.append(context)
                self.schemas.append(schemas)
                self.histories.append([t.model_copy(deep=True) for t in turns])
                if index == 1:
                    return tool_turn("update_work", {})
                question = next(iter(waiting.child_questions.values()))
                return tool_turn("update_work", {"work_request_id": active_work_id(context), "answers": [{
                    "task_id": "task_live", "question_id": question.id, "values": {"choice": "Parent decision"}}]})
            return super().next_tool_call(context, schemas, turns, max_input_tokens=max_input_tokens)
    parent = ParentClient(tool_turns([work(), scientific_finish()]))
    child_client = ScriptedLLMClient(tool_turns([action("ask_user", text="Need parent choice", requested_fields=["choice"]),
                                              action("finish", report="Done")]))
    child = Child(child_client, [])
    controller = build(tmp_path, parent, child)
    run = controller.create_run("run_answer_pairing", request())
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(child.requests) == 2 and child.requests[1].parent_session_id == task_session_id(run.run_id, "task_live", 1)
    assert len(run.task("task_live").attempts) == 1
    assert child.requests[1].user_answers[0].values == {"choice": "Parent decision"}
    assert not run.child_questions


@pytest.mark.parametrize("arrival_stage", ["validation", "rendering"])
def test_input_during_completion_receiving_reconsiders_frozen_opinion(tmp_path, arrival_stage):
    first = scientific_finish()
    second = scientific_finish()
    changed = second["arguments"]["artifacts"][0]["content"]
    changed["statement"] = "Conclusion revised for the new upstream requirement"
    second["arguments"]["artifacts"][0]["content"] = changed
    client = ScriptedLLMClient(tool_turns([first, second]))
    controller = build(tmp_path, client, Child(ScriptedLLMClient([]), []))
    gate, renderer = controller.gate, controller.report_renderer
    submitted = False
    def publish_once():
        nonlocal submitted
        if not submitted:
            controller.control_inbox.submit("run_receiving_input", "input", "Revise the final conclusion")
            submitted = True
    class Gate:
        def validate(self, run, result, refs):
            if arrival_stage == "validation":
                publish_once()
            return gate.validate(run, result, refs)
    class Renderer:
        def render(self, data):
            if arrival_stage == "rendering":
                publish_once()
            return renderer.render(data)
    controller.gate, controller.report_renderer = Gate(), Renderer()
    run = controller.create_run("run_receiving_input", request())
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(client.contexts) == 2
    assert "Revise the final conclusion" in client.contexts[1].task_context
    assert run.final_opinion.statement == changed["statement"]
    opinions = [ref for ref in run.artifacts.values() if ref.kind == "scientific_opinion"]
    assert len(opinions) == 2 and len({ref.sha256 for ref in opinions}) == 2
    # The already frozen first proposal remains readable for reconsideration;
    # the second proposal alone supplies the accepted final conclusion.
    assert opinions[0].id in client.contexts[1].material_index
    assert sum(ref.kind == "final_report" for ref in run.artifacts.values()) == 1
    assert not controller.control_inbox.pending(run.run_id, run.consumed_control_ids)
    assert not run.pending_registrations


def test_input_publication_cannot_cross_terminal_commit_using_stale_ui_snapshot(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from sciagent_cli.main import _submit_control

    controller = build(tmp_path, ScriptedLLMClient(tool_turns([scientific_finish()])),
                       Child(ScriptedLLMClient([]), []))
    final_registration = controller.scheduler.artifact_registry.register_final_report
    registering, allow_finish, submitting = Event(), Event(), Event()
    def register(*args, **kwargs):
        registering.set()
        assert allow_finish.wait(5)
        return final_registration(*args, **kwargs)
    controller.scheduler.artifact_registry.register_final_report = register
    def publish(stale):
        submitting.set()
        return _submit_control(tmp_path, stale, kind="input", text="Too late for this Run",
                               store=controller.scheduler.store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(controller.create_run, "run_commit_fence", request())
        assert registering.wait(5)
        stale = controller.scheduler.store.load("run_commit_fence")
        assert stale.status == RunStatus.RUNNING
        producer = pool.submit(publish, stale)
        try:
            assert submitting.wait(2)
            assert not producer.done()
        finally:
            allow_finish.set()
        completed = owner.result(timeout=5)
        with pytest.raises(ValueError, match="running or paused"):
            producer.result(timeout=5)
    assert completed.status == RunStatus.COMPLETED
    assert not controller.control_inbox.pending(completed.run_id)


@pytest.mark.parametrize("values", [{"choice": "new direction"}, {}])
def test_answer_values_require_a_specific_question(values):
    with pytest.raises(ValueError, match="question_id"):
        WorkAnswer(task_id="task_live", values=values)


def test_controller_rejects_orphan_answer_without_mutation_or_resume(tmp_path):
    from sciagent_contracts import ControlSignal
    child = Child(ScriptedLLMClient(tool_turns([
        action("ask_user", text="Need parent choice", requested_fields=["choice"]),
    ])), [])
    parent = ScriptedLLMClient(tool_turns([work(), action("ask_user",
        assessment={"statement": "Need user choice"}, text="Choose direction", requested_fields=["choice"])]))
    controller = build(tmp_path, parent, child)
    waiting = controller.create_run("run_orphan_reply", request())
    root = waiting.task("task_root")
    bound = controller.scheduler._module_request(waiting, root, 1, parent_session_id=root.attempts[0].session.id)
    before = waiting.model_dump()
    signal = ControlSignal(action="update_work", source_session_id=waiting.scientific_session.id,
        source_turn=99, source_call_id="call_orphan", work_update={
            "work_request_id": waiting.active_child(root.id, 1).id, "provided_input_ids": bound.input_ids,
            "inputs": [], "answers": [{"task_id": "task_live", "question_id": "unrelated_question",
            "values": {"choice": "must not disappear"}}]})
    with pytest.raises(ValueError, match="live direct child business question"):
        controller._apply_work_update(waiting, bound, signal)
    assert waiting.model_dump() == before
    assert len(child.requests) == 1


def test_native_parent_corrects_rejected_orphan_values_then_delivers_answer(tmp_path):
    from e2e.native_fixtures import tool_turn
    class ParentClient(ScriptedLLMClient):
        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            index = len(self.contexts)
            if index in {1, 2}:
                waiting = controller.scheduler.store.load("run_native_orphan")
                assert len(child.requests) == 1
                assert waiting.child_questions and not waiting.task_inputs
                self.contexts.append(context)
                self.schemas.append(schemas)
                self.histories.append([turn.model_copy(deep=True) for turn in turns])
                if index == 1:
                    return tool_turn("update_work", {"work_request_id": active_work_id(context), "answers": [{"task_id": "task_live", "values": {"choice": "original words"}}]})
                assert "question_id" in json.dumps([turn.model_dump(mode="json") for turn in turns])
                question = next(iter(waiting.child_questions.values()))
                return tool_turn("update_work", {"work_request_id": active_work_id(context), "answers": [{"task_id": "task_live", "question_id": question.id,
                    "values": {"choice": "original words"}}]})
            return super().next_tool_call(context, schemas, turns, max_input_tokens=max_input_tokens)
    parent = ParentClient(tool_turns([work(), scientific_finish()]))
    child = Child(ScriptedLLMClient(tool_turns([
        action("ask_user", text="Need parent choice", requested_fields=["choice"]),
        action("finish", report="Used delivered answer"),
    ])), [])
    controller = build(tmp_path, parent, child)
    run = controller.create_run("run_native_orphan", request())
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(child.requests) == 2 and len(run.task("task_live").attempts) == 1
    assert child.requests[1].parent_session_id == task_session_id(child.requests[0].run_id, child.requests[0].task_id, 1)
    assert child.requests[1].user_answers[0].values == {"choice": "original words"}
    assert child.requests[1].user_answers[0].source == "parent"


class ReceivingCrash(BaseException):
    pass


@pytest.mark.parametrize("saved_in_run", [False, True])
def test_old_terminal_checkpoint_cannot_complete_over_new_input(tmp_path, monkeypatch, saved_in_run):
    client = LiveParent(tool_turns([scientific_finish(), scientific_finish()]))
    engine = build(tmp_path, client, Child(ScriptedLLMClient([]), []))
    crashed = False
    if saved_in_run:
        original = engine.scheduler._save
        def save(run):
            nonlocal crashed
            original(run)
            if not crashed and "task:task_root:attempt:1" in run.pending_module_results:
                crashed = True
                raise ReceivingCrash()
        monkeypatch.setattr(engine.scheduler, "_save", save)
    else:
        original = engine.scientific_port.invoke
        def invoke(request):
            nonlocal crashed
            result = original(request)
            if not crashed:
                crashed = True
                raise ReceivingCrash()
            return result
        monkeypatch.setattr(engine.scientific_port, "invoke", invoke)
    with pytest.raises(ReceivingCrash):
        engine.create_run("run_old_terminal", request())
    engine.control_inbox.submit("run_old_terminal", "input", text="New requirement after the old conclusion")
    run = engine.resume_run("run_old_terminal")
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(client.contexts) == 2
    assert "New requirement after the old conclusion" in client.contexts[1].task_context
    assert len(run.task("task_root").attempts) == 1
    assert run.scientific_session.id == task_session_id(run.run_id, "task_root", 1)


def test_saved_pending_control_is_redelivered_without_model_or_tool_replay(tmp_path, monkeypatch):
    client = LiveParent(tool_turns([work(), action("ask_user", assessment={"statement": "Need a choice"},
        text="Which direction?", requested_fields=["direction"])]))
    child = Child(ScriptedLLMClient(tool_turns([action("finish", report="Done")])), [])
    engine = build(tmp_path, client, child)
    original = engine.scientific_port.invoke
    calls = 0
    def invoke(request):
        nonlocal calls
        calls += 1
        result = original(request)
        if calls == 2:
            raise ReceivingCrash()
        return result
    monkeypatch.setattr(engine.scientific_port, "invoke", invoke)
    with pytest.raises(ReceivingCrash):
        engine.create_run("run_pending_control", request())
    saved = engine.scientific_port.store.load(task_session_id("run_pending_control", "task_root", 1))
    pending = saved.pending_control.model_copy(deep=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint redelivery must not invoke model or tools")
    monkeypatch.setattr(engine.scientific_port, "invoke", forbidden)
    run = engine.resume_run("run_pending_control")
    assert run.status == RunStatus.PAUSED, run.terminal_error
    assert run.pending_question == pending.question
    assert len(client.contexts) == 2 and calls == 2 and len(child.requests) == 1
    assert engine.scientific_port.store.load(saved.session_id) == saved


def test_failed_final_report_receiving_does_not_retry_after_recovery(tmp_path, monkeypatch):
    client = LiveParent(tool_turns([scientific_finish()]))
    engine = build(tmp_path, client, Child(ScriptedLLMClient([]), []))
    original = engine.scheduler._save
    crashed = False
    def save(run):
        nonlocal crashed
        original(run)
        if not crashed and "task:task_root:attempt:1" in run.pending_module_results:
            crashed = True
            raise ReceivingCrash()
    monkeypatch.setattr(engine.scheduler, "_save", save)
    with pytest.raises(ReceivingCrash):
        engine.create_run("run_final_receiving_error", request())
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise OSError("Final report disk unavailable")
    monkeypatch.setattr(engine.scheduler.artifact_registry, "register_final_report", fail)
    run = engine.resume_run("run_final_receiving_error")
    assert run.status == RunStatus.FAILED and run.terminal_error.code.value == "artifact_missing"
    assert calls == 1 and len(client.contexts) == 1


@pytest.mark.parametrize("crash_on_update", [False, True])
def test_unreceived_work_control_retains_its_actual_input_boundary(tmp_path, monkeypatch, crash_on_update):
    actions = [work(), action("update_work", work_request_id="active", inputs=[{
        "task_ids": ["task_live"], "text": "Apply current requirement"}]), scientific_finish()]
    if crash_on_update:
        actions.insert(1, action("update_work", work_request_id="active", inputs=[{
            "task_ids": ["task_live"], "text": "Obsolete update must not reach the child"}]))
    if not crash_on_update:
        revised = work()
        revised["arguments"]["work_request"]["tasks"][0]["instruction"] = "Apply current requirement from the latest input"
        actions[1] = revised
    client = LiveParent(tool_turns(actions))
    engine_holder = {}
    class ChildClient(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            if crash_on_update and not self.contexts and not engine_holder.get("injected"):
                engine_holder["injected"] = True
                engine_holder["engine"].control_inbox.submit("run_saved_work_control", "input", text="First requirement")
            return super().next_tool_call(*args, **kwargs)
    child_client = ChildClient(tool_turns([action("finish", report="Done"), action("finish", report="Done")]))
    child = Child(child_client, [])
    engine = engine_holder["engine"] = build(tmp_path, client, child)
    original = engine.scientific_port.invoke
    calls = 0
    def invoke(request):
        nonlocal calls
        calls += 1
        result = original(request)
        if calls == (2 if crash_on_update else 1):
            raise ReceivingCrash()
        return result
    monkeypatch.setattr(engine.scientific_port, "invoke", invoke)
    with pytest.raises(ReceivingCrash):
        engine.create_run("run_saved_work_control", request())
    engine.control_inbox.submit("run_saved_work_control", "input", text="Latest requirement while stopped")
    run = engine.resume_run("run_saved_work_control")
    assert run.status == RunStatus.COMPLETED, (run.terminal_error, [(t.id, t.status, t.attempts[-1].error if t.attempts else None) for t in run.workflow.tasks], [(w.id, w.status) for w in run.work_requests], run.attention_task_ids, len(client.contexts))
    assert len(client.contexts) == (4 if crash_on_update else 3)
    assert "Latest requirement while stopped" in client.contexts[-2].task_context
    if crash_on_update:
        assert "Apply current requirement" in child.requests[-1].upstream_inputs[-1].text
    else:
        assert "Apply current requirement" in run.work(child.requests[-1].work_request_id).request.tasks[0].instruction
    assert all("Obsolete update" not in item.text for req in child.requests for item in req.upstream_inputs)
    assert len(run.task("task_root").attempts) == len(run.task("task_live").attempts) == 1
    rejected = [notice for notice in run.runtime_notices if notice["type"] == "work_control_not_applied"]
    assert len(rejected) == 1 and rejected[0]["applied"] is False


def test_bound_root_report_is_retained_when_new_input_defers_recovered_completion(tmp_path, monkeypatch):
    old = scientific_finish()
    old["arguments"]["report"] = "Old report before the new requirement"
    new = scientific_finish()
    new["arguments"]["report"] = "New report after processing the latest requirement"
    new["arguments"]["artifacts"][0]["content"] = {"verdict": "inconclusive", "statement": "Updated conclusion"}
    client = LiveParent(tool_turns([old, new]))
    engine = build(tmp_path, client, Child(ScriptedLLMClient([]), []))
    original = engine.scheduler.artifact_registry.register_final_report
    crashed = False
    def register(*args, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise ReceivingCrash()
        return original(*args, **kwargs)
    monkeypatch.setattr(engine.scheduler.artifact_registry, "register_final_report", register)
    with pytest.raises(ReceivingCrash):
        engine.create_run("run_bound_old_report", request())
    saved = engine.scheduler.store.load("run_bound_old_report")
    old_report = saved.task("task_root").attempts[0].report_artifact_id
    assert old_report is not None
    engine.control_inbox.submit("run_bound_old_report", "input", text="New requirement after report binding")
    run = engine.resume_run("run_bound_old_report")
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert len(client.contexts) == 2
    assert old_report in run.artifacts
    attempt = run.task("task_root").attempts[0]
    assert attempt.report_artifact_id != old_report
    assert attempt.report == "New report after processing the latest requirement"


def test_paused_checkpoint_preserves_newly_registered_scientific_refs(tmp_path, monkeypatch):
    from sciagent_components import WebPage
    from sciagent_orchestrator.artifacts import ScientificArtifactRegistration
    from sciagent_runtime import ToolCallTurn, NativeToolCall
    class Fetcher:
        calls = 0
        def fetch(self, url):
            self.calls += 1
            return WebPage(source_url=url, final_url=url, title="Source", text="Original source",
                content_type="text/html", parser="fixture", fetched_at="2026-10-21T00:00:00+00:00")
    class Client(LiveParent):
        registration = None
        def next_tool_call(self, context, schemas, turns, **kwargs):
            if len(self.contexts) == 1:
                ref = self.registration.list_artifacts(run_id="run_paused_refs",
                    session_id=task_session_id("run_paused_refs", "task_root", 1))[0]
                arguments = json.loads(self._turns[0].tool_calls[0].arguments)
                arguments["work_request"]["tasks"][0]["input_artifact_ids"] = [ref.id]
                self._turns[0].tool_calls[0].arguments = json.dumps(arguments)
            return super().next_tool_call(context, schemas, turns, **kwargs)
    client = Client(tool_turns([action("web_fetch", url="https://example.test/source"), work(), scientific_finish()]))
    child = Child(ScriptedLLMClient(tool_turns([action("finish", report="Done")])), [])
    engine = build(tmp_path, client, child)
    fetcher = Fetcher()
    registration = ScientificArtifactRegistration(engine.scheduler.artifact_registry, engine.scheduler.store)
    client.registration = registration
    engine.scientific_port.registration_port = registration
    engine.scientific_port.web_page_fetcher = fetcher
    original = engine.scientific_port.loop.run
    crashed = False
    def loop_run(*args, **kwargs):
        nonlocal crashed
        result = original(*args, **kwargs)
        if not crashed:
            crashed = True
            raise ReceivingCrash()
        return result
    monkeypatch.setattr(engine.scientific_port.loop, "run", loop_run)
    with pytest.raises(ReceivingCrash):
        engine.create_run("run_paused_refs", request())
    checkpoint = engine.scientific_port.read_checkpoint_result(run_id="run_paused_refs", task_id="task_root", attempt_number=1)
    refs = [item for item in checkpoint.result.artifacts if item.kind == "web_page"]
    assert len(refs) == 1
    # Restore normal execution only after the saved handoff has been received.
    run = engine.resume_run("run_paused_refs")
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert fetcher.calls == 1 and len(client.contexts) == 3 and len(child.requests) == 1
    assert refs[0].id in run.task("task_root").attempts[0].artifact_ids
    assert refs[0].id in checkpoint.result.control.work_request.request.tasks[0].input_artifact_ids


def test_unreceived_abandon_is_not_applied_across_new_input(tmp_path, monkeypatch):
    holder = {}
    class Client(LiveParent):
        def next_tool_call(self, context, *args, **kwargs):
            if self._turns and self._turns[0].tool_calls[0].name == "abandon_work":
                call = self._turns[0].tool_calls[0]
                arguments = json.loads(call.arguments)
                arguments["work_request_id"] = active_work_id(context)
                call.arguments = json.dumps(arguments)
            return super().next_tool_call(context, *args, **kwargs)
    class ChildClient(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            if not self.contexts and not holder.get("injected"):
                holder["injected"] = True
                holder["engine"].control_inbox.submit("run_old_abandon", "input", text="First new requirement")
            return super().next_tool_call(*args, **kwargs)
    client = Client(tool_turns([work(), action("abandon_work", work_request_id="active", reason="Old decision"),
        action("update_work", work_request_id="active", inputs=[{"task_ids": ["task_live"], "text": "Continue with latest requirement"}]),
        scientific_finish()]))
    child = Child(ChildClient(tool_turns([action("finish", report="Done"), action("finish", report="Done")])), [])
    engine = holder["engine"] = build(tmp_path, client, child)
    original = engine.scientific_port.invoke
    calls = 0
    def invoke(request):
        nonlocal calls
        calls += 1
        result = original(request)
        if calls == 2:
            raise ReceivingCrash()
        return result
    monkeypatch.setattr(engine.scientific_port, "invoke", invoke)
    with pytest.raises(ReceivingCrash):
        engine.create_run("run_old_abandon", request())
    engine.control_inbox.submit("run_old_abandon", "input", text="Keep the child work and continue")
    run = engine.resume_run("run_old_abandon")
    assert run.status == RunStatus.COMPLETED, run.terminal_error
    assert run.task("task_live").status == TaskStatus.COMPLETED
    assert len(client.contexts) == 4 and len(run.task("task_live").attempts) == 1
    assert child.requests[-1].upstream_inputs[-1].text == "Continue with latest requirement"
    rejected = [notice for notice in run.runtime_notices if notice["type"] == "work_control_not_applied"]
    assert len(rejected) == 1 and rejected[0]["action"] == "abandon_work" and rejected[0]["applied"] is False
    assert not any(notice["type"] == "work_interrupted" for notice in run.runtime_notices)
