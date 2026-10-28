"""Model/tool interruption retains facts and prevents later native dispatch."""
import json
from dataclasses import replace
from io import BytesIO

import pytest
from pydantic import BaseModel

from sciagent_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ModuleStatus, SessionStatus, TaskBudget,
)
from sciagent_runtime import (
    AgentDefinition, AgentLoop, AllowListPermissionPolicy, AskUserTool, CompletionDecision,
    ContextSection, FinishTool, InMemorySessionStore, OpenAICompatibleClient,
    PermissionDecision, ToolObservation,
)
from sciagent_runtime.cancellation import (
    CancellationToken, OperationCancelled, bind_cancellation,
)
from sciagent_runtime.lifecycle import model_boundary
from sciagent_runtime.models import RuntimeModel
from sciagent_runtime.tool_calling import tool_messages


class WriteInput(RuntimeModel):
    key: str


class WriteTool:
    name = "write"
    input_model = WriteInput

    def __init__(self, callback=None):
        self.callback = callback
        self.executed = []

    def execute(self, state, arguments: BaseModel):
        self.executed.append(arguments.key)
        if self.callback:
            self.callback()
        return ToolObservation(summary="Written", value={"key": arguments.key},
                               memory_updates={arguments.key: True})


class AcceptFinish:
    def __init__(self, callback=None):
        self.callback = callback

    def evaluate(self, state, candidate):
        if candidate is not None and self.callback:
            self.callback()
        return CompletionDecision(complete=candidate is not None, report="completed")


def request(parent=None):
    return AgentRequest(work_request_id="work_test",
        run_id="run_interrupt", task_id="task_interrupt", attempt_number=1,
        agent=AgentOwner.EXECUTOR, instruction="Preserve completed work and respond to new input",
        parent_session_id=parent, budget=TaskBudget(max_llm_calls=8, timeout_seconds=10),
        permissions=AgentPermissions(execute_commands=True),
    )


def call(identifier, name="write", arguments=None):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments if arguments is not None else {"key": identifier}),
    }}


def reply(calls):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "reasoning_content": "private reasoning", "tool_calls": calls,
    }}]}


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOP_INTERRUPT_KEY", "dummy")
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.test/v1", api_key_env="LOOP_INTERRUPT_KEY",
        trace_dir=tmp_path / "trace", trace_level="full",
    )
    write = WriteTool()
    definition = AgentDefinition(
        name="executor", owner=AgentOwner.EXECUTOR, system_prompt="Complete the task using tools.",
        llm_client=client, tools=(write, FinishTool()),
        permission_policy=AllowListPermissionPolicy({"write", "finish"}),
        context_builder=lambda request, state, limit: [
            ContextSection(name="task", content=request.instruction),
            ContextSection(name="material_index", content="[]"),
        ], completion_check=AcceptFinish(),
    )
    store = InMemorySessionStore()
    sent = []

    def install(responses):
        queue = iter(responses)

        def send(http_request, **kwargs):
            sent.append(json.loads(http_request.content))
            result = next(queue)
            if callable(result):
                result = result()
            return BytesIO(json.dumps(result).encode())

        monkeypatch.setattr("sciagent_runtime.llm.send_request", send)

    return definition, store, write, sent, install


def test_late_native_model_reply_is_not_adopted_or_dispatched(fixture):
    definition, store, write, sent, install = fixture
    token = CancellationToken()

    def late_response():
        token.cancel("user interrupted model")
        return reply([call("late_write")])

    install([late_response, reply([call("finish", "finish", {"report": "done"})])])
    loop = AgentLoop(store=store)
    with bind_cancellation(token):
        result = loop.run(definition, request(), session_id="session_interrupt")
    state = store.load("session_interrupt")
    assert result.status == ModuleStatus.INTERRUPTED
    assert result.llm_calls == state.llm_calls_used == 1
    assert not write.executed
    assert not any(turn.tool_calls for turn in state.tool_turns)
    assert any(turn.runtime_notice for turn in state.tool_turns)

    with bind_cancellation(CancellationToken()):
        resumed = loop.run(definition, request("session_interrupt"), session_id="session_interrupt")
    assert resumed.status == ModuleStatus.COMPLETED
    assert store.load("session_interrupt").llm_calls_used == 2
    assert "late_write" not in json.dumps(sent[1])


def test_interrupt_after_completed_tool_keeps_receipt_and_skips_batch_tail(fixture):
    definition, store, write, sent, install = fixture
    token = CancellationToken()
    write.callback = lambda: token.cancel("pause after write")
    install([reply([call("done"), call("unstarted")])])
    with bind_cancellation(token):
        result = AgentLoop(store=store).run(definition, request(), session_id="session_interrupt")
    assert result.status == ModuleStatus.INTERRUPTED
    state = store.load("session_interrupt")
    assert state.memory["done"]
    assert "unstarted" not in state.memory
    assert write.executed == ["done"]
    turn = next(turn for turn in state.tool_turns if turn.tool_calls)
    assert set(turn.tool_results) == {"done", "unstarted"}
    assert json.loads(turn.tool_results["done"])["ok"]
    assert json.loads(turn.tool_results["unstarted"])["value"]["execution_status"] == "not_executed"
    assert len([m for m in tool_messages(state.tool_turns) if m.get("tool_call_id") == "done"]) == 1


def test_cancelled_tool_receipt_reports_actual_stop_details(fixture):
    definition, store, write, sent, install = fixture

    def interrupted():
        raise OperationCancelled("training interrupted", stopped=True, details={
            "operation": "process", "exit_code": -9, "stdout_path": "logs/train.stdout",
            "files_retained": True,
        })

    write.callback = interrupted
    install([reply([call("training"), call("tail")])])
    result = AgentLoop(store=store).run(definition, request(), session_id="session_interrupt")
    assert result.status == ModuleStatus.INTERRUPTED
    state = store.load("session_interrupt")
    turn = next(turn for turn in state.tool_turns if turn.tool_calls)
    receipt = json.loads(turn.tool_results["training"])
    assert not receipt["ok"]
    assert receipt["value"]["stop_confirmed"] is True
    assert receipt["value"]["exit_code"] == -9
    assert receipt["value"]["stdout_path"] == "logs/train.stdout"
    assert receipt["value"]["files_retained"] is True
    assert set(turn.tool_results) == {"training", "tail"}


def test_ordinary_input_pairs_unstarted_calls_and_resumes_without_replaying_batch(fixture):
    definition, store, write, sent, install = fixture
    queued = []

    def response():
        queued.append("new user instruction")
        return reply([call("first"), call("second")])

    install([response, reply([call("updated")]),
             reply([call("finish", "finish", {"report": "Updated task completed"})])])
    loop = AgentLoop(store=store)
    with model_boundary(lambda: bool(queued)):
        result = loop.run(definition, request(), session_id="session_interrupt")
    assert result.status == ModuleStatus.YIELDED
    assert not write.executed
    assert len(sent) == 1
    state = store.load("session_interrupt")
    turn = next(turn for turn in state.tool_turns if turn.tool_calls)
    assert [item.id for item in turn.tool_calls] == ["first", "second"]
    original_receipts = turn.tool_results.copy()
    assert set(original_receipts) == {"first", "second"}
    assert all(json.loads(value)["value"]["execution_status"] == "not_executed"
               for value in original_receipts.values())
    assert "first" not in state.memory and "second" not in state.memory

    refreshed = request("session_interrupt").model_copy(update={
        "instruction": "Apply the new instruction", "input_ids": ["input_new"],
    })
    resumed = loop.run(definition, refreshed, session_id="unused")
    assert resumed.status == ModuleStatus.COMPLETED
    assert write.executed == ["updated"]
    assert len(sent) == 3
    state = store.load("session_interrupt")
    assert state.tool_turns[0].tool_results == original_receipts
    assert state.provided_input_ids == ["input_new"]
    assert "Apply the new instruction" in json.dumps(sent[1])
    assert state.memory["updated"]


def test_interrupt_during_permission_check_does_not_create_new_question(fixture):
    definition, store, write, sent, install = fixture
    token = CancellationToken()

    class Policy:
        def check(self, action, state, request):
            token.cancel("cancel before approval prompt")
            return PermissionDecision(outcome="ask", reason="Approve exact operation")

    install([reply([call("pending")])])
    with bind_cancellation(token):
        result = AgentLoop(store=store).run(
            replace(definition, permission_policy=Policy()), request(), session_id="session_interrupt",
        )
    assert result.status == ModuleStatus.INTERRUPTED
    state = store.load("session_interrupt")
    assert state.pending_control is None and state.pending_action is None
    assert not write.executed
    turn = next(turn for turn in state.tool_turns if turn.tool_calls)
    assert set(turn.tool_results) == {"pending"}
    assert json.loads(turn.tool_results["pending"])["value"]["execution_status"] == "not_executed"


def test_interrupt_clears_waiting_approval_with_one_not_executed_receipt(fixture):
    definition, store, write, sent, install = fixture

    class Policy:
        def check(self, action, state, request):
            return PermissionDecision(outcome="ask", reason="Approve exact operation")

    definition = replace(definition, permission_policy=Policy())
    install([reply([call("approved_later")])])
    loop = AgentLoop(store=store)
    paused = loop.run(definition, request(), session_id="session_interrupt")
    assert paused.status == ModuleStatus.NEEDS_USER_INPUT
    token = CancellationToken()
    token.cancel("cancel unused approval")
    with bind_cancellation(token):
        cancelled = loop.run(definition, request("session_interrupt"), session_id="session_interrupt")
    assert cancelled.status == ModuleStatus.INTERRUPTED
    state = store.load("session_interrupt")
    assert state.pending_control is None and state.pending_action is None
    assert not write.executed
    assert len(sent) == 1
    turn = next(turn for turn in state.tool_turns if turn.tool_calls)
    assert set(turn.tool_results) == {"approved_later"}
    assert json.loads(turn.tool_results["approved_later"])["value"]["execution_status"] == "not_executed"
    tool_messages(state.tool_turns)


def test_interrupt_at_completion_preserves_finish_receipt_but_does_not_complete(fixture):
    definition, store, write, sent, install = fixture
    token = CancellationToken()
    definition = replace(definition, completion_check=AcceptFinish(lambda: token.cancel("pause at finish")))
    install([reply([call("finish", "finish", {"report": "done"})])])
    with bind_cancellation(token):
        result = AgentLoop(store=store).run(definition, request(), session_id="session_interrupt")
    assert result.status == ModuleStatus.INTERRUPTED
    state = store.load("session_interrupt")
    assert state.status == SessionStatus.PAUSED
    assert state.terminal_result is None
    turn = next(turn for turn in state.tool_turns if turn.tool_calls)
    assert json.loads(turn.tool_results["finish"])["ok"]
    assert set(turn.tool_results) == {"finish"}


def test_failed_stop_confirmation_is_failed_and_never_reported_paused(fixture):
    definition, store, write, sent, install = fixture

    def interrupted():
        raise OperationCancelled("process could not be confirmed stopped", stopped=False,
                                 details={"operation": "process", "pid": 12345})

    write.callback = interrupted
    install([reply([call("unstopped"), call("tail")])])
    result = AgentLoop(store=store).run(definition, request(), session_id="session_interrupt")
    assert result.status == ModuleStatus.FAILED
    assert result.error.details["stop_confirmed"] is False
    assert result.error.details["pid"] == 12345
    state = store.load("session_interrupt")
    assert state.status == SessionStatus.FAILED
    turn = next(turn for turn in state.tool_turns if turn.tool_calls)
    receipt = json.loads(turn.tool_results["unstopped"])
    assert receipt["value"]["stop_confirmed"] is False
    assert receipt["value"]["pid"] == 12345
    assert "has stopped" not in receipt["summary"]
    assert set(turn.tool_results) == {"unstopped", "tail"}


def test_interrupt_waiting_business_question_preserves_accepted_receipt_and_links_notice(fixture):
    definition, store, write, sent, install = fixture
    definition = replace(
        definition, tools=(*definition.tools, AskUserTool()),
        permission_policy=AllowListPermissionPolicy({"write", "finish", "ask_user"}),
    )
    install([reply([call("question", "ask_user", {
        "text": "Which target should this task use?", "requested_fields": ["target"],
    })])])
    loop = AgentLoop(store=store)
    waiting = loop.run(definition, request(), session_id="session_interrupt")
    assert waiting.status == ModuleStatus.NEEDS_USER_INPUT
    question_id = waiting.control.question.id
    before = store.load("session_interrupt").tool_turns[0].tool_results["question"]
    token = CancellationToken()
    token.cancel("interrupt waiting question")
    with bind_cancellation(token):
        result = loop.run(definition, request("session_interrupt"), session_id="session_interrupt")
    assert result.status == ModuleStatus.INTERRUPTED
    state = store.load("session_interrupt")
    assert state.tool_turns[0].tool_results["question"] == before
    assert state.pending_control is None
    assert question_id in " ".join(turn.runtime_notice or "" for turn in state.tool_turns)
    assert len([m for m in tool_messages(state.tool_turns) if m.get("tool_call_id") == "question"]) == 1


def test_interrupt_after_control_acceptance_retains_one_receipt_and_suspends(fixture):
    definition, store, write, sent, install = fixture
    token = CancellationToken()

    class InterruptedAsk(AskUserTool):
        def execute(self, state, arguments):
            observation = super().execute(state, arguments)
            token.cancel("pause after accepted question")
            return observation

    definition = replace(
        definition, tools=(*definition.tools, InterruptedAsk()),
        permission_policy=AllowListPermissionPolicy({"write", "finish", "ask_user"}),
    )
    install([reply([call("question", "ask_user", {
        "text": "Which target should this task use?", "requested_fields": ["target"],
    })])])
    with bind_cancellation(token):
        result = AgentLoop(store=store).run(definition, request(), session_id="session_interrupt")
    assert result.status == ModuleStatus.INTERRUPTED
    state = store.load("session_interrupt")
    assert state.pending_control is None
    assert set(state.tool_turns[0].tool_results) == {"question"}
    receipt = json.loads(state.tool_turns[0].tool_results["question"])
    assert receipt["ok"]
    assert receipt["value"]["status"] == "question_issued"
    assert len([m for m in tool_messages(state.tool_turns) if m.get("tool_call_id") == "question"]) == 1


def test_terminate_crashed_session_preserves_unknown_started_operation_and_unstarted_tail(fixture, tmp_path):
    definition, store, write, sent, install = fixture
    effect = tmp_path / "effect.txt"

    class ProcessStopped(BaseException):
        pass

    def crash_after_effect():
        effect.write_text("The tool already changed this file")
        raise ProcessStopped()

    write.callback = crash_after_effect
    install([reply([call("started"), call("tail")])])
    loop = AgentLoop(store=store)
    with pytest.raises(ProcessStopped):
        loop.run(definition, request(), session_id="session_interrupt")
    checkpoint = store.load("session_interrupt")
    assert checkpoint.tool_turns[0].executing_call_id == "started"
    checkpoint.tool_turns[0].executing_user_comment = "Approved this operation"
    store.save(checkpoint)

    loop.terminate_session("session_interrupt", "The owner ended this work after restart")
    saved = store.load("session_interrupt")
    assert saved.status == SessionStatus.CANCELLED
    assert effect.read_text() == "The tool already changed this file"
    started = json.loads(saved.tool_turns[0].tool_results["started"])
    assert started["value"]["execution_status"] == "unknown"
    assert started["user_comment"] == "Approved this operation"
    tail = json.loads(saved.tool_turns[0].tool_results["tail"])
    assert tail["value"]["execution_status"] == "not_executed"
    assert write.executed == ["started"]
    tool_messages(saved.tool_turns)
    snapshot = saved.model_dump_json()
    loop.terminate_session("session_interrupt", "Repeated cleanup")
    assert store.load("session_interrupt").model_dump_json() == snapshot
