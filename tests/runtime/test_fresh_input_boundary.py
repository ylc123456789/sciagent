"""New parent input cannot authorize a stale native operation or batch tail."""
from dataclasses import replace
from datetime import UTC, datetime
import json

import pytest

from e2e.native_fixtures import tool_turns
from resagent2_contracts import AgentPermissions, AgentRequest, RecordedAnswer, TaskBudget
from resagent2_runtime import (
    AgentDefinition, AgentLoop, CompletionDecision, ContextSection, FinishTool,
    InMemorySessionStore, PermissionDecision, ScriptedLLMClient, ToolObservation,
)
from resagent2_runtime.lifecycle import model_boundary
from resagent2_runtime.models import RuntimeModel


class Arguments(RuntimeModel):
    text: str


class Write:
    name = "write"
    input_model = Arguments

    def __init__(self, callback=None):
        self.values, self.callback = [], callback

    def execute(self, state, args):
        self.values.append(args.text)
        if self.callback:
            self.callback()
        return ToolObservation(summary="Written", value=args.text)


class Policy:
    def __init__(self, callback=None):
        self.callback = callback

    def check(self, action, state, request):
        if action.tool == "finish":
            return PermissionDecision(outcome="allow")
        if self.callback:
            self.callback()
        if state.pending_action and request.resume_answers:
            answer = request.resume_answers[0]
            return PermissionDecision(outcome="allow", approval_id=state.pending_action.action_id,
                                      user_comment=answer.comment)
        return PermissionDecision(outcome="ask", reason="Confirm this write", context={"scope": "test"})


class Complete:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None, report="Done")


def request(**changes):
    return AgentRequest(agent="executor", run_id="run_fresh", work_request_id="work_fresh",
        task_id="task_fresh", attempt_number=1, instruction="Perform the requested write",
        budget=TaskBudget(max_llm_calls=5, timeout_seconds=30),
        permissions=AgentPermissions(execute_commands=True), **changes)


def definition(write, policy):
    return AgentDefinition(name="executor", owner="executor", system_prompt="Follow the task",
        tools=(write, FinishTool()), llm_client=ScriptedLLMClient(tool_turns([
            {"tool": "write", "arguments": {"text": "original"}},
            {"tool": "finish", "arguments": {"report": "Reconsidered"}},
        ])), context_builder=lambda req, *_: [ContextSection(name="task", content=req.instruction)],
        permission_policy=policy, completion_check=Complete())


def paused_case():
    write, store = Write(), InMemorySessionStore()
    spec = definition(write, Policy())
    loop = AgentLoop(store=store)
    result = loop.run(spec, request(), session_id="session_fresh")
    assert result.status == "needs_user_input"
    q = result.control.question
    answer = RecordedAnswer(question_id=q.id, question_text=q.text, requested_fields=q.requested_fields,
        options=q.options, values={"approve": "yes"}, comment="Only the earlier operation was approved",
        answered_at=datetime.now(UTC), action=q.action, run_id=q.run_id, task_id=q.task_id,
        attempt_number=q.attempt_number, session_id=q.session_id)
    resume = request(parent_session_id=result.session.id, resume_answers=[answer], control_ack=result.control.origin)
    return loop, spec, write, store, result, resume


def test_new_direct_input_settles_old_approval_once_and_preserves_comment():
    loop, spec, write, store, paused, resume = paused_case()
    refreshed = resume.model_copy(update={"input_ids": ["input_new"], "instruction": "Do not write; reassess"})
    result = loop.run(spec, refreshed, session_id="unused")
    assert result.status == "completed"
    assert not write.values
    saved = store.load(paused.session.id)
    old = json.loads(saved.tool_turns[0].tool_results[paused.control.source_call_id])
    assert old["value"]["execution_status"] == "not_executed"
    assert old["user_comment"] == "Only the earlier operation was approved"
    assert saved.pending_control is None and saved.pending_action is None
    assert len(saved.control_history) == 1


def test_owner_input_during_approval_recheck_cannot_start_old_operation():
    loop, spec, write, store, paused, resume = paused_case()
    pending = []
    spec = replace(spec, permission_policy=Policy(lambda: pending.append(True)))
    with model_boundary(lambda: bool(pending)):
        result = loop.run(spec, resume, session_id="unused")
    assert result.status == "yielded" and not write.values
    saved = store.load(paused.session.id)
    receipt = json.loads(saved.tool_turns[0].tool_results[paused.control.source_call_id])
    assert receipt["value"]["execution_status"] == "not_executed"
    assert receipt["user_comment"] == "Only the earlier operation was approved"


def test_input_during_permission_check_prevents_even_creating_old_approval():
    pending, store, write = [], InMemorySessionStore(), Write()
    spec = definition(write, Policy(lambda: pending.append(True)))
    with model_boundary(lambda: bool(pending)):
        result = AgentLoop(store=store).run(spec, request(), session_id="session_fresh")
    assert result.status == "yielded" and not write.values
    saved = store.load("session_fresh")
    assert not saved.control_history and saved.pending_action is None
    assert all(json.loads(receipt)["value"]["execution_status"] == "not_executed"
               for receipt in saved.tool_turns[0].tool_results.values())


@pytest.mark.parametrize("mode", ["approved", "outdated", "interrupted"])
def test_resumed_operation_keeps_original_call_receipt_with_new_runtime_notice(mode):
    from resagent2_runtime.cancellation import CancellationToken, bind_cancellation, check_cancelled

    loop, spec, write, store, paused, resume = paused_case()
    resume = resume.model_copy(update={"runtime_notices": [{
        "event_id": "notice_child", "summary": "A child result is now available",
    }]})
    acknowledgements = []
    spec = replace(spec, acknowledge_registration=lambda **values: acknowledgements.append(values))
    if mode == "outdated":
        resume = resume.model_copy(update={"input_ids": ["input_new"]})
    token = CancellationToken()
    if mode == "interrupted":
        def interrupt():
            token.cancel("Stop the approved write")
            check_cancelled()
        write.callback = interrupt
    with bind_cancellation(token):
        result = loop.run(spec, resume, session_id="unused")
    assert result.status == ("interrupted" if mode == "interrupted" else "completed")
    assert write.values == ([] if mode == "outdated" else ["original"])
    saved = store.load(paused.session.id)
    receipt = json.loads(saved.tool_turns[paused.control.source_turn].tool_results[paused.control.source_call_id])
    assert receipt["user_comment"] == "Only the earlier operation was approved"
    if mode == "outdated":
        assert receipt["value"]["execution_status"] == "not_executed"
    elif mode == "interrupted":
        assert receipt["value"]["execution_status"] == "interrupted"
    else:
        assert receipt["ok"] is True
        assert acknowledgements[0]["operation_key"] == (
            f"session:{paused.session.id}:turn:{paused.control.source_turn}:call:{paused.control.source_call_id}"
        )
    assert all(len(turn.tool_results) == len(turn.tool_calls) for turn in saved.tool_turns)
    assert sum("notice_child" in (turn.runtime_notice or "") for turn in saved.tool_turns) == 1
    # Reload through the strict model, as a fresh process would do.
    type(saved).model_validate_json(saved.model_dump_json())
