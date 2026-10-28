"""Production loop behavior for the four-part context and delayed native controls."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from resagent2_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ArtifactRef, ModuleStatus, RecordedAnswer, TaskBudget,
)
from resagent2_runtime import (
    AgentDefinition, AgentLoop, AllowListPermissionPolicy, AskUserTool, CompletionDecision,
    ContextBudgetExceeded, ContextComposer, ContextSection, FinishTool, InMemorySessionStore,
    PermissionDecision, ScriptedLLMClient,
    ToolObservation,
)
from resagent2_runtime.models import NativeToolCall, ToolCallTurn
from resagent2_runtime.tool_calling import native_input, native_tool_schemas, tool_messages
from tool_fixtures import WriteValueTool


def _turn(index, tool, arguments):
    return ToolCallTurn(tool_calls=[NativeToolCall(
        id=f"call_{index}", name=tool, arguments=json.dumps(arguments),
    )])


class _Complete:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None, report="accepted" if candidate else "")


def _request(parent=None, **updates):
    return AgentRequest(work_request_id="work_test",
        run_id="run_context", task_id="task_context", attempt_number=1, agent=AgentOwner.EXECUTOR,
        instruction="Initial task", parent_session_id=parent, permissions=AgentPermissions(),
        budget=TaskBudget(max_llm_calls=10, timeout_seconds=60), **updates,
    )


def _context(request, state, limit):
    return [
        ContextSection(name="task", content=json.dumps({
            "instruction": request.instruction,
            "user_answers": [answer.model_dump(mode="json") for answer in request.user_answers],
        })),
        ContextSection(name="material_index", content='{"artifacts":[]}'),
    ]


def _definition(turns, policy=None, completion=None):
    tools = (WriteValueTool(), AskUserTool(), FinishTool())
    return AgentDefinition(
        name="executor", owner=AgentOwner.EXECUTOR, system_prompt="Fixed identity and rules",
        tools=tools, llm_client=ScriptedLLMClient(turns), context_builder=_context,
        permission_policy=policy or AllowListPermissionPolicy({tool.name for tool in tools}),
        completion_check=completion or _Complete(),
    )


def _answer(question, value="answer"):
    return RecordedAnswer(
        question_id=question.id, question_text=question.text, requested_fields=question.requested_fields,
        options=question.options, values={question.requested_fields[0]: value}, answered_at=datetime.now(UTC),
        run_id=question.run_id, session_id=question.session_id,
        task_id=question.task_id, attempt_number=question.attempt_number, action=question.action,
    )


def test_business_question_accepted_once_without_control_artifact_and_answer_in_task():
    store = InMemorySessionStore()
    definition = _definition([
        _turn(1, "ask_user", {"text": "Which dataset?", "requested_fields": ["dataset"],
                              "options": {"dataset": ["CIFAR-10", "CIFAR-100"]}}),
        _turn(2, "finish", {"report": "done"}),
    ])
    loop = AgentLoop(store=store)
    first = loop.run(definition, _request(), session_id="session_context")
    assert first.status == ModuleStatus.NEEDS_USER_INPUT
    assert first.artifacts == []
    state = store.load(first.session.id)
    receipt = json.loads(state.tool_turns[0].tool_results["call_1"])
    assert receipt["value"]["question_id"] == first.control.question.id
    answer = _answer(first.control.question, "CIFAR-10")
    result = loop.run(definition, _request(
        parent=state.session_id, user_answers=[answer], resume_answers=[answer], control_ack=first.control.origin,
    ), session_id="unused")
    assert result.status == ModuleStatus.COMPLETED
    state = store.load(first.session.id)
    assert len(state.tool_turns[0].tool_results) == 1
    assert "CIFAR-10" in definition.llm_client.contexts[1].task_context
    assert "Which dataset?" in definition.llm_client.contexts[1].task_context
    assert state.pending_control is None


def test_unanswered_control_reuses_identity_without_another_model_call():
    store = InMemorySessionStore()
    definition = _definition([_turn(1, "ask_user", {"text": "Which?", "requested_fields": ["choice"]})])
    loop = AgentLoop(store=store)
    first = loop.run(definition, _request(), session_id="session_context")
    second = loop.run(definition, _request(parent=first.session.id), session_id="unused")
    assert second.control == first.control
    assert second.llm_calls == 0
    assert len(definition.llm_client.contexts) == 1


def test_answer_must_match_the_original_question_not_only_its_id():
    store = InMemorySessionStore()
    definition = _definition([_turn(1, "ask_user", {"text": "Which?", "requested_fields": ["choice"]})])
    loop = AgentLoop(store=store)
    first = loop.run(definition, _request(), session_id="session_context")
    answer = _answer(first.control.question).model_copy(update={"question_text": "Different question"})
    result = loop.run(definition, _request(parent=first.session.id, resume_answers=[answer]), session_id="unused")
    assert result.status == ModuleStatus.FAILED
    assert "does not match the pending question" in result.report
    assert len(definition.llm_client.contexts) == 1


@pytest.mark.parametrize("change", ["prompt", "tools"])
def test_fixed_contract_change_rejects_resume_without_consuming_answer(change):
    store = InMemorySessionStore()
    definition = _definition([_turn(1, "ask_user", {"text": "Which?", "requested_fields": ["choice"]})])
    loop = AgentLoop(store=store)
    first = loop.run(definition, _request(), session_id="session_context")
    before = store.load(first.session.id)
    altered = replace(definition, **({"system_prompt": "Different"} if change == "prompt"
                                    else {"tools": (FinishTool(),)}))
    result = loop.run(altered, _request(parent=first.session.id), session_id="unused")
    assert result.status == ModuleStatus.FAILED
    assert store.load(first.session.id) == before


class _Approval:
    def check(self, action, state, request):
        if action.tool != "write_value":
            return PermissionDecision(outcome="allow")
        if state.pending_action is not None:
            for answer in request.resume_answers:
                if answer.action == state.pending_action:
                    return PermissionDecision(
                        outcome="allow" if answer.values["approve"] == "yes" else "deny",
                        reason="approved" if answer.values["approve"] == "yes" else "declined",
                        approval_id=state.pending_action.action_id,
                    )
        return PermissionDecision(outcome="ask", reason="Confirm this exact operation", context={"cwd": "/workspace"})


@pytest.mark.parametrize("approved", [True, False])
def test_operation_waits_for_answer_then_original_call_gets_only_final_receipt(approved):
    store = InMemorySessionStore()
    batch = ToolCallTurn(tool_calls=[
        NativeToolCall(id="write", name="write_value", arguments='{"key":"result","value":42}'),
        NativeToolCall(id="tail", name="write_value", arguments='{"key":"tail","value":99}'),
    ])
    definition = _definition([batch, _turn(2, "finish", {"report": "done"})], policy=_Approval())
    loop = AgentLoop(store=store)
    first = loop.run(definition, _request(), session_id="session_context")
    paused = store.load(first.session.id)
    assert "write" not in paused.tool_turns[0].tool_results
    assert json.loads(paused.tool_turns[0].tool_results["tail"])["value"]["execution_status"] == "not_executed"
    assert "result" not in paused.memory
    answer = _answer(first.control.question, "yes" if approved else "no")
    result = loop.run(definition, _request(parent=first.session.id, resume_answers=[answer],
                                           control_ack=first.control.origin), session_id="unused")
    state = store.load(first.session.id)
    assert result.status == ModuleStatus.COMPLETED
    assert set(state.tool_turns[0].tool_results) == {"write", "tail"}
    assert (state.memory.get("result") == 42) == approved
    assert "tail" not in state.memory
    assert "approve" not in definition.llm_client.contexts[1].task_context
    tool_messages(state.tool_turns)


def test_approved_tool_registers_material_and_registration_ack_failure_is_not_a_tool_failure(tmp_path):
    class RegisterTool(WriteValueTool):
        def execute(self, state, arguments):
            ref = ArtifactRef(
                id="artifact_registered", kind="web_page", producer="scientific",
                run_id=state.run_id, session_id=state.session_id, uri=(tmp_path / "page.md").as_uri(),
                sha256="a" * 64, media_type="text/markdown", summary="Registered source",
            )
            return ToolObservation(summary="Material saved", artifacts=[ref], value={"artifact_id": ref.id})

    def acknowledge(**kwargs):
        raise OSError("registration journal unavailable")

    store = InMemorySessionStore()
    definition = _definition([_turn(1, "write_value", {"key": "result", "value": 42})], policy=_Approval())
    definition = replace(definition, tools=(RegisterTool(), AskUserTool(), FinishTool()),
                         acknowledge_registration=acknowledge)
    loop = AgentLoop(store=store)
    first = loop.run(definition, _request(), session_id="session_context")
    answer = _answer(first.control.question, "yes")
    with pytest.raises(OSError, match="registration journal unavailable"):
        loop.run(definition, _request(parent=first.session.id, resume_answers=[answer]), session_id="unused")
    state = store.load(first.session.id)
    receipt = json.loads(state.tool_turns[0].tool_results["call_1"])
    assert receipt["ok"] is True
    assert receipt["summary"] == "Material saved"
    assert state.authorized_artifact_ids == ["artifact_registered"]
    assert len(state.tool_turns[0].tool_results) == 1


def test_rejected_finish_receipt_contains_local_rejection_not_proposal():
    class Check:
        def evaluate(self, state, candidate):
            if candidate.report == "bad":
                return CompletionDecision(complete=False, report="Required evidence is missing")
            return CompletionDecision(complete=True, report=candidate.report)
    store = InMemorySessionStore()
    definition = _definition([_turn(1, "finish", {"report": "bad"}),
                              _turn(2, "finish", {"report": "good"})], completion=Check())
    result = AgentLoop(store=store).run(definition, _request(), session_id="session_context")
    state = store.load(result.session.id)
    receipt = json.loads(state.tool_turns[0].tool_results["call_1"])
    assert receipt["ok"] is False
    assert receipt["summary"] == "Required evidence is missing"
    assert len(state.tool_turns[0].tool_results) == 1
    assert "runtime_feedback" not in definition.llm_client.contexts[1].included_sections


def test_wire_order_has_identity_task_protocol_and_complete_directory_without_duplicate_body():
    context = ContextComposer().compose("identity", [
        ContextSection(name="task", content="complete requirements"),
        ContextSection(name="material_index", content="all artifact IDs"),
    ], max_tokens=10000)
    turn = _turn(1, "read", {})
    turn.tool_results["call_1"] = "exact source fragment"
    context = context.model_copy(update={"history_summary": "earlier observations"})
    wire = native_input(context, [], [turn])
    assert [message["role"] for message in wire["messages"]] == ["system", "user", "assistant", "assistant", "tool", "user"]
    assert "identity" in wire["messages"][0]["content"]
    assert "complete requirements" in wire["messages"][1]["content"]
    assert "all artifact IDs" in wire["messages"][-1]["content"]
    assert json.dumps(wire).count("exact source fragment") == 1


def test_complete_directory_never_silently_truncated_or_selected():
    sections = [ContextSection(name="material_index", content="artifact_1 " * 1000)]
    with pytest.raises(ContextBudgetExceeded):
        ContextComposer().compose("identity", sections, max_tokens=100)
