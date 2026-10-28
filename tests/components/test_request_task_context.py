"""Business dialogue is complete while authorization remains operation-local."""
from datetime import UTC,datetime
import pytest
from pydantic import ValidationError
from sciagent_contracts import AgentRequest,AgentPermissions,TaskBudget,RecordedAnswer,ActionSnapshot
from sciagent_components import request_task_context


def request(**kw):
    return AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1, run_id="run_test", agent="scientific", instruction="Original requirement",
        budget=TaskBudget(max_llm_calls=8, timeout_seconds=60), permissions=AgentPermissions(), **kw)


def answer(n):
    return RecordedAnswer(question_id=f"question_{n}", question_text=f"Full question {n}",
        requested_fields=["scope"], options={"scope":["full","small"]}, values={"scope":"full"},
        run_id="run_test", task_id="task_root", attempt_number=1, session_id="session_scientific_run_test", answered_at=datetime.now(UTC))


def test_task_context_preserves_all_business_question_fields_and_initial_facts():
    req=request(parent_session_id="session_scientific_run_test", user_answers=[answer(1),answer(2)], input_ids=["question_1","question_2"],
                task_context={"required_artifacts":["comparison_results"]})
    before=req.model_dump_json()
    value=request_task_context(req)
    assert value["instruction"]=="Original requirement"
    # Internal per-invoke ceilings are not the live Run-wide remaining budget.
    assert "budget" not in value
    assert req.budget.max_llm_calls == 8
    assert value["task_context"]["required_artifacts"]==["comparison_results"]
    assert [item["question_id"] for item in [event["record"] for event in value["business_dialogue"] if event["type"] == "answer"]]==["question_1","question_2"]
    assert [event["record"] for event in value["business_dialogue"] if event["type"] == "answer"][0]["requested_fields"]==["scope"]
    assert [event["record"] for event in value["business_dialogue"] if event["type"] == "answer"][0]["options"]=={"scope":["full","small"]}
    assert req.model_dump_json()==before


def test_resume_answer_is_not_silently_presented_as_a_business_requirement():
    req=request(parent_session_id="session_scientific_run_test",resume_answers=[answer(1)])
    assert [e["record"] for e in request_task_context(req)["business_dialogue"] if e["type"] == "answer"]==[]


def test_business_answers_reject_foreign_scope():
    with pytest.raises(ValidationError):
        request(parent_session_id="session_scientific_run_test",user_answers=[answer(1).model_copy(update={"run_id":"run_foreign"})])


def test_business_dialogue_uses_reception_order_not_untrusted_source_timestamps():
    from datetime import timedelta
    from sciagent_contracts import BusinessInput
    now=datetime.now(UTC)
    old=BusinessInput(event_id="input_old", text="Use random split", source="user", created_at=now)
    correction=BusinessInput(event_id="input_new", text="Withdraw random split; use sequential split", source="user", created_at=now-timedelta(days=1))
    req=request(upstream_inputs=[old,correction], input_ids=[old.event_id,correction.event_id])
    assert [event["event_id"] for event in request_task_context(req)["business_dialogue"]] == ["input_old","input_new"]
    with pytest.raises(ValueError, match="exactly match"):
        request_task_context(req.model_copy(update={"input_ids":[old.event_id]}))
    with pytest.raises(ValueError, match="exactly match"):
        request_task_context(req.model_copy(update={"input_ids":[old.event_id,correction.event_id,"unseen"]}))


def test_shared_dialogue_projection_cannot_mix_operation_approval_into_business_requirements():
    from sciagent_components.materials import ordered_business_dialogue
    approval = answer(1).model_copy(update={"action": ActionSnapshot(
        action_id="action_one", tool="run_shell", arguments={"command": "echo one"},
        run_id="run_test", task_id="task_root", attempt_number=1,
        session_id="session_scientific_run_test")})
    with pytest.raises(ValueError, match="operation approvals are not business dialogue"):
        ordered_business_dialogue(upstream_inputs=[], answers=[approval], input_ids=[approval.question_id])


@pytest.mark.parametrize("order", [["question_1", "question_1"], []])
def test_agent_task_projection_rejects_duplicate_or_missing_reception_identities(order):
    req = request(user_answers=[answer(1)], input_ids=["question_1"])
    with pytest.raises(ValueError, match="exactly match"):
        request_task_context(req.model_copy(update={"input_ids": order}))
