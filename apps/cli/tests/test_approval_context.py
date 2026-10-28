from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

import pytest

from sciagent_cli.approval_context import approval_context
from sciagent_contracts import (
    ActionSnapshot,
    BusinessInput,
    PendingQuestion,
    RecordedAnswer,
    ResearchRequest,
    RunBudget,
    RunPermissions,
    RunStatus,
    Workflow,
    WorkflowTask,
    WorkRequest,
    WorkRequestDraft,
)
from sciagent_orchestrator import InMemoryRunStore, ResearchRun


NOW = datetime(2026, 10, 16, 3, 0, tzinfo=UTC)


def _run() -> ResearchRun:
    return ResearchRun(
        run_id="run_review_context",
        request=ResearchRequest(
            goal="Evaluate the frozen model",
            hypothesis="A scalar temperature improves calibration.\n  用户原始假设  ",
            required_artifacts=["metrics.json", "Summary", "metrics.json"],
            required_evidence_kinds=["literature_fulltext", "literature_paper"],
            context="Original context\n  保留原文和空白  \n",
            constraints=["Do not read test before freezing", "Write only in the workspace"],
            budget=RunBudget(max_llm_calls=20, timeout_seconds=120),
            permissions=RunPermissions(execute_commands=True),
        ),
        workflow=Workflow(run_id="run_review_context", created_from="work_root", revision=1,
            tasks=[WorkflowTask(id="task_root", work_request_id="work_root",
                workflow_agent_kind="scientific", instruction="Evaluate the frozen model")]),
        work_requests=[WorkRequest(id="work_root", run_id="run_review_context",
            request=WorkRequestDraft(objective="Evaluate the frozen model",
                tasks=[{"key": "root", "instruction": "Evaluate the frozen model",
                    "selection": {"kind": "preset", "preset_ref": "scientific@1",
                        "reason": "The root research task is fixed Scientific"}}]),
            created_at=NOW, updated_at=NOW)],
        status=RunStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
    )


def _answer(question_id: str, *, minute: int = 1, **changes) -> RecordedAnswer:
    fields = {
        "question_id": question_id,
        "question_text": "Which split should be used?\nOriginal full question.",
        "requested_fields": ["split"],
        "options": {"split": ["calibration", "test"]},
        "values": {"split": "calibration\n  User free text  "},
        "answered_at": NOW + timedelta(minutes=minute),
        "run_id": "run_review_context",
        "session_id": "session_scientific",
        "task_id": "task_root",
        "attempt_number": 1,
    }
    fields.update(changes)
    return RecordedAnswer(**fields)


def test_context_preserves_original_request_and_root_dialogue_in_existing_order():
    run = _run()
    inputs = [
        BusinessInput(event_id="z_input", text="Newer user constraint\n  unchanged  ",
                      source="user", created_at=NOW + timedelta(minutes=2)),
        BusinessInput(event_id="z_same_time", text="An earlier user requirement",
                      source="user", created_at=NOW + timedelta(minutes=1)),
    ]
    for event in inputs:
        run.append_input("task_root", event)
    run.append_answer(_answer("question_a"))
    store = InMemoryRunStore()
    store.save(run)

    result = approval_context(store, run.run_id)

    assert result["initial_request"] == {
        "source": "user", "created_at": NOW.isoformat(),
        "goal": run.request.goal, "hypothesis": run.request.hypothesis,
        "context": run.request.context, "constraints": run.request.constraints,
        "required_artifacts": run.request.required_artifacts,
        "required_evidence_kinds": run.request.required_evidence_kinds,
    }
    dialogue = result["business_dialogue"]
    assert [item["event_id"] for item in dialogue] == ["z_input", "z_same_time", "question_a"]
    assert dialogue[2]["record"] == run.answers[0].model_dump(mode="json", exclude_none=True)
    assert dialogue[0]["record"] == inputs[0].model_dump(mode="json", exclude_none=True)
    assert dialogue[0]["record"]["source"] == "user"
    assert dialogue[2]["record"]["values"]["split"] == "calibration\n  User free text  "
    json.dumps(result, ensure_ascii=False)


def test_context_excludes_approval_parent_interpretation_and_child_copies():
    run = _run()
    action = ActionSnapshot(
        action_id="action_shell", tool="run_shell", arguments={"command": "echo approved"},
        run_id=run.run_id, session_id="session_coding", task_id="task_coding", attempt_number=1,
    )
    answers = [
        _answer("question_actual_user"),
        _answer("question_approval", requested_fields=["approve"], options=None,
                values={"approve": "yes"}, action=action, task_id="task_coding", attempt_number=1,
                session_id="session_coding", comment="Approval is not a business requirement"),
        _answer("question_parent", source="parent", source_session_id="session_scientific",
                task_id="task_coding", attempt_number=1, session_id="session_coding",
                values={"split": "Parent interpretation must not become a user instruction"}),
        _answer("question_forwarded", source="user", source_session_id="session_scientific",
                task_id="task_coding", attempt_number=1, session_id="session_coding",
                original_question_id="question_actual_user"),
        _answer("question_child", task_id="task_coding", attempt_number=1,
                session_id="session_coding"),
    ]
    inputs = [
        BusinessInput(event_id="input_user", text="Actual root user input", source="user", created_at=NOW),
        BusinessInput(event_id="input_parent", text="Parent interpretation", source="parent",
                      source_session_id="session_scientific", created_at=NOW),
    ]
    run.task_inputs = {"task_coding": [
        BusinessInput(event_id="input_task_only", text="Task-specific interpretation", source="parent",
                      source_session_id="session_scientific", created_at=NOW),
    ]}
    for event in inputs:
        run.append_input("task_root", event)
    for answer in answers:
        run.append_answer(answer)
    run.child_questions = {"question_pending_child": PendingQuestion(
        id="question_pending_child", run_id=run.run_id, session_id="session_coding",
        recipient_session_id="session_scientific", task_id="task_coding", attempt_number=1,
        text="A child question is not user authorization", requested_fields=["split"], created_at=NOW,
    )}
    store = InMemoryRunStore()
    store.save(run)

    result = approval_context(store, run.run_id)

    assert [item["event_id"] for item in result["business_dialogue"]] == [
        "input_user", "question_actual_user",
    ]
    encoded = json.dumps(result["business_dialogue"])
    for excluded in ("question_approval", "question_parent", "question_forwarded", "question_child",
                     "question_pending_child", "input_task_only", "input_parent", "action_shell"):
        assert excluded not in encoded


def test_context_does_not_modify_authoritative_records_or_read_other_sources():
    run = _run()

    class RunOnlyStore:
        def __init__(self):
            self.loads = []

        def load(self, run_id):
            self.loads.append(run_id)
            return run

    store = RunOnlyStore()
    before = run.model_dump(mode="json")

    result = approval_context(store, run.run_id)
    result["initial_request"]["constraints"].append("Changed result only")
    result["initial_request"]["required_artifacts"].append("Changed_result_only")
    result["initial_request"]["required_evidence_kinds"].append("literature_paper")

    assert store.loads == [run.run_id]
    assert run.model_dump(mode="json") == before
    assert result["business_dialogue"] == []
    assert result["prior_operation_decisions"] == []


def test_missing_run_is_explicit_instead_of_empty_authorization_context():
    with pytest.raises(KeyError):
        approval_context(InMemoryRunStore(), "run_missing")


def test_historical_operation_decisions_preserve_refusal_and_task_scope_separately():
    run = _run()
    action = ActionSnapshot(
        action_id="action_test", tool="run_shell", arguments={"command": "python evaluate_test.py"},
        context={"cwd": "/workspace/research", "shell": "bash"},
        run_id=run.run_id, session_id="session_experiment", task_id="task_experiment", attempt_number=2,
    )
    denial = _answer(
        "question_denied", minute=2, requested_fields=["approve"], options={"approve": ["yes", "no"]},
        values={"approve": "no"}, action=action, session_id=action.session_id,
        task_id=action.task_id, attempt_number=action.attempt_number,
        comment="Do not evaluate official test before freezing.\n  原始拒绝说明  ",
    )
    approval = _answer(
        "question_approved", minute=1, requested_fields=["approve"], options=None,
        values={"approve": "yes"}, action=action, session_id=action.session_id,
        task_id=action.task_id, attempt_number=action.attempt_number,
        comment="This exact operation only",
    )
    for answer in [denial, approval, _answer("question_business")]:
        run.append_answer(answer)
    store = InMemoryRunStore()
    store.save(run)

    result = approval_context(store, run.run_id)

    history = result["prior_operation_decisions"]
    assert [item["question_id"] for item in history] == ["question_approved", "question_denied"]
    assert [item["decision"] for item in history] == ["yes", "no"]
    assert history[1] == {**denial.model_dump(mode="json", exclude_none=True), "decision": "no"}
    assert history[1]["comment"] == "Do not evaluate official test before freezing.\n  原始拒绝说明  "
    assert history[1]["action"] == action.model_dump(mode="json")
    assert history[1]["source"] == "user"
    assert history[1]["task_id"] == "task_experiment"
    assert [item["event_id"] for item in result["business_dialogue"]] == ["question_business"]


def test_historical_operation_decisions_exclude_parent_and_forwarded_authority():
    run = _run()
    action = ActionSnapshot(
        action_id="action_parent", tool="run_shell", arguments={"command": "echo unsafe"},
        run_id=run.run_id, session_id="session_coding", task_id="task_coding", attempt_number=1,
    )
    common = {
        "requested_fields": ["approve"], "options": None, "values": {"approve": "yes"},
        "action": action, "session_id": action.session_id,
        "task_id": action.task_id, "attempt_number": action.attempt_number,
    }
    answers = [
        _answer("question_parent_decision", source="parent", source_session_id="session_scientific", **common),
        _answer("question_forwarded_decision", original_question_id="question_original", **common),
        _answer("question_parent_as_user", source="user", source_session_id="session_scientific", **common),
        _answer("question_parent_as_business_user", source="user", source_session_id="session_scientific"),
    ]
    for answer in answers:
        run.append_answer(answer)
    store = InMemoryRunStore()
    store.save(run)

    result = approval_context(store, run.run_id)

    assert result["prior_operation_decisions"] == []
    assert result["business_dialogue"] == []


def _context(run):
    store = InMemoryRunStore()
    store.save(run)
    return approval_context(store, run.run_id)


def test_root_business_answers_survive_attempt_changes_without_promoting_taskless_answers():
    run = _run()
    run.append_answer(_answer("question_old", attempt_number=1))
    run.append_answer(_answer("question_new", attempt_number=2, minute=-2))
    run.append_answer(_answer("question_taskless", task_id=None, attempt_number=None))

    dialogue = _context(run)["business_dialogue"]

    assert [item["event_id"] for item in dialogue] == ["question_old", "question_new"]
    assert [item["record"]["attempt_number"] for item in dialogue] == [1, 2]


def test_root_input_chain_is_authoritative_instead_of_legacy_run_input_copy():
    run = _run()
    run.upstream_inputs = [BusinessInput(event_id="input_legacy", text="stale copy",
        source="user", created_at=NOW)]
    run.append_input("task_root", BusinessInput(event_id="input_canonical", text="current source",
        source="user", created_at=NOW))

    assert [item["event_id"] for item in _context(run)["business_dialogue"]] == ["input_canonical"]


@pytest.mark.parametrize("changes", [
    {"source": "parent", "source_session_id": "session_parent"},
    {"source_session_id": "session_parent"},
    {"original_event_id": "input_original"},
])
def test_root_input_provenance_cannot_promote_forwarded_or_parent_text(changes):
    run = _run()
    run.append_input("task_root", BusinessInput(event_id="input_forwarded", text="old text",
        source="user", created_at=NOW).model_copy(update=changes))
    run.append_input("task_root", BusinessInput(event_id="input_direct", text="real revision",
        source="user", created_at=NOW))

    assert [item["event_id"] for item in _context(run)["business_dialogue"]] == ["input_direct"]


@pytest.mark.parametrize("changes", [
    {"source": "parent", "source_session_id": "session_parent"},
    {"source_session_id": "session_parent"},
    {"original_question_id": "question_original"},
])
def test_root_answer_provenance_cannot_promote_forwarded_or_parent_text(changes):
    run = _run()
    run.append_answer(_answer("question_forwarded", **changes))
    run.append_answer(_answer("question_direct"))

    assert [item["event_id"] for item in _context(run)["business_dialogue"]] == ["question_direct"]


@pytest.mark.parametrize("order", [
    [], ["input_one"], ["input_one", "input_two", "input_unknown"],
    ["input_one", "input_two", "input_two"],
])
def test_incomplete_or_duplicate_reception_order_is_not_repaired_for_authorization(order):
    run = _run()
    for identity in ["input_one", "input_two"]:
        run.append_input("task_root", BusinessInput(event_id=identity, text=identity,
            source="user", created_at=NOW))
    run.task_input_order["task_root"] = order

    with pytest.raises(ValueError, match="exactly match"):
        _context(run)


def test_filtered_records_still_require_complete_order_before_source_selection():
    run = _run()
    run.task_inputs["task_root"] = [BusinessInput(event_id="input_parent", text="not user authority",
        source="parent", source_session_id="session_parent", created_at=NOW)]

    with pytest.raises(ValueError, match="exactly match"):
        _context(run)


@pytest.mark.parametrize("collision", ["duplicate_input", "duplicate_answer", "input_answer"])
def test_duplicate_business_record_identity_never_silently_overwrites_evidence(collision):
    run = _run()
    event = BusinessInput(event_id="question_one", text="instruction", source="user", created_at=NOW)
    answer = _answer("question_one")
    if collision == "duplicate_input":
        run.task_inputs["task_root"] = [event, event]
    elif collision == "duplicate_answer":
        run.answers = [answer, answer]
    else:
        run.task_inputs["task_root"] = [event]
        run.answers = [answer]
    run.task_input_order["task_root"] = ["question_one"]

    with pytest.raises(ValueError, match="exactly match"):
        _context(run)


def test_foreign_run_root_answer_is_not_treated_as_user_authority():
    run = _run()
    run.append_answer(_answer("question_foreign", run_id="run_foreign"))

    with pytest.raises(ValueError, match="another Run"):
        _context(run)


@pytest.mark.parametrize("invalid", ["no_workflow", "no_root_task", "multiple_root_tasks", "wrong_role"])
def test_reviewer_requires_one_actual_root_scientific_task(invalid):
    run = _run()
    if invalid == "no_workflow":
        run.workflow = None
    elif invalid == "no_root_task":
        run.workflow.tasks.clear()
    elif invalid == "multiple_root_tasks":
        run.workflow.tasks.append(run.workflow.tasks[0].model_copy(update={"id": "task_other"}))
    else:
        from sciagent_contracts import WorkflowAgentKind
        run.workflow.tasks[0].workflow_agent_kind = WorkflowAgentKind.EXECUTOR

    with pytest.raises(ValueError, match="one root Scientific"):
        _context(run)
