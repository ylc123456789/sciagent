"""Common Agent boundaries and artifact-owned recovery content."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import sciagent_contracts as contracts
from sciagent_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, AgentResult, ArtifactCandidate,
    ArtifactRef, ControlSignal, ModuleError, QuestionDraft, RecordedAnswer,
    SessionRef, TaskBudget, TaskProposal, PendingQuestion, ControlOrigin, WorkResult,
    WorkRequestDraft, WorkTaskResult, WorkflowProposal,
)

NOW = datetime.now(UTC)


def request(**changes):
    data = dict(
        agent="executor", run_id="run_test", work_request_id="work_test", task_id="task_test", attempt_number=1,
        instruction="Inspect the implementation", budget=TaskBudget(max_llm_calls=5, timeout_seconds=60),
    )
    data.update(changes)
    data.setdefault("permissions", AgentPermissions(execute_commands=True))
    return AgentRequest(**data)


def artifact(**changes):
    data = dict(
        id="artifact_data", kind="data", producer="executor", run_id="run_test",
        task_id="task_test", attempt_number=1, uri="file:///tmp/answer.json", sha256="0" * 64,
        media_type="application/json", summary="Reply", metadata={},
    )
    data.update(changes)
    return ArtifactRef(**data)


def session(module="executor"):
    return SessionRef(
        id="session_test", module=module, state_uri="session://session_test",
        status="paused", created_at=NOW, updated_at=NOW,
    )


@pytest.mark.parametrize("agent", ["executor", "scientific"])
def test_all_agents_use_the_same_request_and_result(agent):
    context = dict(task_id="task_root", work_request_id="work_root") if agent == "scientific" else {}
    req = request(agent=agent, **context)
    assert AgentRequest.model_validate_json(req.model_dump_json()) == req
    result = AgentResult(status="completed", report="Reviewed available materials")
    assert AgentResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("field", [
    "goal", "constraints", "inputs", "dataset_refs", "answers", "facts", "continuation",
    "acceptance", "acceptance_ref", "required_evidence_kinds", "required_artifacts", "capability", "mode",
    "environment_spec",
])
def test_request_rejects_parallel_task_semantics(field):
    with pytest.raises(ValidationError, match="Extra inputs"):
        request(**{field: []})


def test_request_enforces_invocation_scope_and_dispatch_authority():
    assert request(agent="scientific").task_id == "task_test"
    with pytest.raises(ValidationError, match="work_request_id"):
        request(work_request_id=None)
    with pytest.raises(ValidationError, match="task_id"):
        request(task_id=None)
    assert request(permissions=AgentPermissions(request_work=True, execute_commands=True)).permissions.request_work
    with pytest.raises(ValidationError):
        request(work_depth=4, max_work_depth=3)
    with pytest.raises(ValidationError):
        request(agent="orchestrator")


def test_run_and_agent_require_explicit_permissions():
    with pytest.raises(ValidationError, match="permissions"):
        AgentRequest.model_validate(request().model_dump(exclude={"permissions"}))
    with pytest.raises(ValidationError, match="permissions"):
        contracts.ResearchRequest(goal="Inspect", budget=contracts.RunBudget(
            max_llm_calls=5, timeout_seconds=60,
        ))
    assert not contracts.RunPermissions().execute_commands
    assert not AgentPermissions().execute_commands


def test_resume_data_requires_origin_session_and_exact_answer_scope():
    answer = RecordedAnswer(question_id="question_test", question_text="Which data?",
        requested_fields=["dataset"], values={"dataset":"demo"}, answered_at=NOW,
        run_id="run_test", session_id="session_test", task_id="task_test", attempt_number=1)
    with pytest.raises(ValidationError, match="parent session"):
        request(resume_answers=[answer])
    request(parent_session_id="session_test", resume_answers=[answer], user_answers=[answer],
            control_ack=ControlOrigin(session_id="session_test", turn=0, call_id="call_1"))
    with pytest.raises(ValidationError, match="another session"):
        request(parent_session_id="session_other", resume_answers=[answer])
    with pytest.raises(ValidationError, match="this invocation"):
        request(parent_session_id="session_test", user_answers=[answer.model_copy(update={"attempt_number":2})])
    with pytest.raises(ValidationError, match="Extra inputs"):
        request(resume_artifact_ids=[])


def test_input_materials_reject_cross_run_and_duplicate_identity():
    ref = artifact()
    with pytest.raises(ValidationError, match="unique"):
        request(input_artifacts=[ref, ref])
    with pytest.raises(ValidationError, match="same run"):
        request(input_artifacts=[artifact(run_id="run_other")])


def test_pause_control_is_direct_and_matches_paused_session():
    question = PendingQuestion(id="question_test", run_id="run_test", session_id="session_test",
        task_id="task_test", attempt_number=1, text="Which data?", requested_fields=["dataset"], created_at=NOW)
    control = ControlSignal(action="ask_user", source_session_id="session_test", source_turn=0,
                            source_call_id="call_1", question=question)
    result = AgentResult(status="needs_user_input", report="Awaiting dataset", session=session(), control=control)
    assert result.artifacts == []
    assert AgentResult.model_validate_json(result.model_dump_json()) == result
    for change in [dict(session=None), dict(control=None)]:
        with pytest.raises(ValidationError):
            AgentResult.model_validate(result.model_dump() | change)
    with pytest.raises(ValidationError):
        ControlSignal(action="ask_user", source_session_id="session_test", source_turn=0, source_call_id="x")


@pytest.mark.parametrize("field", ["payload", "question", "request_work", "opinion", "assessment"])
def test_result_rejects_legacy_business_outputs(field):
    with pytest.raises(ValidationError, match="Extra inputs"):
        AgentResult(status="completed", report="Done", **{field: {}})


def test_failed_result_keeps_report_and_partial_artifacts():
    log = ArtifactCandidate(kind="log", path="run.log", media_type="text/plain", summary="Failure log", content="failed")
    result = AgentResult(status="failed", report="Command failed", artifacts=[log], error=ModuleError(
        code="tool_failed", message="exit 1", retryable=False,
    ))
    assert result.artifacts == [log]


def test_recorded_answer_retains_question_snapshot_and_exact_scope():
    data = dict(
        question_id="question_test", question_text="Which dataset?", requested_fields=["dataset"],
        options={"dataset": ["demo", "full"]}, values={"dataset": "demo"}, answered_at=NOW,
        run_id="run_test", task_id="task_test", attempt_number=2,
    )
    saved = RecordedAnswer(**data)
    assert RecordedAnswer.model_validate_json(saved.model_dump_json()) == saved
    for changes in [dict(question_text=""), dict(attempt_number=None), dict(values={"other": "demo"})]:
        with pytest.raises(ValidationError):
            RecordedAnswer(**(data | changes))


def test_work_result_allows_honest_zero_task_failure_without_report_body():
    error=ModuleError(code="contract_error", message="Compiler could not form a workflow", retryable=False)
    result=WorkResult(run_id="run_test", session_id="session_test", work_request_id="work_test",
                      objective="Measure", status="failed", error=error)
    assert result.tasks == [] and result.artifacts == []
    with pytest.raises(ValidationError, match="requires error"):
        WorkResult.model_validate(result.model_dump() | {"error":None})
    with pytest.raises(ValidationError, match="Extra inputs"):
        WorkResult.model_validate(result.model_dump() | {"summary":"Invented aggregate"})


def test_graph_represents_root_scientific():
    task = TaskProposal(execution_ref=("test@1" if str("scientific") == "executor" else None), id="task_root", work_request_id="work_root", instruction="Measure",
                        workflow_agent_kind="scientific")
    assert task.workflow_agent_kind.value == "scientific"


def test_contract_revalidation_rejects_unchecked_model_copy():
    invalid_request = request().model_copy(update={"work_depth": 0})
    with pytest.raises(ValidationError, match="work_depth"):
        AgentRequest.model_validate(invalid_request)
    task = TaskProposal(execution_ref="test@1",
        id="task_test", work_request_id="work_test", workflow_agent_kind="executor", instruction="Inspect",
    ).model_copy(update={"execution_ref": ("test@1" if str("unknown") == "executor" else None), "workflow_agent_kind": "unknown"})
    with pytest.raises(ValidationError):
        WorkflowProposal(work_request_id="work_test", tasks=[task])


def test_replaced_contracts_are_not_exported_or_aliased():
    for name in ["Capability", "CapabilityInput", "ModuleTaskRequest", "ModuleResult",
                 "ResearchIndex", "WorkFeedback", "WorkRecord", "WorkOutcome", "CodeUnderstandInput", "CodeModifyResult", "ExperimentResult", "ScientificTurnRequest", "ScientificTurnResult"]:
        assert not hasattr(contracts, name)


@pytest.mark.parametrize("kind", sorted(contracts.SYSTEM_ARTIFACT_KINDS))
def test_system_kind_policy_rejects_wrong_producer_and_source(kind):
    source, scopes = contracts.SYSTEM_ARTIFACT_PROVENANCE[kind]
    for scope in scopes:
        fields = {
            "run": {}, "task": {"task_id": "task_test"},
            "attempt": {"task_id": "task_test", "attempt_number": 1},
            "session": {"session_id": "session_test"},
        }[scope]
        data = {"producer": "orchestrator", "kind": kind, "task_id": None, "attempt_number": None,
                "metadata": {"source_type": source}} | fields
        ref = artifact(**data)
        assert ref.producer == AgentOwner.ORCHESTRATOR
        with pytest.raises(ValidationError, match="orchestrator producer"):
            ArtifactRef.model_validate(ref.model_dump() | {"producer": "executor"})
        with pytest.raises(ValidationError, match="source_type"):
            ArtifactRef.model_validate(ref.model_dump() | {"metadata": {"source_type": "import"}})


def test_unknown_scientific_session_kind_is_rejected():
    with pytest.raises(ValidationError, match="unsupported scientific"):
        artifact(kind="unknown_kind", producer="scientific", task_id=None, attempt_number=None,
                 session_id="session_test", metadata={})


def test_environment_authority_is_not_a_separate_permission_or_workspace_contract():
    with pytest.raises(ValidationError, match="Extra inputs"):
        contracts.RunPermissions(prepare_environment=True)
    with pytest.raises(ValidationError, match="Extra inputs"):
        AgentPermissions(prepare_environment=True)
    with pytest.raises(ValidationError, match="Extra inputs"):
        contracts.WorkspaceSpec(
            workspace_id="ws_test", source_kind="generated",
            access=contracts.WorkspaceAccess(read_paths=["."], write_paths=["."]),
            environment={"python_version": "3.12"},
        )
