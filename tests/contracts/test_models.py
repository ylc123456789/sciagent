
from sciagent_contracts import RunPermissions, ExecutionLimits
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sciagent_contracts import (ArtifactCandidate, ArtifactRef, AgentOwner, Attempt, AttemptStatus, TaskAcceptanceSpec, WorkflowAgentKind, ErrorCode, ModuleError, AgentResult, ModuleStatus, PendingQuestion, QuestionDraft, RecordedAnswer, RunBudget, TaskBudget, TaskProposal, TaskStatus, UserAnswer, Workflow, WorkflowPatch, WorkflowProposal, WorkflowTask, WorkspaceGrant, WorkspaceAccess, WorkspaceRecord, WorkspaceSourceKind, WorkspaceSpec, AgentRequest, ResearchRequest, WarningRecord)


NOW = datetime(2026, 8, 26, tzinfo=UTC)


def research_request() -> ResearchRequest:
    return ResearchRequest(goal='Evaluate the proposed method', context='A small reference implementation is available.', budget=RunBudget(max_llm_calls=20, timeout_seconds=3600), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=8, max_attempts_per_task=2))


def task(task_id: str, depends_on: list[str] | None = None) -> WorkflowTask:
    return WorkflowTask(execution_ref="test@1",
        id=task_id,
        work_request_id="work_test",
        workflow_agent_kind=WorkflowAgentKind.EXECUTOR,
        instruction="Inspect the entry point",

        depends_on=depends_on or [],
    )


def module_error() -> ModuleError:
    return ModuleError(
        code=ErrorCode.TOOL_FAILED,
        message="Tool exited with a non-zero status",
        retryable=True,
    )


def _artifact_ref(**overrides: object) -> ArtifactRef:
    values: dict[str, object] = {
        "id": "artifact_contract",
        "kind": "module_report",
        "producer": AgentOwner.ORCHESTRATOR,
        "run_id": "run_example",
        "task_id": "task_example", "attempt_number": 1,
        "uri": "file:///tmp/work-feedback.json",
        "sha256": "0" * 64,
        "media_type": "application/json",
        "summary": "contract test artifact",
        "metadata": {"source_type": "task_report"},
    }
    values.update(overrides)
    return ArtifactRef(**values)


def test_system_artifact_provenance_shapes_are_explicit() -> None:
    feedback = _artifact_ref()
    assert feedback.task_id == "task_example" and feedback.attempt_number == 1

    acceptance = _artifact_ref(
        id="artifact_acceptance",
        kind="acceptance_requirements",
        session_id=None, attempt_number=None,
        task_id="task_example",
        metadata={"source_type": "task_requirement"},
    )
    assert acceptance.task_id == "task_example"


def test_system_artifact_provenance_rejects_mismatched_kind_scope() -> None:
    with pytest.raises(ValidationError, match="acceptance_requirements"):
        _artifact_ref(
            kind="acceptance_requirements",
            task_id="task_example",
            session_id=None, attempt_number=None,
            metadata={"source_type": "task_report"},
        )

    with pytest.raises(ValidationError, match="orchestrator scope"):
        _artifact_ref(
            kind="module_report",
            task_id="task_example",
            session_id=None, attempt_number=None,
            metadata={"source_type": "task_report"},
        )

def test_research_request_does_not_accept_deployment_resources() -> None:
    from sciagent_contracts import (ResearchRequest)

    assert "dataset_refs" not in ResearchRequest.model_fields
    with pytest.raises(ValidationError, match="dataset_refs"):
        ResearchRequest(goal='Discover resources during execution', budget=RunBudget(max_llm_calls=5, timeout_seconds=60), dataset_refs=[], permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1))


def test_task_acceptance_spec_is_task_control_plane_data() -> None:
    spec = TaskAcceptanceSpec(
        required_metric_keys=["accuracy"],
        required_artifact_paths=["results.json"],
    )
    task = TaskProposal(execution_ref="test@1",
        id="task_experiment",
        work_request_id="work_test",
        workflow_agent_kind=WorkflowAgentKind.EXECUTOR,
        instruction="Run the experiment",

        acceptance_spec=spec,
    )
    assert task.acceptance_spec == spec
    assert "acceptance_ref" not in TaskProposal.model_fields

    materialized = WorkflowTask(execution_ref=("test@1" if str(task.workflow_agent_kind) == "executor" else None),
        id=task.id,
        work_request_id=task.work_request_id,
        workflow_agent_kind=task.workflow_agent_kind,
        instruction=task.instruction,

        acceptance_ref=_artifact_ref(
            kind="acceptance_requirements", session_id=None, task_id=task.id, attempt_number=None,
            metadata={"source_type": "task_requirement"},
        ),
    )
    assert "acceptance_spec" not in WorkflowTask.model_fields
    assert materialized.acceptance_ref.kind == "acceptance_requirements"


def test_schema_round_trip_preserves_contract() -> None:
    workflow = Workflow(
        run_id="run_example",
        revision=1,
        tasks=[task("task_plan")],
        created_from="work_test",
    )

    restored = Workflow.model_validate_json(workflow.model_dump_json())

    assert restored == workflow
    assert restored.schema_version == '32.0'


@pytest.mark.parametrize("schema_version", ["3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0", "25.0", "26.0", "27.0", "28.0", "29.0"])
def test_previous_schema_state_is_rejected(schema_version: str) -> None:
    with pytest.raises(ValidationError):
        Workflow(
            run_id="run_example",
            revision=1,
            tasks=[],
            created_from="work_test",
            schema_version=schema_version,
        )


def test_previous_run_request_schema_is_rejected() -> None:
    value = research_request().model_dump(mode="json")
    value["schema_version"] = "23.0"
    with pytest.raises(ValidationError, match="schema_version"):
        ResearchRequest.model_validate(value)


def test_task_budget_rejects_removed_max_steps_field() -> None:
    with pytest.raises(ValidationError, match="max_steps"):
        TaskBudget(max_llm_calls=5, timeout_seconds=60, max_steps=5)


def test_user_answer_requires_at_least_one_value() -> None:
    with pytest.raises(ValidationError, match="values"):
        UserAnswer(question_id="question_x", values={}, answered_at=NOW)


def test_unsupported_evidence_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(goal='Evaluate the method', required_evidence_kinds=['code_change'], budget=RunBudget(max_llm_calls=10, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1))


@pytest.mark.parametrize(
    ("field", "bad_id"),
    [
        ("run_id", "task_example"),
        ("task_id", "run_example"),
        ("question_id", "artifact_example"),
    ],
)
def test_id_namespaces_are_not_interchangeable(field: str, bad_id: str) -> None:
    if field == "run_id":
        with pytest.raises(ValidationError):
            Workflow(run_id=bad_id, revision=1, tasks=[], created_from="work_test")
    elif field == "task_id":
        with pytest.raises(ValidationError):
            task(bad_id)
    else:
        with pytest.raises(ValidationError):
            UserAnswer(question_id=bad_id, values={"choice": "yes"}, answered_at=NOW)


def test_workflow_rejects_unknown_dependency() -> None:
    with pytest.raises(ValidationError, match="unknown task"):
        Workflow(
            run_id="run_example",
            revision=1,
            tasks=[task("task_analyze", ["task_missing"])],
            created_from="work_test",
        )


def test_workflow_rejects_dependency_cycle() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Workflow(
            run_id="run_example",
            revision=1,
            tasks=[
                task("task_first", ["task_second"]),
                task("task_second", ["task_first"]),
            ],
            created_from="work_test",
        )


def test_proposal_rejects_duplicate_task_ids() -> None:
    proposal_task = TaskProposal(execution_ref="test@1",
        id="task_plan",
        work_request_id="work_test",
        workflow_agent_kind=WorkflowAgentKind.EXECUTOR,
        instruction="Inspect the entry point",

    )

    with pytest.raises(ValidationError, match="duplicate task"):
        WorkflowProposal(
            work_request_id="work_test",
            tasks=[proposal_task, proposal_task],
        )


@pytest.mark.parametrize("status", [ModuleStatus.FAILED, ModuleStatus.BLOCKED])
def test_failed_or_blocked_result_requires_error(status: ModuleStatus) -> None:
    with pytest.raises(ValidationError, match="error"):
        AgentResult(status=status, report="Could not continue")


def test_question_draft_requires_at_least_one_field() -> None:
    with pytest.raises(ValidationError, match="requested_fields"):
        QuestionDraft(text="Which?", requested_fields=[])


def test_question_keeps_background_in_visible_text_not_a_hidden_reason() -> None:
    question = QuestionDraft(
        text="No dataset is registered. Which dataset should be prepared?",
        requested_fields=["dataset"],
    )
    assert QuestionDraft.model_validate_json(question.model_dump_json()) == question
    with pytest.raises(ValidationError, match="reason"):
        QuestionDraft(**question.model_dump(), reason="A second explanation")


@pytest.mark.parametrize("field", ["summary", "compilation_rationale"])
def test_proposal_rejects_removed_graph_prose(field: str) -> None:
    candidate = WorkflowProposal(
        work_request_id="work_test",
        tasks=[TaskProposal(execution_ref="test@1",
            id="task_inspect", work_request_id="work_test",
            workflow_agent_kind=WorkflowAgentKind.EXECUTOR, instruction="Inspect the entry point",

        )],
    )
    assert WorkflowProposal.model_validate_json(candidate.model_dump_json()) == candidate
    with pytest.raises(ValidationError, match=field):
        WorkflowProposal(**candidate.model_dump(), **{field: "Unused graph prose"})


def test_patch_rejects_removed_graph_reason() -> None:
    patch = WorkflowPatch(work_request_id="work_test", based_on_revision=1)
    assert WorkflowPatch.model_validate_json(patch.model_dump_json()) == patch
    with pytest.raises(ValidationError, match="reason"):
        WorkflowPatch(**patch.model_dump(), reason="Unused graph prose")


def test_warning_status_and_warning_records_cannot_disagree() -> None:
    warning = WarningRecord(code="unverified", message="One metric was not verified")

    with pytest.raises(ValidationError, match="warnings"):
        AgentResult(
            status=ModuleStatus.COMPLETED_WITH_WARNINGS,
            report="Done with a limitation",
        )

    with pytest.raises(ValidationError, match="warnings"):
        AgentResult(
            status=ModuleStatus.COMPLETED,
            report="Done",
            warnings=[warning],
        )


def test_artifact_candidate_is_not_a_registered_artifact() -> None:
    candidate = ArtifactCandidate(
        kind="experiment_result",
        path="outputs/metrics.json",
        media_type="application/json",
        summary="Unregistered metrics",
    )

    dumped = candidate.model_dump()
    assert "id" not in dumped
    assert "run_id" not in dumped
    assert "sha256" not in dumped


def test_attempt_rejects_illegal_terminal_combinations() -> None:
    with pytest.raises(ValidationError, match="finished_at"):
        Attempt(number=1, status=AttemptStatus.COMPLETED, started_at=NOW)

    with pytest.raises(ValidationError, match="error"):
        Attempt(
            number=1,
            status=AttemptStatus.FAILED,
            started_at=NOW,
            finished_at=NOW,
        )


def test_attempt_report_round_trips() -> None:
    attempt = Attempt(
        number=1,
        status=AttemptStatus.COMPLETED,
        started_at=NOW,
        finished_at=NOW,
        report="Measured accuracy; numeric result is stored as an artifact",
    )
    restored = Attempt.model_validate_json(attempt.model_dump_json())
    assert restored.report == attempt.report


def test_question_and_answer_have_distinct_owners() -> None:
    pending = PendingQuestion(
        id="question_dataset",
        run_id="run_example", session_id="session_example",
        task_id="task_plan",
        text="Which dataset should be used?",
        requested_fields=["dataset"],
        created_at=NOW,
    )
    answer = UserAnswer(
        question_id=pending.id,
        values={"dataset": "demo"},
        answered_at=NOW,
    )

    assert pending.id == answer.question_id


def test_recorded_answer_requires_question_but_user_input_does_not() -> None:
    answer = UserAnswer(
        question_id="question_choice", values={"answer": "second"}, answered_at=NOW,
    )
    assert "question_text" not in UserAnswer.model_fields
    with pytest.raises(ValidationError, match="question_text"):
        RecordedAnswer.model_validate(answer.model_dump())
    with pytest.raises(ValidationError, match="question_text"):
        RecordedAnswer(**answer.model_dump(), question_text="   ")
    recorded = RecordedAnswer(
        **answer.model_dump(), question_text="First: train. Second: evaluate.",
        requested_fields=["answer"], run_id="run_example", session_id="session_example",
    )
    assert RecordedAnswer.model_validate_json(recorded.model_dump_json()) == recorded


def test_workspace_grant_rejects_paths_outside_root() -> None:
    with pytest.raises(ValidationError, match="relative"):
        WorkspaceGrant(root='/work/repo', source=WorkspaceSourceKind.LOCAL, access=WorkspaceAccess(read_paths=['/etc/passwd'], write_paths=[]))


@pytest.mark.parametrize("bad_path", ["../x", "..\\x", "a/../../x", "a\\..\\..\\x"])
def test_relative_path_rejects_cross_platform_traversal(bad_path: str) -> None:
    with pytest.raises(ValidationError, match="relative"):
        ArtifactCandidate(
            kind="text",
            path=bad_path,
            media_type="text/plain",
            summary="evidence",
        )


@pytest.mark.parametrize("good_path", ["train.py", "src/model.py", "src\\model.py"])
def test_relative_path_accepts_workspace_relative_forms(good_path: str) -> None:
    candidate = ArtifactCandidate(
        kind="text",
        path=good_path,
        media_type="text/plain",
        summary="evidence",
    )
    assert candidate.path == good_path


def test_workspace_source_kind_values() -> None:
    assert {kind.value for kind in WorkspaceSourceKind} == {
        "git",
        "local",
        "copy",
        "generated",
    }


def test_workspace_spec_and_record_round_trip() -> None:
    spec = WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.LOCAL, location='/tmp/repo', access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))
    record = WorkspaceRecord(
        workspace_id="ws_main", root="/tmp/repo", source=spec, managed=False
    )

    restored = WorkspaceRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.source.source_kind == WorkspaceSourceKind.LOCAL


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RunBudget(max_llm_calls=20, timeout_seconds=3600, undocumented_switch=True)


@pytest.mark.parametrize("write_paths", [["."], ["data"]])
def test_parent_read_workspaces_cannot_carry_write_authority(write_paths):
    from sciagent_contracts import AgentPermissions

    grant = WorkspaceGrant(root="/authorized/source", source="local", access=WorkspaceAccess(
        read_paths=["."], write_paths=write_paths, denied_paths=["private"],
    ))
    with pytest.raises(ValidationError, match="read_workspaces must contain read-only grants"):
        AgentRequest(run_id="run_read_scope", work_request_id="work_root", task_id="task_root", attempt_number=1, agent=AgentOwner.SCIENTIFIC,
            instruction="Inspect the current file", permissions=AgentPermissions(),
            budget=TaskBudget(max_llm_calls=2, timeout_seconds=30), read_workspaces=[grant])
