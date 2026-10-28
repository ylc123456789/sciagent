"""Current WorkRequest lifecycle, scientific opinions and artifact provenance."""

from resagent2_contracts import RunPermissions, ExecutionLimits

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from resagent2_contracts import (AgentOwner, ArtifactId, ArtifactRef, ModuleError, ModuleStatus, ErrorCode, ResearchRequest, RunBudget, RunId, ScientificAssessment, ScientificOpinion, ScientificVerdict, SessionRef, SessionStatus, TaskBudget, WorkResult, WorkRequest, WorkRequestDraft, WorkRequestStatus, WorkTaskResult, QuestionDraft, RecordedAnswer)

NOW = datetime(2026, 8, 28, tzinfo=UTC)


def research_request() -> ResearchRequest:
    return ResearchRequest(goal='Evaluate the method', budget=RunBudget(max_llm_calls=20, timeout_seconds=3600), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=8, max_attempts_per_task=2))


def session_ref(status: SessionStatus = SessionStatus.PAUSED) -> SessionRef:
    return SessionRef(
        id="session_sci",
        module=AgentOwner.SCIENTIFIC,
        state_uri="memory://sci",
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


def work_request(status: WorkRequestStatus = WorkRequestStatus.REQUESTED) -> WorkRequest:
    fields = {
        "id": "work_round1",
        "run_id": "run_example",
        "owner_session_id": "session_sci", "owner_task_id": "task_root", "owner_attempt_number": 1, "depth": 2,
        "request": WorkRequestDraft(objective='Measure the method', tasks=[{"selection": {"kind": "preset", "preset_ref": "test@1", "reason": "Test task capability"}, 'key': 'work', 'instruction': 'Measure the method'}]),
        "status": status,
        "created_at": NOW,
        "updated_at": NOW,
    }
    if status in {
        WorkRequestStatus.EXECUTING,
        WorkRequestStatus.STABLE,
        WorkRequestStatus.CONSUMED,
    }:
        fields["workflow_revision"] = 1
    if status in {WorkRequestStatus.STABLE, WorkRequestStatus.CONSUMED}:
        fields["outcome"] = WorkResult(
            run_id="run_example", session_id="session_sci", objective="Measure the method", status="stable",
            work_request_id="work_round1",
            workflow_revision=1,
            tasks=[
                WorkTaskResult(execution_ref="test@1",
                    task_id="task_experiment",
                    status="completed",
                    instruction="Run experiment", workflow_agent_kind="executor",
                )
            ],
        )
    if status in {WorkRequestStatus.FAILED, WorkRequestStatus.CANCELLED}:
        fields["error"] = ModuleError(
            code=ErrorCode.CONTRACT_ERROR,
            message="compile failed",
            retryable=False,
        )
    return WorkRequest(**fields)


def test_work_request_round_trips() -> None:
    for status in WorkRequestStatus:
        request = work_request(status)
        restored = WorkRequest.model_validate_json(request.model_dump_json())
        assert restored == request
        assert restored.status == status


def test_work_request_rejects_execution_fields_when_requested() -> None:
    with pytest.raises(ValidationError, match="cannot carry execution fields"):
        WorkRequest(
            id="work_round1",
            run_id="run_example",
            owner_session_id="session_sci", owner_task_id="task_root", owner_attempt_number=1, depth=2,
            request=WorkRequestDraft(objective='Measure', tasks=[{"selection": {"kind": "preset", "preset_ref": "test@1", "reason": "Test task capability"}, 'key': 'work', 'instruction': 'Measure'}]),
            status=WorkRequestStatus.REQUESTED,
            workflow_revision=1,
            created_at=NOW,
            updated_at=NOW,
        )


def test_work_request_rejects_execution_fields_from_llm() -> None:
    """WorkRequestDraft must reject capability/task/path/env/status fields."""
    with pytest.raises(ValidationError):
        WorkRequestDraft(objective='Measure', capability='experiment_run', tasks=[{"selection": {"kind": "preset", "preset_ref": "test@1", "reason": "Test task capability"}, 'key': 'work', 'instruction': 'Measure'}])
    with pytest.raises(ValidationError):
        WorkRequestDraft(objective='Measure', task_id='task_experiment', tasks=[{"selection": {"kind": "preset", "preset_ref": "test@1", "reason": "Test task capability"}, 'key': 'work', 'instruction': 'Measure'}])


def test_work_request_draft_requires_tasks() -> None:
    with pytest.raises(ValidationError, match="tasks"):
        WorkRequestDraft(objective="Measure")


def test_supports_opinion_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        ScientificOpinion(verdict=ScientificVerdict.SUPPORTS, statement="it works")


def test_inconclusive_opinion_allows_empty_evidence() -> None:
    opinion = ScientificOpinion(
        verdict=ScientificVerdict.INCONCLUSIVE,
        statement="not enough evidence",
    )
    assert opinion.evidence_artifact_ids == []


def _execution_artifact() -> ArtifactRef:
    return ArtifactRef(
        id="artifact_metrics",
        kind="experiment_result",
        producer=AgentOwner.EXECUTOR,
        run_id="run_example",
        task_id="task_experiment",
        attempt_number=1,
        uri="file:///artifacts/metrics.json",
        sha256="0" * 64,
        media_type="application/json",
        summary="metrics",
    )


def test_artifact_ref_accepts_execution_provenance() -> None:
    artifact = _execution_artifact()
    assert artifact.task_id == "task_experiment"
    assert artifact.attempt_number == 1
    assert artifact.session_id is None


def test_artifact_ref_rejects_half_task_attempt() -> None:
    with pytest.raises(ValidationError, match="both task_id and attempt_number"):
        ArtifactRef(
            id="artifact_metrics",
            kind="experiment_result",
            producer=AgentOwner.EXECUTOR,
            run_id="run_example",
            task_id="task_experiment",
            uri="file:///artifacts/metrics.json",
            sha256="0" * 64,
            media_type="application/json",
            summary="metrics",
        )


def test_artifact_ref_accepts_scientific_session_provenance() -> None:
    artifact = ArtifactRef(
        id="artifact_lit",
        kind="literature_paper",
        producer=AgentOwner.SCIENTIFIC,
        run_id="run_example",
        session_id="session_sci",
        uri="file:///artifacts/lit.json",
        sha256="0" * 64,
        media_type="application/json",
        summary="literature results",
    )
    assert artifact.session_id == "session_sci"
    assert artifact.task_id is None


def test_artifact_ref_rejects_incomplete_session_task_provenance() -> None:
    with pytest.raises(ValidationError, match="complete Task/Attempt pair"):
        ArtifactRef(
            id="artifact_lit",
            kind="literature_paper",
            producer=AgentOwner.SCIENTIFIC,
            run_id="run_example",
            session_id="session_sci",
            task_id="task_experiment",
            uri="file:///artifacts/lit.json",
            sha256="0" * 64,
            media_type="application/json",
            summary="literature results",
        )


def test_artifact_ref_accepts_orchestrator_import_provenance() -> None:
    artifact = ArtifactRef(
        id="artifact_imported",
        kind="input",
        producer=AgentOwner.ORCHESTRATOR,
        run_id="run_example",
        uri="file:///artifacts/input.json",
        sha256="0" * 64,
        media_type="application/json",
        summary="imported input",
        metadata={"source_type": "import"},
    )
    assert artifact.task_id is None
    assert artifact.session_id is None


def test_artifact_ref_rejects_orchestrator_without_source_type() -> None:
    with pytest.raises(ValidationError, match="source_type"):
        ArtifactRef(
            id="artifact_imported",
            kind="input",
            producer=AgentOwner.ORCHESTRATOR,
            run_id="run_example",
            uri="file:///artifacts/input.json",
            sha256="0" * 64,
            media_type="application/json",
            summary="imported input",
        )


def test_work_result_rejects_duplicate_task_ids() -> None:
    task=WorkTaskResult(execution_ref="test@1", task_id="task_x", status="completed", instruction="Measure", workflow_agent_kind="executor")
    with pytest.raises(ValidationError, match="task ids must be unique"):
        WorkResult(run_id="run_example", session_id="session_sci", objective="Measure", status="stable",
                   work_request_id="work_round1", workflow_revision=1, tasks=[task,task])


def test_work_request_rejects_legacy_schema() -> None:
    """A legacy WorkRequest cannot validate as the current contract."""
    with pytest.raises(ValidationError):
        WorkRequest.model_validate(
            {
                "schema_version": "1.1",
                "id": "work_round1",
                "run_id": "run_example",
                "owner_session_id": "session_sci", "owner_task_id": "task_root", "owner_attempt_number": 1, "depth": 2,
                "request": {'objective': 'Measure', 'tasks': [{"selection": {"kind": "preset", "preset_ref": "test@1", "reason": "Test task capability"}, 'key': 'work', 'instruction': 'Measure'}]},
                "status": "requested",
                "created_at": NOW.isoformat(),
                "updated_at": NOW.isoformat(),
            }
        )


def test_orchestrator_artifact_cannot_pose_as_execution() -> None:
    with pytest.raises(ValidationError, match="orchestrator artifact kind cannot"):
        ArtifactRef(
            id="artifact_x",
            kind="input",
            producer=AgentOwner.ORCHESTRATOR,
            run_id="run_example",
            task_id="task_x",
            attempt_number=1,
            uri="file:///artifacts/x.json",
            sha256="0" * 64,
            media_type="application/json",
            summary="imported input",
            metadata={"source_type": "import"},
        )
