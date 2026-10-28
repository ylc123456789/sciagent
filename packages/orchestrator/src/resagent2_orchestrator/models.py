"""Persisted orchestration state for one research run."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from resagent2_components.literature import PreparedLiteratureImport

from resagent2_contracts import (
    ArtifactId,
    ArtifactImport,
    ArtifactRef,
    AgentResult,
    AgentRequest,
    BusinessInput,
    DatasetRef,
    ModuleError,
    PendingQuestion,
    QuestionId,
    RecordedAnswer,
    ResearchRequest,
    RunId,
    RunStatus,
    ScientificAssessment,
    ScientificOpinion,
    SessionRef,
    Workflow,
    WorkflowProposal,
    WorkflowPatch,
    WorkRequest,
    WorkRequestStatus,
    WorkspaceRecord,
    WorkResult,
    ExecutionDefinition, ExecutionPreset, WorkflowAgentRegistry, WorkflowAgentKind,
)


class OrchestratorModel(BaseModel):
    """Base for strict orchestrator-owned state."""

    model_config = ConfigDict(extra="forbid")


class CompletionViolationCode(StrEnum):
    """Stable categories for deterministic completion failures."""

    INVALID_SESSION = "invalid_session"
    ACTIVE_CONTROL_STATE = "active_control_state"
    INVALID_OPINION = "invalid_opinion"
    MISSING_EVIDENCE_KIND = "missing_evidence_kind"
    REQUIRED_ARTIFACT_MISSING = "required_artifact_missing"
    MISSING_LIMITATIONS = "missing_limitations"
    INCONSISTENT_TASK_RESULT = "inconsistent_task_result"


class CompletionViolation(OrchestratorModel):
    """One persisted machine-labelled reason a Run cannot complete."""

    code: CompletionViolationCode
    message: str = Field(min_length=1)
    subject: str | None = None
    related_ids: list[str] = Field(default_factory=list)


class RunUsage(OrchestratorModel):
    """Durable reservations; unknown outcomes remain consumed after recovery."""

    requests: dict[str, Literal["succeeded", "failed", "unknown"]] = Field(default_factory=dict)

    @property
    def used(self) -> int:
        return len(self.requests)

    @property
    def outcomes(self) -> dict[str, int]:
        return {outcome: sum(value == outcome for value in self.requests.values())
                for outcome in ("succeeded", "failed", "unknown")}


class PendingRegistration(OrchestratorModel):
    """Allocated identity and expected frozen facts until registration commits."""

    ref: ArtifactRef
    candidate_path: str
    source_path: str | None = None
    await_receipt: bool = False


class RunInitialization(OrchestratorModel):
    """Original inputs and progress saved before the first file is frozen."""

    imports: list[ArtifactImport] = Field(default_factory=list)
    literature: list[PreparedLiteratureImport] = Field(default_factory=list)
    next_import: int = Field(default=0, ge=0)
    next_literature: int = Field(default=0, ge=0)
    complete: bool = False


class PendingModuleResult(OrchestratorModel):
    """Result already returned by a port, awaiting its receiving gate and registration."""

    request: AgentRequest
    result: AgentResult
    previous_artifact_ids: list[ArtifactId] = Field(default_factory=list)
    # The receiving gate's decision is distinct from the original port result.
    # Once saved, recovery completes its handoff without adjudicating it again.
    received_result: AgentResult | None = None
    # One logical completion candidate, stable across receiving interruptions.
    # A replacement result after new input must not inherit its allocation.
    final_report_key: str | None = None


class PendingCancellation(OrchestratorModel):
    """One exact subtree cancellation retained until all its cleanup commits."""

    operation_id: str = Field(min_length=1)
    work_request_id: str | None = None
    reason: str = Field(min_length=1)
    source: Literal["interrupt", "abandon", "owner_failure"]


class ResearchRun(OrchestratorModel):
    """Complete persisted state owned by the Research Orchestrator."""

    run_id: RunId
    request: ResearchRequest
    # Catalog references discovered by the system, not datasets actually used.
    dataset_refs: list[DatasetRef] = Field(default_factory=list)
    status: RunStatus
    workflow: Workflow | None = None
    execution_definitions: dict[str, ExecutionDefinition] = Field(default_factory=dict)
    execution_presets: list[ExecutionPreset] = Field(default_factory=list)
    workspaces: dict[str, WorkspaceRecord] = Field(default_factory=dict)
    artifacts: dict[ArtifactId, ArtifactRef] = Field(default_factory=dict)
    pending_question: PendingQuestion | None = None
    child_questions: dict[str, PendingQuestion] = Field(default_factory=dict)
    upstream_inputs: list[BusinessInput] = Field(default_factory=list)
    task_inputs: dict[str, list[BusinessInput]] = Field(default_factory=dict)
    consumed_control_ids: list[str] = Field(default_factory=list)
    delivered_input_ids: list[str] = Field(default_factory=list)
    runtime_notices: list[dict] = Field(default_factory=list)
    parent_attention: bool = False
    attention_task_ids: list[str] = Field(default_factory=list)
    manually_paused_at: datetime | None = None
    conclusion_requirements_ref: ArtifactRef | None = None
    dataset_catalog_ref: ArtifactRef | None = None
    initialization: RunInitialization = Field(default_factory=RunInitialization)
    pending_registrations: dict[str, PendingRegistration] = Field(default_factory=dict)
    pending_module_results: dict[str, PendingModuleResult] = Field(default_factory=dict)
    pending_cancellation: PendingCancellation | None = None
    pending_compilation: WorkflowProposal | WorkflowPatch | None = None
    received_controls: dict[str, str] = Field(default_factory=dict)
    scientific_artifact_ids: list[ArtifactId] = Field(default_factory=list)
    scientific_report: str = ""
    answers: list[RecordedAnswer] = Field(default_factory=list)
    task_input_order: dict[str, list[str]] = Field(default_factory=dict)
    workflow_history: list[Workflow] = Field(default_factory=list)
    scientific_session: SessionRef | None = None
    latest_scientific_assessment: ScientificAssessment | None = None
    work_requests: list[WorkRequest] = Field(default_factory=list)
    final_opinion: ScientificOpinion | None = None
    final_report_artifact_id: ArtifactId | None = None
    delivered_answer_ids: list[QuestionId] = Field(default_factory=list)
    usage: RunUsage = Field(default_factory=RunUsage)
    # Settled user waits; the currently open question/manual pause is derived below.
    user_wait_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    completion_violations: list[CompletionViolation] = Field(default_factory=list)
    terminal_error: ModuleError | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def llm_calls_used(self) -> int:
        return self.usage.used

    @property
    def user_wait_started_at(self) -> datetime | None:
        """Start of the open user-wait interval, counted once if reasons overlap."""
        if self.status != RunStatus.PAUSED:
            return None
        starts = [self.manually_paused_at] if self.manually_paused_at is not None else []
        if self.pending_question is not None:
            starts.append(self.pending_question.created_at)
        return min(starts) if starts else None

    def remaining_timeout_seconds(self, now: datetime) -> float:
        """One Run clock for Controller and Scheduler: wall time minus user wait.

        Installation, provider latency and ordinary process downtime still count.
        No user-supplied answer timestamp is used to extend this budget.
        """
        waited = self.user_wait_seconds
        started = self.user_wait_started_at
        if started is not None:
            waited += max(0.0, (now - started).total_seconds())
        elapsed = max(0.0, (now - self.created_at).total_seconds() - waited)
        return max(0.0, self.request.budget.timeout_seconds - elapsed)

    def work(self, work_id):
        return next(item for item in self.work_requests if item.id == work_id)

    def task(self, task_id):
        if self.workflow is None:
            raise ValueError("run has no task graph")
        return next(item for item in self.workflow.tasks if item.id == task_id)

    def active_child(self, task_id, attempt_number=None):
        return next((work for work in self.work_requests
            if work.owner_task_id == task_id
            and (attempt_number is None or work.owner_attempt_number == attempt_number)
            and not work.result_delivered), None)

    def append_input(self, task_id, value):
        entries = self.task_inputs.setdefault(task_id, [])
        if not any(entry.event_id == value.event_id for entry in entries):
            entries.append(value)
            self.task_input_order.setdefault(task_id, []).append(value.event_id)

    def append_answer(self, answer):
        self.answers.append(answer)
        if answer.action is None and answer.task_id is not None:
            self.task_input_order.setdefault(answer.task_id, []).append(answer.question_id)

    def input_ids(self, task_id, attempt_number=None):
        # Reception order is persisted; caller-controlled timestamps cannot
        # insert a newly received answer before an already projected input.
        identities = [event.event_id for event in self.task_inputs.get(task_id, [])]
        identities.extend(answer.question_id for answer in self.answers
            if answer.task_id == task_id and answer.action is None
            and (attempt_number is None or answer.attempt_number == attempt_number))
        available = set(identities)
        ordered = [identity for identity in self.task_input_order.get(task_id, []) if identity in available]
        return list(dict.fromkeys([*ordered, *identities]))

    def needs_parent_processing(self, work):
        return work.owner_task_id is not None and self.input_ids(
            work.owner_task_id, work.owner_attempt_number) != work.processed_input_ids

    def owner_chain_blocked(self, task_id):
        task = self.task(task_id)
        seen = set()
        while True:
            work = self.work(task.work_request_id)
            if work.id in seen:
                raise ValueError("cycle in work ownership")
            seen.add(work.id)
            if self.needs_parent_processing(work):
                return True
            if work.owner_task_id is None:
                return False
            task = self.task(work.owner_task_id)

    @model_validator(mode="after")
    def validate_active_work_requests(self) -> "ResearchRun":
        WorkflowAgentRegistry(execution_definitions=self.execution_definitions, presets=self.execution_presets)
        if self.workflow is not None:
            for task in self.workflow.tasks:
                if task.workflow_agent_kind == WorkflowAgentKind.EXECUTOR and task.execution_ref not in self.execution_definitions:
                    raise ValueError("task execution configuration is not frozen in this Run")
        owners = set()
        roots = []
        work_ids = {work.id for work in self.work_requests}
        if len(work_ids) != len(self.work_requests):
            raise ValueError("duplicate WorkRequest identity")
        for work in self.work_requests:
            if work.owner_task_id is None:
                roots.append(work)
                continue
            if self.workflow is not None:
                owner = self.task(work.owner_task_id)
                containing = self.work(owner.work_request_id)
                if work.depth != containing.depth + 1:
                    raise ValueError("child work depth does not match its owner")
            if work.depth > self.request.execution_limits.max_work_depth:
                raise ValueError("work exceeds the Run depth limit")
            if not work.result_delivered:
                key = (work.owner_task_id, work.owner_attempt_number)
                if key in owners:
                    raise ValueError("an Attempt may own only one active delegation")
                owners.add(key)
        if len(roots) > 1:
            raise ValueError("a Run may have only one root WorkRequest")
        return self
