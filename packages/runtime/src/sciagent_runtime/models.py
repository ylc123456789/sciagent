"""Internal runtime models used by the shared Agentic Loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from sciagent_contracts import (
    SCHEMA_VERSION,
    AgentOwner,
    AgentResult,
    ActionSnapshot,
    ArtifactId,
    ArtifactRef,
    ArtifactOutput,
    ControlSignal,
    ModuleError,
    ModuleStatus,
    QuestionDraft,
    RunId,
    SessionId,
    SessionStatus,
    TaskId,
    WarningRecord,
)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RuntimeModel(BaseModel):
    """Base for strict runtime-internal structured values."""

    model_config = ConfigDict(extra="forbid")


class AgentAction(RuntimeModel):
    """One typed Tool request selected by an LLM."""

    tool: NonEmptyStr
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ContextSection(RuntimeModel):
    """One named, losslessly retained part of the fixed context."""

    name: NonEmptyStr
    content: str


class ComposedContext(RuntimeModel):
    """Complete fixed input, protocol summary and capacity measurement."""

    text: str
    protocol_mode: Literal["agent", "output"] = "agent"
    system_prompt: str = ""
    task_context: str = ""
    runtime_context: str = ""  # Derived execution facts, separate from business requirements.
    material_index: str = ""
    pending_notifications: str = ""
    runtime_status: str = ""
    history_summary: str | None = None
    included_sections: list[str]
    omitted_sections: list[str] = Field(default_factory=list, max_length=0)
    estimated_tokens: int = Field(ge=0)


class FinishCandidate(RuntimeModel):
    """LLM-proposed result that still requires deterministic finalization."""

    status: Literal[ModuleStatus.COMPLETED, ModuleStatus.FAILED] = Field(
        default=ModuleStatus.COMPLETED,
        description="Whether the assigned work is completed or cannot be completed. "
        "Explain the judgment in report; actual execution records remain authoritative.",
    )
    report: Annotated[str, StringConstraints(min_length=1)]
    artifacts: list[ArtifactOutput] = Field(default_factory=list)


class ToolObservation(RuntimeModel):
    """Normalized output from one Tool without direct state mutation.

    ``ok`` is a machine-readable success flag: it is True for a successful
    read/command and False for a failed command (non-zero exit), rejected
    argument, missing path, or other recoverable failure. Downstream code must
    not infer failure by parsing ``summary`` text.
    """

    summary: NonEmptyStr
    value: JsonValue | None = None
    user_comment: str | None = None
    ok: bool = True
    memory_updates: dict[str, JsonValue] = Field(default_factory=dict)
    finish_candidate: FinishCandidate | None = None
    question: QuestionDraft | None = None
    work_update: dict[str, JsonValue] | None = None
    abandon_work_id: str | None = None
    control_reason: str | None = None
    request_work: JsonValue | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_control_signal(self) -> "ToolObservation":
        signals = (self.finish_candidate, self.question, self.request_work, self.work_update, self.abandon_work_id)
        if sum(signal is not None for signal in signals) > 1:
            raise ValueError(
                "a tool observation can carry at most one of "
                "finish_candidate, question, and request_work"
            )
        return self


class CompletionDecision(RuntimeModel):
    """Fact validation of a finish candidate.

    complete=True accepts the candidate; the loop applies its proposed
    terminal status. failure reports a deterministic technical error.
    Neither means keep working with the supplied corrective feedback.
    Acceptance and a technical failure are mutually exclusive.
    """

    complete: bool
    report: str = ""
    artifacts: list[ArtifactOutput] = Field(default_factory=list)
    warnings: list[WarningRecord] = Field(default_factory=list)
    failure: ModuleError | None = None

    @model_validator(mode="after")
    def validate_failure_consistency(self) -> "CompletionDecision":
        if self.complete and self.failure is not None:
            raise ValueError("a completed decision cannot carry a failure")
        return self


class ApprovalReviewVerdict(RuntimeModel):
    """Untrusted model judgment; the runtime binds it to the prepared operation."""

    reason: NonEmptyStr = Field(description=(
        "Briefly state the supplied evidence for actual effects and any material unknowns, "
        "then assess authorization and scope. User intent is not implementation evidence."
    ))
    decision: Literal["allow", "ask"] = Field(description=(
        "allow only when actual effects are established and authorization and scope are clear; "
        "ask when either is unresolved."
    ))


@dataclass(frozen=True)
class ApprovalReviewResult:
    """Host-produced evidence binding; never an LLM output schema or a grant."""

    verdict: ApprovalReviewVerdict
    context: dict[str, JsonValue]


class PermissionDecision(RuntimeModel):
    """Permission result produced before a Tool can execute."""

    outcome: Literal["allow", "ask", "deny"]
    reason: str = ""
    context: dict[str, JsonValue] = Field(default_factory=dict)
    prepared: dict[str, JsonValue] | None = None
    approval_id: str | None = None
    user_comment: str | None = None
    reviewed_action: ActionSnapshot | None = None


class AgentEvent(RuntimeModel):
    """Append-only audit event persisted inside an Agent session snapshot."""

    sequence: int = Field(ge=1)
    step: int = Field(ge=0)
    type: Literal["action", "observation", "error", "compaction", "permission_review"]
    tool: str | None = None
    data: JsonValue
    created_at: datetime


class NativeToolCall(RuntimeModel):
    """Provider call identity and raw arguments; validation precedes execution."""

    id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    arguments: str


class ToolCallTurn(RuntimeModel):
    """One assistant reply and its paired tool receipts, owned by one Session.

    Reasoning is protocol continuation, not evidence. Raw arguments may be
    invalid JSON; keeping a rejected call with its error receipt never executes it.
    An absent receipt means an interrupted/unfinished call, not a successful one.
    """

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[NativeToolCall] = Field(default_factory=list)
    tool_results: dict[str, str] = Field(default_factory=dict)
    tool_result_files: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    # Checkpointed immediately before dispatch. Only this missing receipt has
    # an unknown outcome; other unreceipted calls have not started.
    executing_call_id: str | None = None
    executing_user_comment: str | None = None
    protocol_error: str | None = None
    runtime_notice: str | None = None

    @model_validator(mode="after")
    def validate_call_ids(self) -> "ToolCallTurn":
        ids = [call.id for call in self.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError("native tool call IDs must be unique within a reply")
        if not self.tool_result_files.keys() <= self.tool_results.keys():
            raise ValueError("tool-result files must refer to completed receipts")
        if not self.tool_results.keys() <= set(ids):
            raise ValueError("tool receipt must refer to a call in this reply")
        if self.executing_call_id is not None and (
            self.executing_call_id not in ids or self.executing_call_id in self.tool_results
        ):
            raise ValueError("executing call must be an unreceipted call in this reply")
        if self.executing_user_comment is not None and self.executing_call_id is None:
            raise ValueError("executing user comment requires an executing call")
        return self


class HistoryCheckpoint(RuntimeModel):
    """Lossy navigation recap and absolute boundary; original turns stay intact."""

    history_start: int = Field(ge=1)
    summary: NonEmptyStr


class AgentState(RuntimeModel):
    """Persisted generic state owned by one child Agent session."""

    schema_version: Literal["32.0"] = SCHEMA_VERSION
    session_id: SessionId
    upstream_session_id: SessionId | None = None
    work_request_id: str | None = None
    provided_input_ids: list[str] = Field(default_factory=list)
    work_depth: int = Field(default=1, ge=1)
    max_work_depth: int = Field(default=3, ge=1)
    agent_name: NonEmptyStr
    owner: AgentOwner
    run_id: RunId
    task_id: TaskId | None = None
    attempt_number: int | None = Field(default=None, ge=1)
    status: SessionStatus = SessionStatus.ACTIVE
    step: int = Field(default=0, ge=0)
    llm_calls_used: int = Field(default=0, ge=0)
    memory: dict[str, JsonValue] = Field(default_factory=dict)
    last_observation: ToolObservation | None = None
    events: list[AgentEvent] = Field(default_factory=list)
    # Native continuation requires this identity and is bound to the
    # client's protocol/configuration identity, never credentials.
    tool_protocol_key: NonEmptyStr | None = None
    contract_fingerprint: NonEmptyStr | None = None
    execution_ref: NonEmptyStr | None = None
    execution_config_hash: NonEmptyStr | None = None
    authorized_artifact_ids: list[ArtifactId] = Field(default_factory=list)
    pending_control: ControlSignal | None = None
    pending_control_result: AgentResult | None = None
    control_history: list[ControlSignal] = Field(default_factory=list)
    current_operation_key: str | None = None
    terminal_result: AgentResult | None = None
    tool_turns: list[ToolCallTurn] = Field(default_factory=list)
    tool_result_dir: str | None = None
    history_checkpoint: HistoryCheckpoint | None = None
    pending_action: ActionSnapshot | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_pending_control_result(self):
        if (self.execution_ref is None) != (self.execution_config_hash is None):
            raise ValueError("Session execution ref and config hash must be paired")
        if self.pending_control_result is not None and (
                self.pending_control is None or self.pending_control_result.control != self.pending_control
                or self.pending_control_result.session is None
                or self.pending_control_result.session.id != self.session_id):
            raise ValueError("saved public control result does not match the pending control")
        return self

    @model_validator(mode="after")
    def validate_history_checkpoint(self) -> "AgentState":
        if self.terminal_result is not None:
            result = self.terminal_result
            if (self.status not in {SessionStatus.COMPLETED, SessionStatus.FAILED}
                    or result.session is None or result.session.id != self.session_id
                    or result.session.module != self.owner or result.session.status != self.status):
                raise ValueError("terminal result must match the owning terminal Session")
        checkpoint = self.history_checkpoint
        if checkpoint is not None:
            if checkpoint.history_start >= len(self.tool_turns):
                raise ValueError("history checkpoint must retain a recent complete turn")
            if any(
                turn.executing_call_id is not None
                or len(turn.tool_results) != len(turn.tool_calls)
                for turn in self.tool_turns[:checkpoint.history_start]
            ):
                raise ValueError("history checkpoint cannot include unfinished tool calls")
        return self
