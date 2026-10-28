"""Shared Agentic Loop and injectable profile definition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Callable, Literal, Protocol
from uuid import uuid4

from pydantic import ValidationError

from resagent2_contracts import (
    AgentOwner, AgentCheckpoint,
    ActionSnapshot,
    AgentPermissions,
    ArtifactOutput,
    ControlSignal,
    ErrorCode,
    ModuleError,
    AgentResult,
    ModuleStatus,
    AgentRequest,
    RunId,
    SessionId,
    SessionRef,
    SessionStatus,
    TaskBudget,
    TaskId,
    QuestionDraft, PendingQuestion, ScientificAssessment, WorkRequest, WorkRequestDraft,
)

from .cancellation import OperationCancelled, check_cancelled
from .lifecycle import boundary_requested
from .context import (
    DEFAULT_AGENT_CONTEXT_TOKENS, ContextBudgetExceeded, ContextComposer,
)
from .budget import (
    BudgetExhaustedError, DeadlineExceededError, UsagePersistenceError, current_budget, current_budget_status,
    execution_budget, invoke_model,
)
from .compaction import plan_compaction
from .llm import LLMClient, LLMExhaustedError
from .models import (
    AgentAction,
    AgentEvent,
    AgentState,
    ApprovalReviewResult,
    ApprovalReviewVerdict,
    CompletionDecision,
    ContextSection,
    FinishCandidate,
    HistoryCheckpoint,
    PermissionDecision,
    ToolCallTurn,
    ToolObservation,
)
from .store import InMemorySessionStore, SessionLoadError, SessionStore
from .tools import Tool, ToolRegistry
from .tool_output import sync_tool_receipt_files
from .tool_calling import (
    NativeToolCallError, complete_pending_turn, native_actions,
    contract_fingerprint, native_input_text, native_tool_schemas, tool_receipt,
    record_tool_result, recover_pending_turn,
)

_CONSECUTIVE_FAILURE_LIMIT = 5


class ContextBuilder(Protocol):
    """Build Agent-specific context sections from request and generic state."""

    def __call__(
        self,
        request: Any,
        state: AgentState,
        max_context_tokens: int,
    ) -> list[ContextSection]:
        """Pack sections against the effective module/model input limit."""


class LoopRequest(Protocol):
    """The request surface the AgentLoop reads directly.

    AgentRequest supplies this surface for task-scoped and run-scoped Agents.
    The loop leaves materials and operation grants to the injected context
    builder and permission policy.
    """

    run_id: RunId
    task_id: TaskId | None
    attempt_number: int | None
    budget: TaskBudget
    parent_session_id: SessionId | None
    permissions: AgentPermissions


def _resume_binding_matches(state: AgentState, request: LoopRequest, *, owner: AgentOwner) -> bool:
    frozen = getattr(request, "execution_definition", None)
    return (
        state.session_id, state.run_id, state.task_id, state.attempt_number,
        state.owner, state.execution_ref, state.execution_config_hash,
    ) == (
        request.parent_session_id, request.run_id, request.task_id, request.attempt_number,
        owner, frozen.ref if frozen is not None else None,
        frozen.config_hash if frozen is not None else None,
    )


class CompletionCheck(Protocol):
    """Agent-specific deterministic completion boundary."""

    def evaluate(
        self,
        state: AgentState,
        candidate: FinishCandidate | None,
    ) -> CompletionDecision:
        """Validate the candidate and execution constraints, not task semantics."""


class PermissionPolicy(Protocol):
    """Check whether one validated action may execute in the current task."""

    def check(
        self,
        action: AgentAction,
        state: AgentState,
        request: Any,
    ) -> PermissionDecision:
        """Return a structured allow, ask or deny decision."""


class OperationReviewer(Protocol):
    """Stateless model review, called only after deterministic permission checks.

    Implementations use the shared budget, cancellation and model transport;
    bounded inspection uses already granted reads without recursive review;
    implementations cannot perform the proposed operation or manufacture user answers.
    """

    def review(self, operation: ActionSnapshot, request: AgentRequest) -> ApprovalReviewVerdict | ApprovalReviewResult: ...


class AllowListPermissionPolicy:
    """Minimal policy that permits only an explicit set of Tool names."""

    def __init__(self, allowed_tools: set[str]) -> None:
        self._allowed_tools = frozenset(allowed_tools)

    def check(
        self,
        action: AgentAction,
        state: AgentState,
        request: AgentRequest,
    ) -> PermissionDecision:
        if action.tool in self._allowed_tools:
            return PermissionDecision(outcome="allow")
        return PermissionDecision(
            outcome="deny",
            reason=f"tool {action.tool!r} is not allowed by this Agent profile",
        )


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Injected differences that let several Agents share one AgentLoop."""

    name: str
    owner: AgentOwner
    system_prompt: str
    tools: tuple[Tool, ...]
    llm_client: LLMClient
    context_builder: ContextBuilder
    permission_policy: PermissionPolicy
    completion_check: CompletionCheck
    max_context_tokens: int = DEFAULT_AGENT_CONTEXT_TOKENS
    execution_ref: str | None = None
    execution_config_hash: str | None = None
    acknowledge_registration: Callable[..., None] | None = None
    recover_registration: Callable[..., list] | None = None
    operation_reviewer: OperationReviewer | None = None
    finalize_result: Callable[[AgentResult, AgentState], AgentResult] | None = None
    tool_result_dir: str | None = None


class AgentLoop:
    """Run typed actions with validation, permission, persistence and finalization."""

    def __init__(
        self,
        *,
        store: SessionStore | None = None,
        context_composer: ContextComposer | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.store = store or InMemorySessionStore()
        self.context_composer = context_composer or ContextComposer()
        self.clock = clock
        self._run_llm_calls = 0
        self._active_call_id: str | None = None
        self._tool_result_memo: dict = {}
        self._result_finalizer: Callable[[AgentResult, AgentState], AgentResult] | None = None

    def validate_resume_binding(self, request: LoopRequest, *, owner: AgentOwner) -> None:
        """Check persisted identity before Agent preparation, without changing a Session.

        This precheck covers task ownership and frozen configuration only. The
        full run gate still validates status, provider protocol, tools and control.
        """
        if request.parent_session_id is None:
            return
        if not self.store.exists(request.parent_session_id):
            raise ValueError("cannot resume unknown session")
        state = self.store.load(request.parent_session_id)
        if not _resume_binding_matches(state, request, owner=owner):
            raise ValueError("resume target does not match request identity or frozen execution configuration")

    def read_checkpoint_result(
        self, *, session_id: SessionId, run_id: RunId, task_id: TaskId,
        attempt_number: int, owner: AgentOwner,
    ) -> AgentCheckpoint | None:
        """Read saved public facts and their original input boundary without execution."""
        if not self.store.exists(session_id):
            return None
        state = self.store.load(session_id)
        if (state.session_id, state.run_id, state.task_id, state.attempt_number, state.owner) != (
            session_id, run_id, task_id, attempt_number, owner,
        ):
            raise ValueError("saved terminal result does not belong to this invocation")
        result = None
        if state.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
            result = state.terminal_result
        elif state.status == SessionStatus.PAUSED and state.pending_control is not None:
            result = state.pending_control_result
            if result is None:
                result = self._control_result(state, llm_calls=0)
        if result is None:
            return None
        return AgentCheckpoint(result=result.model_copy(deep=True, update={"llm_calls": 0}),
                               provided_input_ids=list(state.provided_input_ids))

    def read_terminal_result(
        self, *, session_id: SessionId, run_id: RunId, task_id: TaskId,
        attempt_number: int, owner: AgentOwner,
    ) -> AgentResult | None:
        """Read one saved terminal outcome without resuming or preparing execution."""
        checkpoint = self.read_checkpoint_result(session_id=session_id, run_id=run_id,
            task_id=task_id, attempt_number=attempt_number, owner=owner)
        if checkpoint is None or checkpoint.result.status not in {
                ModuleStatus.COMPLETED, ModuleStatus.COMPLETED_WITH_WARNINGS, ModuleStatus.FAILED}:
            return None
        return checkpoint.result

    def cancel_question(self, session_id: str, question_id: str, reason: str) -> None:
        state = self.store.load(session_id)
        control = state.pending_control
        if control is not None and control.action == "ask_user" and control.question.id == question_id:
            if control.question.action is not None:
                raise ValueError("business escalation cannot cancel a permission request")
            state.pending_control = None
            state.pending_control_result = None
            state.tool_turns.append(ToolCallTurn(runtime_notice=reason + "; question_id=" + question_id))
            self._save(state)

    def defer_completion(self, session_id: str, reason: str) -> None:
        """The owner received new input before committing Run completion."""
        state = self.store.load(session_id)
        if state.status not in {SessionStatus.COMPLETED, SessionStatus.PAUSED}:
            raise ValueError("completion receiving does not match this Session")
        state.status = SessionStatus.PAUSED
        state.terminal_result = None
        state.tool_turns.append(ToolCallTurn(runtime_notice=reason))
        self._save(state)

    def terminate_session(self, session_id: str, reason: str) -> None:
        if not self.store.exists(session_id):
            return
        state = self.store.load(session_id)
        if state.status in {SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED}:
            return
        for turn in state.tool_turns:
            if turn.executing_call_id is not None:
                record_tool_result(turn, turn.executing_call_id, tool_receipt(ToolObservation(
                    ok=False, summary="Delegation ended before the started operation had a durable result; execution outcome is unknown",
                    value={"execution_status": "unknown"}, user_comment=turn.executing_user_comment,
                )))
            complete_pending_turn(turn, tool_receipt(ToolObservation(ok=False,
                summary="Not executed: delegation ended", value={"execution_status": "not_executed"})))
        state.pending_control = None
        state.pending_control_result = None
        state.pending_action = None
        state.tool_turns.append(ToolCallTurn(runtime_notice=reason + "; this child session is terminal and will not resume"))
        state.status = SessionStatus.CANCELLED
        self._save(state)

    def run(
        self,
        definition: AgentDefinition,
        request: LoopRequest,
        *,
        session_id: str,
        initial_memory: dict | None = None,
    ) -> AgentResult:
        """Run one new Agent session until completion, pause, or structured failure."""

        with execution_budget(max_llm_calls=request.budget.max_llm_calls,
                              timeout_seconds=request.budget.timeout_seconds, clock=self.clock):
            self._state = None
            self._result_finalizer = definition.finalize_result
            try:
                return self._run(definition, request, session_id=session_id, initial_memory=initial_memory)
            except OperationCancelled as error:
                if self._state is None:
                    raise
                self._sync_usage(self._state)
                if not error.stopped:
                    if self._active_call_id is not None:
                        self._record_observation(self._state, "runtime", ToolObservation(ok=False,
                            summary="Execution stop could not be confirmed; no further dispatch permitted",
                            value={"execution_status": "unknown", "stop_confirmed": False, **error.details}))
                    return self._failure(self._state, ErrorCode.INTERRUPTED,
                        "Cannot confirm managed execution stopped: " + str(error), retryable=False,
                        details={**error.details, "stop_confirmed": False})
                return self._suspend(self._state, interrupted=True, reason=str(error), details=error.details)
            finally:
                self._result_finalizer = None

    def _run(self, definition, request, *, session_id, initial_memory):

        self._run_llm_calls = 0
        self._tool_result_memo = {}
        self._initial_usage = current_budget().usage.used
        self._active_call_id = None
        now = datetime.now(UTC)
        frozen = getattr(request, "execution_definition", None)
        execution_ref = frozen.ref if frozen is not None else None
        execution_hash = frozen.config_hash if frozen is not None else None
        if (definition.execution_ref, definition.execution_config_hash) != (execution_ref, execution_hash):
            message = "execution definition does not match the frozen request configuration"
            return AgentResult(status=ModuleStatus.FAILED, report=message,
                error=ModuleError(code=ErrorCode.CONTRACT_ERROR, message=message, retryable=False))
        if not callable(getattr(definition.llm_client, "next_tool_call", None)):
            return AgentResult(
                status=ModuleStatus.FAILED,
                report="Agent client must implement native tool calls",
                error=ModuleError(
                    code=ErrorCode.CONTRACT_ERROR,
                    message="Agent client must implement native tool calls",
                    retryable=False,
                ),
            )
        protocol_key = getattr(definition.llm_client, "tool_session_key", None)
        if not isinstance(protocol_key, str) or not protocol_key.strip():
            return AgentResult(
                status=ModuleStatus.FAILED,
                report="native client must provide a stable tool_session_key",
                error=ModuleError(
                    code=ErrorCode.CONTRACT_ERROR,
                    message="native client must provide a stable tool_session_key",
                    retryable=False,
                ),
            )
        try:
            schemas = native_tool_schemas(definition.tools)
        except OperationCancelled:
            raise
        except Exception as error:
            return AgentResult(status=ModuleStatus.FAILED, report=str(error), error=ModuleError(
                code=ErrorCode.CONTRACT_ERROR, message="tool schema construction failed: " + str(error), retryable=False,
            ))
        fingerprint = contract_fingerprint(definition.system_prompt, schemas)
        resume_id = request.parent_session_id
        if resume_id is not None:
            if not self.store.exists(resume_id):
                error = ModuleError(
                    code=ErrorCode.CONTRACT_ERROR,
                    message="cannot resume unknown session",
                    retryable=False,
                )
                return AgentResult(
                    status=ModuleStatus.FAILED,
                    report=error.message,
                    error=error,
                )
            try:
                state = self.store.load(resume_id)
            except SessionLoadError as error:
                failure = ModuleError(
                    code=ErrorCode.CONTRACT_ERROR,
                    message=f"cannot resume session: {error}",
                    retryable=False,
                )
                return AgentResult(
                    status=ModuleStatus.FAILED,
                    report=failure.message,
                    error=failure,
                )
            if (
                state.status not in {SessionStatus.PAUSED, SessionStatus.ACTIVE, SessionStatus.COMPLETED, SessionStatus.FAILED}
                or not _resume_binding_matches(state, request, owner=definition.owner)
                or state.upstream_session_id != request.upstream_session_id
                or state.work_request_id != request.work_request_id
                or state.work_depth != request.work_depth
                or state.max_work_depth != request.max_work_depth
                or state.agent_name != definition.name
                or state.tool_protocol_key != protocol_key
                or state.contract_fingerprint != fingerprint
            ):
                error = ModuleError(
                    code=ErrorCode.CONTRACT_ERROR,
                    message=(
                        "resume target does not match request or is not "
                        "recoverable (including model/provider protocol identity)"
                    ),
                    retryable=False,
                )
                return AgentResult(
                    status=ModuleStatus.FAILED,
                    report=error.message,
                    error=error,
                )
            if state.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
                if state.terminal_result is None:
                    return AgentResult(status=ModuleStatus.FAILED, report="Terminal Session has no saved result",
                        error=ModuleError(code=ErrorCode.CONTRACT_ERROR,
                                          message="Terminal Session has no saved result", retryable=False))
                return state.terminal_result.model_copy(update={"llm_calls": 0})
            if state.tool_result_dir not in {None, definition.tool_result_dir}:
                message = "Resumed tool-result directory differs from the bound Session"
                return AgentResult(status=ModuleStatus.FAILED, report=message,
                    error=ModuleError(code=ErrorCode.CONTRACT_ERROR, message=message, retryable=False))
            state.tool_result_dir = definition.tool_result_dir
            state.status = SessionStatus.ACTIVE
            state.updated_at = now
        else:
            state = AgentState(
                session_id=session_id,
                agent_name=definition.name,
                owner=definition.owner,
                upstream_session_id=request.upstream_session_id,
                work_request_id=request.work_request_id,
                work_depth=request.work_depth, max_work_depth=request.max_work_depth,
                run_id=request.run_id,
                task_id=request.task_id,
                attempt_number=request.attempt_number,
                memory=initial_memory or {},
                tool_protocol_key=protocol_key, contract_fingerprint=fingerprint,
                tool_result_dir=definition.tool_result_dir,
                execution_ref=execution_ref, execution_config_hash=execution_hash,
                created_at=now,
                updated_at=now,
            )
            if self.store.exists(state.session_id):
                error = ModuleError(
                    code=ErrorCode.CONTRACT_ERROR,
                    message="session_id already exists",
                    retryable=False,
                )
                return AgentResult(
                    status=ModuleStatus.FAILED,
                    report=error.message,
                    error=error,
                )
            self._save(state)

        try:
            registry = ToolRegistry(definition.tools)
        except ValueError as error:
            return self._failure(
                state,
                ErrorCode.CONTRACT_ERROR,
                str(error),
                retryable=False,
            )

        self._state = state
        self._accept_notices(request, state)
        check_cancelled()
        if state.pending_action is not None and (state.provided_input_ids != request.input_ids or boundary_requested()):
            self._settle_outdated_operation(request, state)
        resumed = self._resume_control(definition, registry, request, state)
        if resumed is not None:
            return resumed
        interrupted = False
        for ordinal, turn in enumerate(state.tool_turns):
            if definition.acknowledge_registration is not None:
                # The Session receipt is the authority for this association.
                # A previous process may have stopped immediately before ack.
                for call_id in turn.tool_results:
                    definition.acknowledge_registration(
                        run_id=state.run_id,
                        operation_key=f"session:{state.session_id}:turn:{ordinal}:call:{call_id}",
                    )
            if len(turn.tool_results) < len(turn.tool_calls):
                operation_key = None
                if turn.executing_call_id is not None and definition.recover_registration is not None:
                    call_id = turn.executing_call_id
                    operation_key = f"session:{state.session_id}:turn:{ordinal}:call:{call_id}"
                    recovered = definition.recover_registration(run_id=state.run_id, operation_key=operation_key)
                    if recovered:
                        for ref in recovered:
                            if ref.run_id != state.run_id or ref.session_id != state.session_id:
                                raise ValueError("Recovered registration belongs to another invocation")
                            if ref.id not in state.authorized_artifact_ids:
                                state.authorized_artifact_ids.append(ref.id)
                        record_tool_result(turn, call_id, tool_receipt(ToolObservation(
                            ok=False,
                            summary="Materials were registered before interruption; the original tool result was not saved",
                            value={"artifact_ids": [ref.id for ref in recovered], "execution_status": "interrupted"},
                            user_comment=turn.executing_user_comment,
                        )))
                recover_pending_turn(turn)
                self._save(state)
                if operation_key is not None and definition.acknowledge_registration is not None:
                    definition.acknowledge_registration(run_id=state.run_id, operation_key=operation_key)
                interrupted = True
        if interrupted:
            self._feedback(
                state, "An interrupted tool batch was not replayed. Inspect its receipts and current state before retrying.",
                tool="runtime",
            )

        consecutive_failures = 0
        while self._run_llm_calls < request.budget.max_llm_calls:
            check_cancelled()
            if boundary_requested():
                return self._suspend(state, interrupted=False, reason="New user input awaits the parent Agent at this model boundary")
            if current_budget().expired:
                return self._failure(
                    state,
                    ErrorCode.TIMEOUT,
                    "Agent session exceeded timeout",
                    retryable=True,
                )

            state.provided_input_ids = list(request.input_ids)
            self._save(state)
            tracer = getattr(definition.llm_client, "set_trace_context", None)
            if tracer is not None:
                tracer(run_id=request.run_id, session_id=state.session_id,
                       task_id=request.task_id, agent=definition.name, step=state.step)
            try:
                context_limit = definition.max_context_tokens
                budgeter = getattr(definition.llm_client, "tool_input_limit", None)
                if budgeter is not None:
                    context_limit = min(context_limit, budgeter(context_limit))
                if context_limit < 1:
                    raise ContextBudgetExceeded("effective context limit must be positive")
                context = None
                context_error = None
                try:
                    context = self._compose_context(
                        definition, request, state, schemas, context_limit,
                    )
                except ContextBudgetExceeded as error:
                    context_error = error
                if callable(getattr(definition.llm_client, "summarize_history", None)):
                    checkpoint = state.history_checkpoint
                    plan = plan_compaction(
                        current_context=self._fixed_context(definition, request, state, context_limit),
                        schemas=schemas, turns=state.tool_turns,
                        history_start=checkpoint.history_start if checkpoint else 0,
                        previous_summary=checkpoint.summary if checkpoint else None,
                        max_input_tokens=context_limit, force=context is None,
                    )
                    if plan is not None:
                        failure = self._compact_history(
                            definition, request, state, schemas, context_limit, plan,
                        )
                        if failure is not None:
                            return failure
                        context = self._compose_context(
                            definition, request, state, schemas, context_limit,
                        )
                if context is None:
                    raise context_error or ContextBudgetExceeded("current context does not fit")
            except (BudgetExhaustedError, DeadlineExceededError) as error:
                self._sync_usage(state)
                return self._failure(state,
                                     ErrorCode.TIMEOUT if isinstance(error, DeadlineExceededError) else ErrorCode.BUDGET_EXHAUSTED,
                                     str(error), retryable=False)
            except ContextBudgetExceeded as error:
                return self._failure(
                    state,
                    ErrorCode.BUDGET_EXHAUSTED,
                    str(error),
                    retryable=False,
                )
            except OperationCancelled:
                raise
            except Exception as error:
                return self._failure(
                    state,
                    ErrorCode.CONTRACT_ERROR,
                    f"context builder failed: {error}",
                    retryable=False,
                )

            if current_budget().expired:
                return self._failure(state, ErrorCode.TIMEOUT, "Agent session exceeded timeout", retryable=True)
            if boundary_requested():
                return self._yield_for_input(state)
            limiter = getattr(definition.llm_client, "set_attempt_limit", None)
            if limiter is not None:
                limiter(request.budget.max_llm_calls - self._run_llm_calls)
            charged = False
            try:
                check_cancelled()
                reply = invoke_model(definition.llm_client, "next_tool_call", context, schemas,
                                     self._active_turns(state), max_input_tokens=context_limit)
                self._sync_usage(state)
                charged = True
                if self._run_llm_calls > request.budget.max_llm_calls:
                    return self._failure(state, ErrorCode.BUDGET_EXHAUSTED,
                                         "Client exceeded the allocated LLM-call budget", retryable=False)
                check_cancelled()
                turn = ToolCallTurn.model_validate(reply)
                used_ids = {call.id for old in state.tool_turns for call in old.tool_calls}
                rejection = None
                if (turn.tool_results or turn.executing_call_id is not None
                        or turn.executing_user_comment is not None or turn.protocol_error is not None
                        or turn.runtime_notice is not None):
                    rejection = "A model reply cannot provide tool execution receipts"
                if any(call.id in used_ids for call in turn.tool_calls):
                    rejection = "Provider reused a historical tool call ID; no tool was executed"
                if rejection is not None:
                    validator = getattr(definition.llm_client, "record_validation", None)
                    if validator is not None:
                        validator(rejection)
                    raise NativeToolCallError(rejection)
                if turn.tool_calls or turn.content or turn.reasoning_content:
                    state.tool_turns.append(turn)
                    # Checkpoint before dispatch. Missing receipts on resume
                    # are unknown outcomes, never replay instructions.
                    self._save(state)
                if boundary_requested():
                    return self._yield_for_input(state)
                actions = [AgentAction.model_validate(raw) for raw in native_actions(turn)]
                if len(actions) > 1:
                    failure = self._preflight_batch(definition, registry, request, state, actions)
                    if failure is not None:
                        return failure
            except (json.JSONDecodeError, ValidationError, NativeToolCallError, PermissionError) as error:
                if not charged:
                    self._sync_usage(state)
                if isinstance(error, json.JSONDecodeError):
                    # Count preceding transport retries as well. Invalid argument
                    # JSON never executes; calls remain in paired protocol history.
                    summary = (
                        f"Native tool arguments were not valid JSON: {error}. "
                        "Return native tool calls with JSON objects of arguments. "
                        "No tool was executed."
                    )
                    details = None
                elif isinstance(error, PermissionError):
                    summary = f"Tool execution denied: {error}. No tool was executed."
                    details = None
                elif isinstance(error, NativeToolCallError):
                    summary = str(error) + " Correct the native tool batch. No tool was executed."
                    details = None
                else:
                    details = self._validation_details(error)
                    validator = getattr(definition.llm_client, "record_validation", None)
                    if validator is not None:
                        validator(str(details))
                    summary = "LLM action did not match the action schema: " + str(details)
                # A malformed action is recoverable: record it as durable
                # feedback and let the LLM correct it, bounded by the same
                # consecutive-failure limit as any other recoverable error.
                self._feedback(
                    state,
                    summary,
                    tool="llm",
                    value=details,
                )
                failure = self._note_failure(state, consecutive_failures)
                if failure is not None:
                    return failure
                consecutive_failures += 1
                continue
            except (BudgetExhaustedError, DeadlineExceededError) as error:
                self._sync_usage(state)
                return self._failure(state,
                                     ErrorCode.TIMEOUT if isinstance(error, DeadlineExceededError) else ErrorCode.BUDGET_EXHAUSTED,
                                     str(error), retryable=False)
            except ContextBudgetExceeded as error:
                return self._failure(
                    state, ErrorCode.BUDGET_EXHAUSTED, str(error), retryable=False,
                )
            except LLMExhaustedError as error:
                self._sync_usage(state)
                return self._failure(
                    state,
                    ErrorCode.TOOL_FAILED,
                    str(error),
                    retryable=False,
                    details={"component": "llm"},
                )
            except OperationCancelled:
                raise
            except Exception as error:
                # The transport failed after one or more real HTTP attempts;
                # those attempts still count toward the Run ledger.
                if not charged:
                    self._sync_usage(state)
                return self._failure(
                    state,
                    ErrorCode.TOOL_FAILED,
                    f"LLM call failed: {error}",
                    retryable=True,
                    details={"component": "llm"},
                )

            for index, action in enumerate(actions):
                check_cancelled()
                if boundary_requested():
                    return self._yield_for_input(state)
                self._active_call_id = turn.tool_calls[index].id
                state.current_operation_key = (
                    f"session:{state.session_id}:turn:{len(state.tool_turns) - 1}:call:{self._active_call_id}"
                )
                state.step += 1
                self._append_event(
                    state,
                    event_type="action",
                    tool=action.tool,
                    data=action.model_dump(mode="json"),
                )
                self._save(state)

                if not registry.contains(action.tool):
                    self._feedback(state, f"Unknown tool: {action.tool}. Use the provided tools.",
                                   tool=action.tool)
                    failure = self._note_failure(state, consecutive_failures)
                    if failure is not None:
                        return failure
                    consecutive_failures += 1
                    break

                try:
                    registry.validate(action.tool, action.arguments)
                    permission = definition.permission_policy.check(action, state, request)
                    permission = self._review_permission(definition, registry, request, state, action, permission)
                except BudgetExhaustedError as error:
                    return self._failure(state, ErrorCode.BUDGET_EXHAUSTED, str(error), retryable=False)
                except DeadlineExceededError as error:
                    return self._failure(state, ErrorCode.TIMEOUT, str(error), retryable=False)
                except (ValidationError, PermissionError, ValueError) as error:
                    self._feedback(state, f"Tool request rejected: {error}", tool=action.tool)
                    failure = self._note_failure(state, consecutive_failures)
                    if failure is not None:
                        return failure
                    consecutive_failures += 1
                    break
                except OperationCancelled:
                    raise
                except Exception as error:
                    return self._failure(
                        state,
                        ErrorCode.CONTRACT_ERROR,
                        f"permission policy failed: {error}",
                        retryable=False,
                    )
                check_cancelled()
                if boundary_requested():
                    return self._yield_for_input(state)
                if permission.outcome == "deny":
                    if permission.approval_id:
                        state.pending_action = None
                    self._feedback(state, permission.reason or "Tool execution denied", tool=action.tool,
                                   value={"execution_status": "not_executed"})
                    failure = self._note_failure(state, consecutive_failures)
                    if failure is not None:
                        return failure
                    consecutive_failures += 1
                    break

                # Model calls and permission checks can consume the remaining
                # deadline. Their completed work is already accounted for, but an
                # expired action must not start a new tool or its side effects.
                if current_budget().expired:
                    return self._failure(
                        state,
                        ErrorCode.TIMEOUT,
                        "Agent session exceeded timeout before tool dispatch",
                        retryable=True,
                    )

                try:
                    current_budget().remaining_timeout()
                    if permission.outcome == "ask":
                        self._pause_approval(registry, request, state, action, permission)
                        return self._control_result(state)
                    else:
                        if permission.approval_id:
                            if state.pending_action is None or state.pending_action.action_id != permission.approval_id:
                                raise PermissionError("approval is not pending")
                            state.pending_action = None
                        state.tool_turns[-1].executing_call_id = self._active_call_id
                        self._save(state)
                        observation = registry.dispatch(action.tool, action.arguments, state,
                                                        prepared=permission.prepared)
                except DeadlineExceededError as error:
                    return self._failure(state, ErrorCode.TIMEOUT, str(error), retryable=False)
                except ValidationError as error:
                    self._feedback(
                        state,
                        "Tool arguments did not match the input schema: "
                        + str(self._validation_details(error)),
                        tool=action.tool,
                    )
                    failure = self._note_failure(state, consecutive_failures)
                    if failure is not None:
                        return failure
                    consecutive_failures += 1
                    break
                except PermissionError as error:
                    self._feedback(
                        state,
                        f"Tool execution denied: {error}",
                        tool=action.tool,
                    )
                    failure = self._note_failure(state, consecutive_failures)
                    if failure is not None:
                        return failure
                    consecutive_failures += 1
                    break
                except OperationCancelled:
                    raise
                except Exception as error:
                    self._feedback(
                        state,
                        f"Tool execution failed: {error}",
                        tool=action.tool,
                    )
                    failure = self._note_failure(state, consecutive_failures)
                    if failure is not None:
                        return failure
                    consecutive_failures += 1
                    break

                state.memory.update(observation.memory_updates)
                for ref in observation.artifacts:
                    if ref.run_id != state.run_id or ref.session_id != state.session_id:
                        return self._failure(state, ErrorCode.CONTRACT_ERROR,
                                             "Tool artifact belongs to another invocation", retryable=False)
                    if ref.id not in state.authorized_artifact_ids:
                        state.authorized_artifact_ids.append(ref.id)
                state.last_observation = observation

                if any(value is not None for value in (observation.question, observation.request_work, observation.work_update, observation.abandon_work_id)):
                    if boundary_requested():
                        # These tools only propose a control handoff. No owner
                        # mutation has occurred, so an outdated proposal cannot
                        # become an accepted WorkRequest/update/question.
                        return self._yield_for_input(state)
                    if observation.request_work is not None and (
                        not request.permissions.request_work or request.work_depth >= request.max_work_depth
                        or request.task_context.get("active_work_request_id")
                    ):
                        return self._failure(state, ErrorCode.PERMISSION_DENIED,
                                             "This invocation may not request work", retryable=False)
                    try:
                        self._accept_control(state, request, observation)
                    except (PermissionError, ValueError) as error:
                        self._feedback(state, "Control request rejected: " + str(error), tool=action.tool)
                        failure = self._note_failure(state, consecutive_failures)
                        if failure is not None:
                            return failure
                        consecutive_failures += 1
                        break
                    check_cancelled()
                    return self._control_result(state)

                is_finish = observation.finish_candidate is not None
                if not is_finish:
                    self._record_observation(state, action.tool, observation)
                    self._acknowledge_registration(definition, state)
                    if observation.ok:
                        consecutive_failures = 0
                    else:
                        failure = self._note_failure(state, consecutive_failures)
                        if failure is not None:
                            return failure
                        consecutive_failures += 1
                    if current_budget().expired:
                        return self._failure(state, ErrorCode.TIMEOUT,
                                             "Agent session exceeded timeout", retryable=True)

                try:
                    decision = definition.completion_check.evaluate(
                        state,
                        observation.finish_candidate,
                    )
                except DeadlineExceededError as error:
                    return self._failure(state, ErrorCode.TIMEOUT, str(error), retryable=False)
                except OperationCancelled:
                    raise
                except Exception as error:
                    return self._failure(
                        state,
                        ErrorCode.CONTRACT_ERROR,
                        f"completion check failed: {error}",
                        retryable=False,
                    )

                if is_finish:
                    final_receipt = ToolObservation(
                        ok=decision.complete and decision.failure is None,
                        summary=decision.report or (
                            "Completion accepted" if decision.complete else "Completion rejected"
                        ),
                        value={"completion_check": "accepted" if decision.complete else "rejected",
                               "status": observation.finish_candidate.status.value},
                    )
                    self._record_observation(state, action.tool, final_receipt)

                if decision.failure is not None:
                    return self._failure(
                        state,
                        decision.failure.code,
                        decision.failure.message,
                        retryable=decision.failure.retryable,
                        details=decision.failure.details,
                        artifacts=decision.artifacts,
                        report=decision.report or decision.failure.message,
                    )

                check_cancelled()
                if decision.complete and boundary_requested():
                    self._feedback(state, "Completion deferred: new upstream input must be processed before ending this Run", tool="runtime")
                    return self._suspend(state, interrupted=False, reason="New input arrived during the final call")
                if decision.complete:
                    candidate = observation.finish_candidate
                    if candidate is not None and candidate.status == ModuleStatus.FAILED:
                        return self._failure(
                            state, ErrorCode.AGENT_REPORTED_FAILURE,
                            "Agent reported that the assigned work could not be completed",
                            retryable=False, artifacts=decision.artifacts,
                            report=decision.report or candidate.report,
                        )
                    self._cancel_pending_calls(state)
                    state.status = SessionStatus.COMPLETED
                    status = (
                        ModuleStatus.COMPLETED_WITH_WARNINGS
                        if decision.warnings
                        else ModuleStatus.COMPLETED
                    )
                    result = AgentResult(
                        status=status,
                        report=decision.report or "Completion check passed",
                        artifacts=decision.artifacts,
                        warnings=decision.warnings,
                        session=self._session_ref(state),
                        llm_calls=self._run_llm_calls,
                    )
                    return self._persist_terminal_result(state, result)
                if observation.finish_candidate is not None:
                    # A proposed finish was rejected: count it so a model that keeps
                    # proposing the same finish is eventually stopped.
                    failure = self._note_failure(state, consecutive_failures)
                    if failure is not None:
                        return failure
                    consecutive_failures += 1

                if not observation.ok:
                    break

            self._cancel_pending_calls(state)
            self._active_call_id = None

        return self._failure(
            state,
            ErrorCode.BUDGET_EXHAUSTED,
            "Agent session exhausted LLM-call budget",
            retryable=False,
            details=self._failure_details(state),
        )


    def _accept_notices(self, request, state):
        if not request.runtime_notices:
            return
        seen = set(state.memory.get("_received_notice_ids", []))
        for notice in request.runtime_notices:
            key = notice["event_id"]
            if key not in seen:
                state.tool_turns.append(ToolCallTurn(runtime_notice=json.dumps(notice, ensure_ascii=False)))
                seen.add(key)
        state.memory["_received_notice_ids"] = sorted(seen)
        self._save(state)

    def _suspend(self, state, *, interrupted, reason, details=None):
        if interrupted:
            control = state.pending_control
            control_ref = (f"; original {control.action} call={control.source_call_id} "
                           + (f"question_id={control.question.id}" if control.question else
                              f"work_request_id={control.work_request.id}" if control.work_request else "")) if control else ""
            if state.tool_turns:
                turn = self._current_turn(state)
                if turn.executing_call_id is not None and turn.executing_call_id not in turn.tool_results:
                    record_tool_result(turn, turn.executing_call_id, tool_receipt(ToolObservation(
                        ok=False, summary="Operation interrupted; managed execution has stopped; existing files retained",
                        value={"execution_status": "interrupted", "stop_confirmed": True, **(details or {})},
                        user_comment=turn.executing_user_comment)))
                complete_pending_turn(turn, tool_receipt(ToolObservation(ok=False,
                    summary="Not executed: user interrupted this batch", value={"execution_status": "not_executed"})))
            state.tool_turns.append(ToolCallTurn(runtime_notice=reason + control_ref + "; interrupted response will not dispatch tools; files and completed outcomes retained"))
            state.pending_control = None
            state.pending_control_result = None
            state.pending_action = None
        state.status = SessionStatus.PAUSED
        self._save(state)
        return AgentResult(status=ModuleStatus.INTERRUPTED if interrupted else ModuleStatus.YIELDED,
                           report=reason, session=self._session_ref(state), llm_calls=self._run_llm_calls)

    def _yield_for_input(self, state: AgentState) -> AgentResult:
        """Pair every unstarted call while retaining completed batch results."""
        if state.tool_turns:
            turn = self._current_turn(state)
            if turn.executing_call_id is not None and turn.executing_call_id not in turn.tool_results:
                # Control proposal tools have no side effects before acceptance.
                turn.executing_call_id = None
            complete_pending_turn(turn, tool_receipt(ToolObservation(
                ok=False, summary="Not executed: new upstream input must be processed first",
                value={"execution_status": "not_executed"},
            )))
        self._active_call_id = None
        return self._suspend(state, interrupted=False,
                             reason="New upstream input awaits processing at this execution boundary")

    @staticmethod
    def _active_turns(state: AgentState) -> list[ToolCallTurn]:
        start = state.history_checkpoint.history_start if state.history_checkpoint else 0
        return state.tool_turns[start:]

    def _fixed_context(self, definition, request, state, context_limit):
        sections = list(definition.context_builder(request, state, context_limit))
        context = self.context_composer.compose(definition.system_prompt, sections, max_tokens=context_limit)
        return context.model_copy(update={"runtime_status": current_budget_status(),
                                         "pending_notifications": json.dumps(
            [q.model_dump(mode="json") for q in request.child_questions], ensure_ascii=False
        ) if request.child_questions else ""})

    def _compose_context(self, definition, request, state, schemas, context_limit):
        context = self._fixed_context(definition, request, state, context_limit)
        if state.history_checkpoint is not None:
            context = context.model_copy(update={"history_summary": state.history_checkpoint.summary})
        tokens = self.context_composer.estimate_tokens(
            native_input_text(context, schemas, self._active_turns(state))
        )
        if tokens > context_limit:
            raise ContextBudgetExceeded("complete required context and native history exceed input budget")
        return context.model_copy(update={"estimated_tokens": tokens})

    def _compact_history(self, definition, request, state, schemas, context_limit, plan) -> AgentResult | None:
        """Charge one bounded summary request and atomically publish a usable checkpoint."""
        client = definition.llm_client
        remaining = request.budget.max_llm_calls - self._run_llm_calls
        if remaining < 2:
            return self._failure(state, ErrorCode.BUDGET_EXHAUSTED,
                                 "Insufficient calls for compaction and continuation", retryable=False)
        limiter = getattr(client, "set_attempt_limit", None)
        if limiter is not None:
            limiter(remaining - 1)
        try:
            check_cancelled()
            summary = invoke_model(client, "summarize_history", plan.prompt, max_input_tokens=context_limit)
            check_cancelled()
        except (BudgetExhaustedError, DeadlineExceededError):
            raise
        except OperationCancelled:
            raise
        except Exception as error:
            self._sync_usage(state)
            return self._failure(state, ErrorCode.TOOL_FAILED,
                                 f"History compaction failed: {error}", retryable=False,
                                 details={"component": "compaction"})
        self._sync_usage(state)
        if self._run_llm_calls >= request.budget.max_llm_calls:
            return self._failure(state, ErrorCode.BUDGET_EXHAUSTED,
                                 "Compaction consumed the remaining call budget", retryable=False)
        if not isinstance(summary, str) or not summary.strip():
            return self._failure(state, ErrorCode.CONTRACT_ERROR,
                                 "Compaction handoff must be nonempty text",
                                 retryable=False, details={"component": "compaction"})
        checkpoint = HistoryCheckpoint(history_start=plan.history_start, summary=summary)
        # Compose against the proposed checkpoint before publishing it. A failed
        # summary or still-oversized request never advances the durable boundary.
        # The writing target is not a second hard budget: retain the complete
        # summary when it fits with current required context and recent turns.
        proposed = state.model_copy(update={"history_checkpoint": checkpoint})
        self._compose_context(definition, request, proposed, schemas, context_limit)
        state.history_checkpoint = checkpoint
        self._append_event(state, event_type="compaction", tool=None,
                           data=checkpoint.model_dump(mode="json"))
        self._save(state)
        return None

    def _current_turn(self, state: AgentState) -> ToolCallTurn:
        """Keep resumed receipts with their call even after notices were appended."""
        if self._active_call_id is not None:
            for turn in reversed(state.tool_turns):
                if any(call.id == self._active_call_id for call in turn.tool_calls):
                    return turn
            raise ValueError("active tool call does not belong to the Session")
        if (state.pending_control is not None
                and state.pending_control.source_turn < len(state.tool_turns)):
            return state.tool_turns[state.pending_control.source_turn]
        return state.tool_turns[-1]

    def _append_event(
        self,
        state: AgentState,
        *,
        event_type: str,
        tool: str | None,
        data,
    ) -> None:
        if event_type == "observation" and state.tool_turns:
            turn = self._current_turn(state)
            receipt = tool_receipt(ToolObservation.model_validate(data), len(state.events) + 1)
            if self._active_call_id is None:
                # Batch preflight/protocol rejection: nothing was dispatched.
                complete_pending_turn(turn, receipt)
            elif self._active_call_id not in turn.tool_results:
                record_tool_result(turn, self._active_call_id, receipt)
        state.events.append(
            AgentEvent(
                sequence=len(state.events) + 1,
                step=state.step,
                type=event_type,
                tool=tool,
                data=data,
                created_at=datetime.now(UTC),
            )
        )

    def _sync_usage(self, state: AgentState) -> None:
        """Project authoritative request reservations into Session diagnostics."""
        used = current_budget().usage.used - self._initial_usage
        state.llm_calls_used += used - self._run_llm_calls
        self._run_llm_calls = used

    def _preflight_batch(self, definition, registry, request, state, actions) -> AgentResult | None:
        """Validate the whole batch before its first side effect; recheck at dispatch."""
        for action in actions:
            if not registry.contains(action.tool):
                raise NativeToolCallError(f"Unknown tool: {action.tool}. Use the provided tools.")
            registry.validate(action.tool, action.arguments)
            try:
                permission = definition.permission_policy.check(action, state, request)
            except DeadlineExceededError:
                raise
            except (PermissionError, ValueError) as error:
                raise PermissionError(str(error)) from error
            except OperationCancelled:
                raise
            except Exception as error:
                return self._failure(state, ErrorCode.CONTRACT_ERROR,
                                     f"permission policy failed: {error}", retryable=False)
            check_cancelled()
            if permission.outcome == "deny":
                if permission.approval_id:
                    state.pending_action = None
                raise PermissionError(permission.reason or "Tool execution denied")
        return None

    def _cancel_pending_calls(self, state: AgentState, reason: str = "an earlier call stopped the batch") -> None:
        if state.tool_turns:
            turn = self._current_turn(state)
            if len(turn.tool_results) == len(turn.tool_calls):
                return
            if turn.executing_call_id is not None:
                record_tool_result(turn, turn.executing_call_id, tool_receipt(ToolObservation(
                    ok=False, summary=f"Execution interrupted: {reason}. Outcome is unknown.",
                    user_comment=turn.executing_user_comment,
                )))
            complete_pending_turn(turn, tool_receipt(ToolObservation(
                ok=False, summary=f"Not executed: {reason}.",
            )))
            self._save(state)

    def _save(self, state: AgentState) -> None:
        # Audit events own the review facts. Receipts only project their latest
        # status, including cancelled/recovered calls that never returned normally.
        reviews = {event.data["operation_key"]: event for event in state.events
                   if event.type == "permission_review"}
        if reviews:
            for ordinal, turn in enumerate(state.tool_turns):
                for call_id, receipt in turn.tool_results.items():
                    key = f"session:{state.session_id}:turn:{ordinal}:call:{call_id}"
                    event = reviews.get(key)
                    if event is not None:
                        value = json.loads(receipt)
                        value["permission_review"] = {
                            "source": "auto_review", "action_id": event.data["operation"]["action_id"],
                            "status": event.data["status"], "reason": event.data["reason"],
                            "audit_sequence": event.sequence,
                        }
                        turn.tool_results[call_id] = json.dumps(value, ensure_ascii=False)
        state.updated_at = datetime.now(UTC)
        self.store.save(state)
        # Raw command/control facts are durable before optional file export.
        if sync_tool_receipt_files(state, self._tool_result_memo):
            try:
                self.store.save(state)
            except OSError:
                # A missing reference can be rebuilt from the durable raw receipt.
                # Do not turn an optional export checkpoint into command failure.
                pass

    def _review_permission(self, definition, registry, request, state, action, permission):
        """Review once at dispatch; preflight and human resume remain deterministic."""
        if permission.outcome != "ask" or request.permissions.approval_mode != "auto_review":
            return permission
        operation = ActionSnapshot(
            action_id=f"action_{uuid4().hex}", tool=action.tool,
            arguments=registry.validate(action.tool, action.arguments).model_dump(mode="json"),
            context=permission.context, run_id=request.run_id, session_id=state.session_id,
            task_id=request.task_id, attempt_number=request.attempt_number,
        )

        def record(status, reason):
            self._append_event(state, event_type="permission_review", tool=action.tool, data={
                "source": "auto_review", "operation_key": state.current_operation_key,
                "operation": operation.model_dump(mode="json"), "status": status, "reason": reason,
            })
            self._save(state)

        record("started", "Automatic review requested; the operation has not executed")
        reviewer = definition.operation_reviewer
        try:
            if reviewer is None:
                raise ValueError("No operation reviewer is configured")
            tracer = getattr(reviewer, "set_trace_context", None)
            if tracer is not None:
                tracer(run_id=request.run_id, session_id=state.session_id, task_id=request.task_id,
                       attempt_number=request.attempt_number, agent="permission_reviewer", step=state.step,
                       operation_key=state.current_operation_key, action_id=operation.action_id)
            reviewed = reviewer.review(operation.model_copy(deep=True), request.model_copy(deep=True))
            verdict = ApprovalReviewVerdict.model_validate(
                reviewed.verdict if isinstance(reviewed, ApprovalReviewResult) else reviewed)
            check_cancelled()
        except OperationCancelled:
            record("interrupted", "Automatic review interrupted; no operation was approved or executed")
            raise
        except (BudgetExhaustedError, DeadlineExceededError) as error:
            record("error", f"Automatic review stopped: {type(error).__name__}; operation not executed")
            raise
        except UsagePersistenceError:
            # The Run ledger is authoritative. A failed save must stop execution,
            # rather than look like an unavailable model that a user can bypass.
            raise
        except Exception as error:
            # Do not copy arbitrary provider exception text (possibly credentials)
            # into the user's question. Persistence errors are outside this try.
            reason = f"Automatic review unavailable ({type(error).__name__}); manual approval required"
            # Failure is not permission to save a stale snapshot for approval.
            current = definition.permission_policy.check(action, state, request)
            check_cancelled()
            if current.outcome == "deny" or current.context != operation.context:
                reason = "Automatic review snapshot changed; operation not executed. Submit a new tool call if needed."
                record("invalidated", reason)
                return permission.model_copy(update={"outcome": "deny", "reason": reason})
            record("error", reason)
            return permission.model_copy(update={"reason": reason + ". " + permission.reason,
                                                 "reviewed_action": operation})
        finally:
            self._sync_usage(state)

        # The model's verdict cannot change a launch or a grant. Rebuild both
        # the original evidence and any host-collected inspection binding before
        # either automatic execution OR saving a question for human approval.
        try:
            current = definition.permission_policy.check(action, state, request)
            unchanged = current.outcome != "deny" and current.context == operation.context
            if unchanged and isinstance(reviewed, ApprovalReviewResult):
                # The extra context comes from the host inspection service, never
                # from the review_operation output schema. The policy still
                # rebuilds it from current bytes rather than trusting its hashes.
                current = definition.permission_policy.check(
                    action, state, request, expected_context=reviewed.context,
                )
                unchanged = current.outcome != "deny" and current.context == reviewed.context
                if unchanged:
                    operation = operation.model_copy(update={"context": current.context}, deep=True)
            if not unchanged:
                reason = ("Automatic review snapshot changed or permission was withdrawn; operation not executed. "
                          "Submit a new tool call if it is still needed.")
                record("invalidated", reason)
                return permission.model_copy(update={"outcome": "deny", "reason": reason})
        except Exception:
            record("invalidated", "The operation could not be revalidated; it was not executed")
            raise
        permission = current.model_copy(update={"reviewed_action": operation})
        record(verdict.decision, verdict.reason)
        if verdict.decision == "ask":
            return permission.model_copy(update={
                "outcome": "ask", "reason": "Automatic review requests your decision: " + verdict.reason,
            })
        return permission.model_copy(update={"outcome": "allow"})

    def _note_failure(
        self, state: AgentState, count: int
    ) -> AgentResult | None:
        """Return a failure result once the recoverable-failure limit is hit."""
        if count + 1 < _CONSECUTIVE_FAILURE_LIMIT:
            return None
        return self._failure(
            state,
            ErrorCode.TOOL_FAILED,
            "consecutive tool failures exceeded the recoverable limit",
            retryable=False,
            details=self._failure_details(state),
        )

    @staticmethod
    def _failure_details(state: AgentState) -> dict:
        return {"last_observation": state.last_observation.model_dump(mode="json")} if state.last_observation else {}

    def _record_observation(self, state, tool, observation):
        state.last_observation = observation
        self._append_event(state, event_type="observation", tool=tool,
                           data=observation.model_dump(mode="json"))
        self._save(state)

    def _feedback(self, state, summary, *, tool=None, value=None) -> None:
        """A rejection is a historical receipt, never a second context projection."""
        observation = ToolObservation(summary=summary, value=value, ok=False)
        if self._active_call_id is None and (
            not state.tool_turns or len(state.tool_turns[-1].tool_results) == len(state.tool_turns[-1].tool_calls)
        ):
            state.tool_turns.append(ToolCallTurn(protocol_error=summary))
        self._record_observation(state, tool, observation)

    def _control_result(self, state, *, llm_calls=None) -> AgentResult:
        control = state.pending_control
        if control is None:
            raise ValueError("paused control is missing")
        saved = state.pending_control_result
        if saved is not None and saved.control == control:
            return saved.model_copy(deep=True, update={"llm_calls": self._run_llm_calls if llm_calls is None else llm_calls})
        return AgentResult(
            status={"request_work": ModuleStatus.REQUEST_WORK, "ask_user": ModuleStatus.NEEDS_USER_INPUT,
                    "update_work": ModuleStatus.UPDATE_WORK, "abandon_work": ModuleStatus.ABANDON_WORK}[control.action],
            report=(state.last_observation.summary if state.last_observation else "Operation confirmation required"),
            control=control, session=self._session_ref(state),
            llm_calls=self._run_llm_calls if llm_calls is None else llm_calls,
        )

    def _checkpoint_control_result(self, state) -> None:
        # The public handoff must be complete before the first paused snapshot;
        # recovery is a read and cannot rerun the Agent's result finalizer.
        state.pending_control_result = None
        state.status = SessionStatus.PAUSED
        result = self._control_result(state, llm_calls=0)
        state.pending_control_result = (self._result_finalizer(result, state)
            if self._result_finalizer is not None else result)

    def _accept_control(self, state, request, observation) -> None:
        """Save the stable control identity before saving its accepted receipt."""
        now = datetime.now(UTC)
        payload = {
            "source_session_id": state.session_id,
            "source_turn": len(state.tool_turns) - 1,
            "source_call_id": self._active_call_id,
        }
        assessment = observation.memory_updates.get("latest_assessment")
        if assessment is not None:
            payload["assessment"] = ScientificAssessment.model_validate(assessment)
        if observation.work_update is not None:
            control = ControlSignal(action="update_work", work_update={
                **observation.work_update, "provided_input_ids": list(request.input_ids),
            }, **payload)
        elif observation.abandon_work_id is not None:
            control = ControlSignal(action="abandon_work", abandon_work_id=observation.abandon_work_id,
                                    reason=observation.control_reason, **payload)
        elif observation.question is not None:
            if observation.question.related_question_id is not None and not any(
                    q.id == observation.question.related_question_id and q.recipient_session_id == state.session_id for q in request.child_questions):
                raise PermissionError("escalation must reference your actual pending child question")
            control = ControlSignal(action="ask_user", question=PendingQuestion(
                id=f"question_{uuid4().hex}", run_id=request.run_id, session_id=state.session_id,
                task_id=request.task_id, attempt_number=request.attempt_number, created_at=now,
                recipient_session_id=request.upstream_session_id, **observation.question.model_dump(),
            ), **payload)
        else:
            work = WorkRequestDraft.model_validate(observation.request_work["work_request"])
            control = ControlSignal(action="request_work", work_request=WorkRequest(
                id=f"work_{uuid4().hex}", run_id=request.run_id,
                owner_session_id=state.session_id, owner_task_id=request.task_id,
                owner_attempt_number=request.attempt_number, depth=request.work_depth + 1,
                processed_input_ids=list(request.input_ids),
                request=work, created_at=now, updated_at=now,
            ), **payload)
        state.pending_control = control
        self._checkpoint_control_result(state)
        state.control_history.append(control)
        state.status = SessionStatus.PAUSED
        self._save(state)
        self._record_control_receipt(state)

    def _record_control_receipt(self, state) -> None:
        control = state.pending_control
        turn = state.tool_turns[control.source_turn]
        self._active_call_id = control.source_call_id
        if self._active_call_id in turn.tool_results:
            return
        if control.action in {"update_work", "abandon_work"}:
            value = {"status": "accepted", "action": control.action}
            summary = "Parent control accepted by the owner; no execution permission was granted"
        elif control.action == "request_work":
            value = {"status": "accepted", "work_request_id": control.work_request.id}
            summary = "Work request accepted; results will appear under this work ID in the complete directory"
        else:
            value = {"status": "question_issued", "question_id": control.question.id}
            summary = "Question issued; the answer will be included in the complete task dialogue"
        self._record_observation(state, control.action,
                                 ToolObservation(summary=summary, value=value))

    def _pause_approval(self, registry, request, state, action, permission) -> None:
        state.pending_action = permission.reviewed_action or ActionSnapshot(
            action_id=f"action_{uuid4().hex}", tool=action.tool,
            arguments=registry.validate(action.tool, action.arguments).model_dump(mode="json"),
            context=permission.context, run_id=request.run_id, session_id=state.session_id,
            task_id=request.task_id, attempt_number=request.attempt_number,
        )
        now = datetime.now(UTC)
        question = PendingQuestion(
            id=f"question_{uuid4().hex}", run_id=request.run_id, session_id=state.session_id,
            task_id=request.task_id, attempt_number=request.attempt_number, created_at=now,
            text=(permission.reason + "\nTool: " + action.tool
                  + "\nArguments: " + json.dumps(state.pending_action.arguments, ensure_ascii=False)
                  + "\nContext: " + json.dumps({key: value for key, value in permission.context.items()
                                                if key != "deletion"}, ensure_ascii=False)),
            requested_fields=["approve"], options={"approve": ["yes", "no"]}, action=state.pending_action,
        )
        state.pending_control = ControlSignal(
            action="ask_user", source_session_id=state.session_id,
            source_turn=len(state.tool_turns) - 1, source_call_id=self._active_call_id, question=question,
        )
        self._checkpoint_control_result(state)
        state.control_history.append(state.pending_control)
        turn = state.tool_turns[-1]
        # No receipt for the pending operation. Every unstarted tail call receives
        # a final, truthful receipt now and will never be automatically replayed.
        for call in turn.tool_calls:
            if call.id != self._active_call_id and call.id not in turn.tool_results:
                record_tool_result(turn, call.id, tool_receipt(ToolObservation(
                    summary="Not executed: an earlier operation is awaiting approval", ok=False,
                    value={"execution_status": "not_executed"},
                )))
        turn.executing_call_id = None
        state.status = SessionStatus.PAUSED
        self._save(state)

    def _resume_control(self, definition, registry, request, state) -> AgentResult | None:
        control = state.pending_control
        if control is None:
            return None
        if control.source_session_id != state.session_id or control.source_turn >= len(state.tool_turns):
            return self._failure(state, ErrorCode.CONTRACT_ERROR, "Stored control has an invalid source", retryable=False)
        ack = request.control_ack
        if ack is not None and (ack.session_id, ack.turn, ack.call_id) != (
            control.source_session_id, control.source_turn, control.source_call_id
        ):
            return self._failure(state, ErrorCode.CONTRACT_ERROR, "Resume acknowledges a different control", retryable=False)
        if control.action in {"update_work", "abandon_work"} or (request.release_control and control.action == "request_work"):
            self._record_control_receipt(state)
            # The owner has applied this parent control, or temporarily woken the
            # parent while its delegated work remains pending in the Run.
            state.pending_control = None
            state.pending_control_result = None
            state.status = SessionStatus.ACTIVE
            self._active_call_id = None
            self._save(state)
            return None
        if control.action == "request_work":
            self._record_control_receipt(state)
            available = any(result.work_request_id == control.work_request.id for result in request.work_results)
        else:
            answers = [answer for answer in request.resume_answers if answer.question_id == control.question.id]
            if any(
                (answer.question_text, answer.requested_fields, answer.options, answer.action,
                 answer.run_id, answer.session_id, answer.task_id, answer.attempt_number)
                != (control.question.text, control.question.requested_fields, control.question.options,
                    control.question.action, control.question.run_id, control.question.session_id,
                    control.question.task_id, control.question.attempt_number)
                for answer in answers
            ):
                return self._failure(state, ErrorCode.CONTRACT_ERROR,
                                     "Answer does not match the pending question", retryable=False)
            available = bool(answers)
            if control.question.action is None:
                self._record_control_receipt(state)
            elif available:
                return self._resume_approved_operation(definition, registry, request, state)
        if not available:
            state.status = SessionStatus.PAUSED
            self._save(state)
            return self._control_result(state)
        state.pending_control = None
        state.pending_control_result = None
        state.status = SessionStatus.ACTIVE
        self._active_call_id = None
        self._save(state)
        return None

    def _settle_outdated_operation(self, request, state) -> None:
        """Close an unexecuted old call before considering new parent input."""
        control, pending = state.pending_control, state.pending_action
        if control is None or control.question is None or pending != control.question.action:
            raise ValueError("Stored pending operation has no exact question")
        self._active_call_id = control.source_call_id
        comment = next((answer.comment for answer in request.resume_answers
                        if answer.question_id == control.question.id and answer.action == pending), None)
        self._record_observation(state, pending.tool, ToolObservation(
            ok=False, summary="Not executed: new upstream input must be processed before this old operation",
            value={"execution_status": "not_executed"}, user_comment=comment,
        ))
        state.pending_action = None
        state.pending_control = None
        state.pending_control_result = None
        state.status = SessionStatus.ACTIVE
        self._active_call_id = None
        self._save(state)

    def _resume_approved_operation(self, definition, registry, request, state) -> AgentResult | None:
        pending = state.pending_action
        control = state.pending_control
        if pending is None or pending != control.question.action:
            return self._failure(state, ErrorCode.CONTRACT_ERROR, "Stored approval has no matching operation", retryable=False)
        action = AgentAction(tool=pending.tool, arguments=pending.arguments)
        self._active_call_id = control.source_call_id
        state.current_operation_key = (
            f"session:{state.session_id}:turn:{control.source_turn}:call:{control.source_call_id}"
        )
        try:
            registry.validate(action.tool, action.arguments)
            permission = definition.permission_policy.check(action, state, request)
            if permission.outcome == "allow" and permission.approval_id != pending.action_id:
                raise PermissionError("Permission policy did not consume the matching approval")
        except OperationCancelled:
            raise
        except Exception as error:
            return self._failure(state, ErrorCode.CONTRACT_ERROR,
                                 "Cannot recheck approved operation: " + str(error), retryable=False)
        if boundary_requested():
            self._settle_outdated_operation(request, state)
            return self._yield_for_input(state)
        if permission.outcome == "ask":
            # One native call owns one received control identity. Settle the
            # reviewed operation instead of issuing another question for it.
            comment = next(answer.comment for answer in request.resume_answers
                           if answer.question_id == control.question.id)
            permission = permission.model_copy(update={
                "outcome": "deny",
                "reason": ("Approval snapshot changed; the original operation was not executed. "
                           "Submit a new tool call for fresh approval if it is still needed."),
                "user_comment": comment,
            })
        state.pending_action = None
        state.pending_control = None
        state.pending_control_result = None
        state.status = SessionStatus.ACTIVE
        if permission.outcome == "deny":
            observation = ToolObservation(ok=False, summary=permission.reason or "Operation denied",
                                          value={"execution_status": "not_executed"})
        else:
            state.tool_turns[control.source_turn].executing_call_id = self._active_call_id
            state.tool_turns[control.source_turn].executing_user_comment = permission.user_comment
            self._save(state)
            try:
                current_budget().remaining_timeout()
                observation = registry.dispatch(action.tool, action.arguments, state, prepared=permission.prepared)
            except DeadlineExceededError as error:
                # Dispatch may have started; retain the matched feedback without
                # claiming that an interrupted operation had no side effects.
                self._record_observation(state, action.tool, ToolObservation(
                    ok=False, summary="Approved operation timed out: " + str(error),
                    value={"execution_status": "unknown"}, user_comment=permission.user_comment,
                ))
                return self._failure(state, ErrorCode.TIMEOUT, str(error), retryable=False)
            except OperationCancelled:
                raise
            except Exception as error:
                observation = ToolObservation(ok=False, summary="Approved operation failed: " + str(error))
        observation = observation.model_copy(update={"user_comment": permission.user_comment})
        state.memory.update(observation.memory_updates)
        for ref in observation.artifacts:
            if ref.run_id != state.run_id or ref.session_id != state.session_id:
                return self._failure(state, ErrorCode.CONTRACT_ERROR,
                                     "Tool artifact belongs to another invocation", retryable=False)
            if ref.id not in state.authorized_artifact_ids:
                state.authorized_artifact_ids.append(ref.id)
        self._record_observation(state, action.tool, observation)
        self._acknowledge_registration(definition, state)
        self._active_call_id = None
        return None

    def _acknowledge_registration(self, definition, state) -> None:
        if definition.acknowledge_registration is not None and self._active_call_id is not None:
            definition.acknowledge_registration(
                run_id=state.run_id,
                operation_key=state.current_operation_key,
            )

    def _failure(
        self,
        state: AgentState,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool,
        details: dict | None = None,
        artifacts: list[ArtifactOutput] | None = None,
        report: str | None = None,
    ) -> AgentResult:
        error = ModuleError(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        )
        self._cancel_pending_calls(state, message)
        self._append_event(
            state,
            event_type="error",
            tool=None,
            data=error.model_dump(mode="json"),
        )
        state.status = SessionStatus.FAILED
        result = AgentResult(
            status=ModuleStatus.FAILED,
            report=report or message,
            artifacts=artifacts or [],
            error=error,
            session=self._session_ref(state),
            llm_calls=self._run_llm_calls,
        )
        return self._persist_terminal_result(state, result)

    def _persist_terminal_result(self, state: AgentState, result: AgentResult) -> AgentResult:
        """Save the complete public outcome once, after its owning Agent packages it."""
        if self._result_finalizer is not None:
            result = AgentResult.model_validate(self._result_finalizer(result, state))
        if (result.session is None or result.session.id != state.session_id
                or result.session.module != state.owner or result.session.status != state.status):
            raise ValueError("final result must match its owning terminal Session")
        state.terminal_result = result
        self._save(state)
        return result

    @staticmethod
    def _session_ref(state: AgentState) -> SessionRef:
        return SessionRef(
            id=state.session_id,
            module=state.owner,
            state_uri=f"session://{state.session_id}",
            status=state.status,
            created_at=state.created_at,
            updated_at=state.updated_at,
        )

    @staticmethod
    def _validation_details(error: ValidationError) -> dict:
        """Convert Pydantic errors to the stable JSON subset used by contracts."""

        return {
            "validation_errors": [
                {
                    "type": item["type"],
                    "loc": list(item["loc"]),
                    "message": item["msg"],
                }
                for item in error.errors(include_url=False)
            ]
        }
