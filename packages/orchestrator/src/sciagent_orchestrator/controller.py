"""Run lifecycle and Scientific invocations through the common Agent protocol."""

from __future__ import annotations

from datetime import UTC, datetime
from contextlib import contextmanager, nullcontext
from threading import Thread, Event
from contextvars import ContextVar
_OWNER = ContextVar("sciagent_run_owner", default=None)
from sciagent_runtime.cancellation import CancellationToken, OperationCancelled, bind_cancellation, current_cancellation, check_cancelled
from sciagent_runtime.lifecycle import model_boundary
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Protocol
from uuid import uuid4
from pydantic import BaseModel
from sciagent_components.literature import PreparedLiteratureImport, render_paper
from sciagent_runtime.budget import BudgetExhaustedError, DeadlineExceededError, execution_budget
from sciagent_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, AgentResult, BusinessInput, TaskStatus, AttemptStatus, ArtifactImport, ConclusionRequirements,
    DatasetRef, ErrorCode, ModuleError, ModuleStatus,
    RecordedAnswer, ResearchRequest, RunStatus, ControlOrigin,
    SessionRef, SessionStatus, TaskBudget, UserAnswer,
    WorkRequestStatus, WorkspaceDescriptor, WorkspaceGrant, WorkspaceAccess, WorkspaceSourceKind,
    task_session_id, WorkRequest, WorkRequestDraft, Workflow, WorkflowTask, WorkflowAgentKind, WorkflowAgentRegistry, Attempt,
)
from .compiler import CompilationError
from .completion import FinalReportRenderer, ScientificCompletionValidator
from .handoffs import read_json, receive_artifacts, receive_control, system_artifact
from .interpreter import WorkInterpreter
from .models import ResearchRun, RunInitialization, PendingModuleResult, PendingCancellation
from .ports import ModulePort, ModuleBinding
from .scheduler import _transition_work_request, _validate_answer
from .usage import RunUsagePort


class ScientificGate(Protocol):
    """Validate a Scientific completion against persisted Run evidence."""
    def validate(self, run, result, refs): ...


class DatasetRefSource(Protocol):
    """Discover currently registered dataset references."""
    def references(self) -> list[DatasetRef]: ...


def _scientific_instruction(request: ResearchRequest) -> str:
    parts = [request.goal]
    if request.hypothesis:
        parts.append("Hypothesis: " + request.hypothesis)
    if request.context:
        parts.append(request.context)
    parts.extend(request.constraints)
    return "\n\n".join(parts)


class ResearchController:
    """Coordinate Scientific reasoning, execution feedback and user pauses."""
    def __init__(self, *, scientific_port: ModulePort, compiler, scheduler, registry, interpreter: WorkInterpreter,
                 gate=None, report_renderer=None, dataset_ref_source=None, control_inbox=None):
        self.scientific_port = scientific_port
        self.compiler = compiler
        self.interpreter = interpreter
        self.scheduler = scheduler
        self.registry = registry
        self.gate = gate or ScientificCompletionValidator(scheduler.artifact_registry)
        self.report_renderer = report_renderer or FinalReportRenderer()
        self.dataset_ref_source = dataset_ref_source
        self.control_inbox = control_inbox
        self.scheduler.bindings[WorkflowAgentKind.SCIENTIFIC] = ModuleBinding(AgentOwner.SCIENTIFIC, scientific_port)
        self.scheduler.control_handler = self._receive_task_control
        self.scheduler.completion_handler = self._after_task_result
        self.scheduler.request_enricher = self._enrich_request
        self.scheduler.input_boundary = self._input_boundary

    def create_run(self, run_id, request, *, literature=()):
        with self._execution(run_id):
            return self._create_owned(run_id, request, literature=literature)

    def _create_owned(self, run_id, request, *, literature=()):
        if self.scheduler.store.exists(run_id):
            raise ValueError("run already exists")
        now = datetime.now(UTC)
        request = ResearchRequest.model_validate(request)
        run = ResearchRun(run_id=run_id, request=request, status=RunStatus.RUNNING,
                          workspaces=self.scheduler._resolve_workspaces(run_id),
                          execution_definitions=self.registry.execution_definitions,
                          execution_presets=self.registry.presets,
                          created_at=now, updated_at=now)
        root_work = WorkRequest(id="work_root", run_id=run_id,
            request=WorkRequestDraft(objective=request.goal,
                tasks=[{"key": "root", "instruction": _scientific_instruction(request),
                        "selection": {"kind": "preset", "preset_ref": "scientific@1",
                                      "reason": "The trusted root is fixed to Scientific"}}]),
            status=WorkRequestStatus.EXECUTING, workflow_revision=1, created_at=now, updated_at=now)
        run.work_requests = [root_work]
        run.workflow = Workflow(run_id=run_id, revision=1, created_from=root_work.id,
            tasks=[WorkflowTask(id="task_root", work_request_id=root_work.id,
                workflow_agent_kind=WorkflowAgentKind.SCIENTIFIC, instruction=_scientific_instruction(request))])
        run.initialization = RunInitialization(imports=request.input_artifacts,
                                               literature=[PreparedLiteratureImport.model_validate(item) for item in literature])
        self._save(run)
        with self._execution(run_id):
            self._complete_initialization(run)
            return self.run_until_stable(run_id)

    def _complete_initialization(self, run):
        """Recover original input positions before invoking any model."""
        progress = run.initialization
        for index in range(progress.next_import, len(progress.imports)):
            key = f"initialization:input:{index}"
            ref = self.scheduler.artifact_registry.register_import(progress.imports[index], run_id=run.run_id,
                                                                   operation_key=key, run=run)
            run.artifacts[ref.id] = ref
            progress.next_import = index + 1
            self._save(run)
            self.scheduler.artifact_registry.acknowledge_registration(key, run_id=run.run_id, run=run)
        self._import_literature(run)
        if run.conclusion_requirements_ref is None:
            run.conclusion_requirements_ref = system_artifact(
                self.scheduler.artifact_registry, run, "conclusion_requirements",
                ConclusionRequirements(required_evidence_kinds=run.request.required_evidence_kinds,
                                       required_artifacts=run.request.required_artifacts),
                operation_key="initialization:requirements",
            )
        progress.complete = True
        self._save(run)
        self.scheduler.artifact_registry.acknowledge_registration("initialization:requirements", run_id=run.run_id, run=run)

    def import_literature(self, run_id, literature):
        with self._execution(run_id):
            return self._import_owned(run_id, literature)

    def _import_owned(self, run_id, literature):
        run = self.scheduler.store.load(run_id)
        self._finish_pending_cancellation(run)
        if run.status != RunStatus.PAUSED:
            raise ValueError("literature may only be appended to a paused Run")
        run.initialization.literature.extend(PreparedLiteratureImport.model_validate(item) for item in literature)
        run.initialization.complete = False
        self._save(run)
        self._complete_initialization(run)
        return run

    def _import_literature(self, run):
        progress = run.initialization
        with tempfile.TemporaryDirectory(prefix="sciagent-literature-") as temporary:
            paper_path = Path(temporary) / "paper.md"
            for index in range(progress.next_literature, len(progress.literature)):
                item = progress.literature[index]
                metadata = {"paper_key": item.paper.key, "paper": item.paper.model_dump(mode="json")}
                pdf_digest = None
                if item.pdf_path is not None:
                    with item.pdf_path.open("rb") as handle:
                        if handle.read(5) != b"%PDF-":
                            raise ValueError("imported file has no PDF signature")
                        handle.seek(0)
                        pdf_digest = hashlib.file_digest(handle, "sha256").hexdigest()
                    metadata["import_pdf_sha256"] = pdf_digest
                paper_path.write_text(render_paper(item.paper), encoding="utf-8")
                paper_key = f"initialization:paper:{index}"
                paper_ref = self.scheduler.artifact_registry.register_import(
                    ArtifactImport(uri=str(paper_path), kind="literature_paper", media_type="text/markdown",
                                   summary=item.paper.title, metadata=metadata), run_id=run.run_id,
                    operation_key=paper_key, run=run,
                )
                run.artifacts[paper_ref.id] = paper_ref
                if item.pdf_path is not None:
                    pdf_ref = self.scheduler.artifact_registry.register_import(
                        ArtifactImport(uri=str(item.pdf_path), kind="literature_pdf", media_type="application/pdf",
                                       summary=item.paper.title + " — original PDF", expected_sha256=pdf_digest,
                                       metadata={"paper_artifact_id": paper_ref.id, "source_artifact_id": paper_ref.id,
                                                 "source_url": item.paper.source_url}),
                        run_id=run.run_id, operation_key=f"initialization:pdf:{index}", run=run,
                    )
                    run.artifacts[pdf_ref.id] = pdf_ref
                progress.next_literature = index + 1
                self._save(run)
                for prefix in ("paper", "pdf"):
                    self.scheduler.artifact_registry.acknowledge_registration(
                        f"initialization:{prefix}:{index}", run_id=run.run_id, run=run,
                    )

    def answer_question(self, run_id, answer: UserAnswer):
        with self._execution(run_id):
            return self._answer_owned(run_id, answer)

    def _answer_owned(self, run_id, answer):
        run = self.scheduler.store.load(run_id)
        self._finish_pending_cancellation(run)
        question = run.pending_question
        _validate_answer(question, answer)
        task_id = question.task_id
        session_id = question.session_id
        paired = RecordedAnswer(
            question_id=question.id, question_text=question.text, requested_fields=question.requested_fields,
            options=question.options, values=answer.values, answered_at=answer.answered_at,
            comment=answer.comment,
            run_id=run.run_id, session_id=session_id, task_id=task_id, attempt_number=question.attempt_number,
            action=question.action,
        )
        if task_id:
            task = self.scheduler._task(run, task_id)
            if not task.attempts or task.attempts[-1].number != question.attempt_number:
                raise ValueError("answer does not match the paused Attempt")
        run.append_answer(paired)
        self._settle_user_wait(run)
        run.pending_question = None
        run.status = RunStatus.RUNNING
        if task_id:
            self.scheduler.resume_task_in_place(run, task_id)
        self._save(run)
        return self.run_until_stable(run_id)

    def run_until_stable(self, run_id):
        with self._execution(run_id):
            try:
                return self._advance(run_id)
            except DeadlineExceededError as error:
                run = self.scheduler.store.load(run_id)
                return self._fail_run(run, ModuleError(code=ErrorCode.TIMEOUT,
                    message=f"Result reception timed out; saved execution results retained: {error}", retryable=False,
                    details={"stage": "result_reception", "saved_results_retained": True}))
            except OperationCancelled as error:
                if not error.stopped:
                    run = self.scheduler.store.load(run_id)
                    return self._fail_run(run, ModuleError(code=ErrorCode.INTERRUPTED,
                        message="Unable to confirm execution stopped: " + str(error), retryable=False,
                        details={**error.details, "stop_confirmed": False}))
                return self._pause_interrupted(run_id, str(error))

    def _advance(self, run_id):
        while True:
            run = self.scheduler.store.load(run_id)
            if run.pending_cancellation is not None:
                interrupted = run.pending_cancellation.source == "interrupt"
                self._finish_pending_cancellation(run)
                if interrupted:
                    return run
                continue
            if run.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
                # A deadline stops receiving as well as dispatch. Preserve its
                # saved facts without silently restarting the unfinished I/O.
                if run.terminal_error and run.terminal_error.code == ErrorCode.TIMEOUT:
                    return run
                self.scheduler._recover_received_results_in_place(run)
                if run.final_report_artifact_id is not None:
                    self._settle_final_report_registrations(run)
                return run
            if self._consume_controls(run):
                return self._pause_interrupted(run_id, "User interrupted current execution")
            if run.status == RunStatus.PAUSED:
                return run
            self._settle_final_report_registrations(run)
            if run.terminal_error and run.terminal_error.details.get("stop_confirmed") is False:
                return run
            if not run.initialization.complete:
                self._complete_initialization(run)
            self._merge_registered_datasets(run)
            # Already-observed stop confirmation/failure outranks a later
            # timeout. Settle its exact saved result without dispatching work.
            for task in run.workflow.tasks if run.workflow else []:
                for attempt in task.attempts:
                    pending = run.pending_module_results.get(f"task:{task.id}:attempt:{attempt.number}")
                    result = pending.received_result or pending.result if pending is not None else None
                    if result is not None and (result.status == ModuleStatus.INTERRUPTED
                            or result.error is not None and result.error.details.get("stop_confirmed") is False):
                        self.scheduler._apply_module_result(run, task, attempt, pending.request, pending.result)
                        if run.status in {RunStatus.PAUSED, RunStatus.FAILED}:
                            return run
            if run.remaining_timeout_seconds(datetime.now(UTC)) <= 0:
                return self._fail_run(run, ModuleError(code=ErrorCode.TIMEOUT,
                    message="Run execution-time budget exhausted", retryable=False))
            if self.scheduler._recover_interrupted_attempts_in_place(run):
                self._save(run)
                continue
            if run.remaining_timeout_seconds(datetime.now(UTC)) <= 0:
                return self._fail_run(run, ModuleError(code=ErrorCode.TIMEOUT, message="Run execution-time budget exhausted", retryable=False))
            if run.llm_calls_used >= run.request.budget.max_llm_calls:
                return self._fail_run(run, ModuleError(code=ErrorCode.BUDGET_EXHAUSTED, message="Run LLM-call budget exhausted", retryable=False))
            if self._deliver_child_results(run):
                self._save(run)
                continue
            attention = self._owner_needing_attention(run)
            if attention is not None:
                if attention.status in {TaskStatus.NEEDS_USER_INPUT, TaskStatus.WAITING_FOR_WORK}:
                    attention.status = TaskStatus.PENDING
                self._save(run)
                result = self.scheduler.execute_task(run_id, attention.id)
                if result.status == RunStatus.PAUSED and current_cancellation() and current_cancellation().cancelled:
                    return self._pause_interrupted(run_id, current_cancellation().reason)
                continue
            pending = next((work for work in run.work_requests
                if work.owner_task_id is not None and work.status in {WorkRequestStatus.REQUESTED, WorkRequestStatus.COMPILING}
                and not run.owner_chain_blocked(work.owner_task_id)
                and not run.needs_parent_processing(work)), None)
            if pending is not None:
                self._execute_work_request(run_id, pending.id)
                continue
            ready = self.scheduler._ready_task_ids(run)
            if ready:
                result = self.scheduler.execute_task(run_id, ready[0])
                if result.status == RunStatus.PAUSED and current_cancellation() and current_cancellation().cancelled:
                    return self._pause_interrupted(run_id, current_cancellation().reason)
                continue
            self.scheduler._evaluate_run(run)
            self._save(run)
            if self._deliver_child_results(run):
                self._save(run)
                continue
            return run

    def _owner_needing_attention(self, run):
        for work in run.work_requests:
            if work.owner_task_id is not None and not work.result_delivered and run.needs_parent_processing(work):
                if work.owner_task_id not in run.attention_task_ids:
                    run.attention_task_ids.append(work.owner_task_id)
        for task_id in list(run.attention_task_ids):
            task = run.task(task_id)
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}:
                run.attention_task_ids.remove(task_id)
                continue
            if run.owner_chain_blocked(task.id):
                continue
            if any(question.task_id == task.id for question in run.child_questions.values()):
                continue
            return task
        return None

    def _deliver_child_results(self, run):
        changed = False
        for work in run.work_requests:
            if work.owner_task_id is None or work.result_delivered or work.status not in {
                    WorkRequestStatus.STABLE, WorkRequestStatus.FAILED, WorkRequestStatus.CANCELLED}:
                continue
            if work.outcome is None:
                work.outcome = self.interpreter.interpret(work, tasks=run.workflow.tasks, artifacts=run.artifacts)
            owner = run.task(work.owner_task_id)
            if not owner.attempts or owner.attempts[-1].number != work.owner_attempt_number:
                raise ValueError("child result owner Attempt changed")
            owner.input_artifacts = list(dict.fromkeys([*owner.input_artifacts, *[ref.id for ref in work.outcome.artifacts]]))
            work.result_delivered = True
            if work.status == WorkRequestStatus.STABLE:
                _transition_work_request(work, WorkRequestStatus.CONSUMED)
            if owner.status == TaskStatus.WAITING_FOR_WORK:
                owner.status = TaskStatus.PENDING
            if owner.id == "task_root":
                run.scientific_artifact_ids = list(dict.fromkeys([*run.scientific_artifact_ids,
                    *[ref.id for ref in work.outcome.artifacts]]))
            changed = True
        return changed

    @contextmanager
    def _execution(self, run_id):
        """One owner writes Run/Session; the watcher reads inbox and signals only."""
        if _OWNER.get() == (id(self), run_id):
            yield
            return
        owner_marker = _OWNER.set((id(self), run_id))
        token = current_cancellation() or CancellationToken()
        stop = Event()
        thread = None
        lock = None
        if self.control_inbox is not None:
            # An OS lock fences separate CLI invocations as well as local threads.
            import fcntl
            lock_path = self.control_inbox.root / run_id / "owner.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = lock_path.open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                _OWNER.reset(owner_marker)
                raise ValueError("Run already has an active execution owner")
            def watch():
                while not stop.wait(0.05):
                    if not self.scheduler.store.exists(run_id):
                        continue
                    run = self.scheduler.store.load(run_id)
                    if any(e.kind == "interrupt" for e in self.control_inbox.pending(run_id, run.consumed_control_ids)):
                        token.cancel("User interrupted current execution")
                        return
            thread = Thread(target=watch, daemon=True)
            thread.start()
        def boundary():
            if self.control_inbox is None:
                return False
            run = self.scheduler.store.load(run_id)
            return any(e.kind == "input" for e in self.control_inbox.pending(run_id, run.consumed_control_ids))
        try:
            with bind_cancellation(token), model_boundary(boundary):
                yield
        finally:
            stop.set()
            if thread:
                thread.join()
            if lock:
                lock.close()
            _OWNER.reset(owner_marker)

    def _consume_controls(self, run, *, records=None):
        if self.control_inbox is None:
            return False
        interrupted = False
        if records is None:
            records = self.control_inbox.pending(run.run_id, run.consumed_control_ids)
        records = list(records)
        interrupt = next((event for event in records if event.kind == "interrupt"), None)
        if interrupt is not None and run.pending_cancellation is None:
            work = None if self._has_pending_terminal_result(run) else self._active_work_request(run)
            self._settle_subtree_results(run, work)
            self._set_cancellation(run, work, "User interrupted current execution",
                source="interrupt", operation_id="interrupt:" + interrupt.event_id)
        for event in records:
            if event.text.strip():
                value = BusinessInput(event_id=event.event_id, text=event.text,
                    source="user", created_at=event.created_at)
                run.append_input("task_root", value)
                run.upstream_inputs.append(value)
                if "task_root" not in run.attention_task_ids:
                    run.attention_task_ids.append("task_root")
                run.parent_attention = True
            interrupted |= event.kind == "interrupt"
            run.consumed_control_ids.append(event.event_id)
        self._save(run)
        return interrupted

    def resume_run(self, run_id):
        with self._execution(run_id):
            return self._resume_owned(run_id)

    def _resume_owned(self, run_id):
        run = self.scheduler.store.load(run_id)
        if run.pending_cancellation is not None:
            interrupted = run.pending_cancellation.source == "interrupt"
            self._finish_pending_cancellation(run)
            if interrupted:
                return run
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
            raise ValueError("terminal Run cannot resume")
        if self._consume_controls(run):
            return self._pause_interrupted(run_id, "User interrupted while the Run was waiting")
        if run.pending_question is not None:
            raise ValueError("answer the pending question before continuing")
        self._settle_user_wait(run)
        run.status = RunStatus.RUNNING
        self._save(run)
        return self.run_until_stable(run_id)

    @staticmethod
    def _settle_user_wait(run):
        """End one user wait without counting overlapping pause reasons twice."""
        started = run.user_wait_started_at
        if started is not None:
            run.user_wait_seconds += max(0, (datetime.now(UTC)-started).total_seconds())
        run.manually_paused_at = None

    def _settle_subtree_results(self, run, work):
        """Receive already-known terminal results before cancelling unfinished work."""
        if work is None:
            return
        work_ids = {work.id}
        for child in sorted(run.work_requests, key=lambda item: item.depth):
            if child.owner_task_id is not None and run.task(child.owner_task_id).work_request_id in work_ids:
                work_ids.add(child.id)
        terminal = {ModuleStatus.COMPLETED, ModuleStatus.COMPLETED_WITH_WARNINGS,
                    ModuleStatus.FAILED, ModuleStatus.BLOCKED}
        for task in run.workflow.tasks if run.workflow else []:
            if task.work_request_id not in work_ids:
                continue
            for attempt in task.attempts:
                pending = run.pending_module_results.get(f"task:{task.id}:attempt:{attempt.number}")
                if pending is not None and (pending.received_result or pending.result).status in terminal:
                    self.scheduler._apply_module_result(run, task, attempt, pending.request, pending.result)

    @staticmethod
    def _set_cancellation(run, work, reason, *, source, operation_id):
        active = {WorkRequestStatus.REQUESTED, WorkRequestStatus.COMPILING, WorkRequestStatus.EXECUTING}
        run.pending_cancellation = PendingCancellation(operation_id=operation_id,
            work_request_id=work.id if work is not None and work.status in active else None,
            source=source, reason=reason)

    def _cancel_work(self, run, work, reason, *, source, operation_id):
        """Persist one cancellation identity, then run its repeatable cleanup."""
        if run.pending_cancellation is None:
            self._settle_subtree_results(run, work)
            if work.status not in {WorkRequestStatus.REQUESTED, WorkRequestStatus.COMPILING, WorkRequestStatus.EXECUTING}:
                return
            self._set_cancellation(run, work, reason, source=source, operation_id=operation_id)
            self._save(run)
        elif run.pending_cancellation.operation_id != operation_id:
            raise ValueError("another cancellation must finish before starting this one")
        self._finish_pending_cancellation(run)

    def _finish_pending_cancellation(self, run):
        """Finish the original target before clearing its durable intent."""
        pending = run.pending_cancellation
        if pending is None:
            return run
        work = run.work(pending.work_request_id) if pending.work_request_id is not None else None
        if work is not None:
            self._cancel_work_contents(run, work, pending.reason)
            if pending.source == "owner_failure":
                work.result_delivered = True
            elif pending.source == "abandon":
                run.received_controls[pending.operation_id] = work.id
        if pending.source == "interrupt":
            stopped = not (run.terminal_error and run.terminal_error.details.get("stop_confirmed") is False)
            if stopped:
                if "task_root" not in run.attention_task_ids:
                    run.attention_task_ids.append("task_root")
                run.parent_attention = True
                run.status = RunStatus.PAUSED
                run.manually_paused_at = run.manually_paused_at or datetime.now(UTC)
            else:
                run.status = RunStatus.FAILED
            notice = {"event_id": "pause:" + pending.operation_id,
                "type": "user_interrupted", "reason": pending.reason,
                "stop_confirmed": stopped, "files_retained": True}
            if not any(item["event_id"] == notice["event_id"] for item in run.runtime_notices):
                run.runtime_notices.append(notice)
        run.pending_cancellation = None
        self._save(run)
        return run

    def _cancel_work_contents(self, run, work, reason):
        """Idempotent subtree cleanup; no business operation is dispatched here."""
        error = ModuleError(code=ErrorCode.INTERRUPTED, message=reason, retryable=False)
        for child in list(run.work_requests):
            if child.owner_task_id is not None and run.task(child.owner_task_id).work_request_id == work.id and (
                    child.status in {WorkRequestStatus.REQUESTED, WorkRequestStatus.COMPILING, WorkRequestStatus.EXECUTING}
                    or child.status == WorkRequestStatus.CANCELLED and not child.result_delivered):
                self._cancel_work_contents(run, child, reason)
                child.result_delivered = True
        for task in run.workflow.tasks if run.workflow else []:
            if task.work_request_id != work.id or task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED}:
                continue
            task.status = TaskStatus.CANCELLED
            if task.attempts:
                attempt = task.attempts[-1]
                if attempt.status in {AttemptStatus.RUNNING, AttemptStatus.NEEDS_USER_INPUT,
                                      AttemptStatus.WAITING_FOR_WORK, AttemptStatus.CANCELLED}:
                    attempt.status = AttemptStatus.CANCELLED
                    attempt.finished_at = attempt.finished_at or datetime.now(UTC)
                    attempt.error = error
                    attempt.report = attempt.report or reason
                    if attempt.session and attempt.session.status != SessionStatus.CANCELLED:
                        terminate = getattr(self.scheduler.bindings[task.workflow_agent_kind].port, "terminate_session", None)
                        if terminate:
                            terminate(attempt.session.id, reason)
                        attempt.session = attempt.session.model_copy(update={"status": SessionStatus.CANCELLED})
                    self._save(run)
                    self.scheduler._persist_attempt_report(run, task, attempt, acknowledge=False)
                    self.scheduler._acknowledge_attempt_result(run, task, attempt)
        child_ids = {task.id for task in run.workflow.tasks if task.work_request_id == work.id} if run.workflow else set()
        cancelled_question_ids = {key for key, question in run.child_questions.items() if question.task_id in child_ids}
        if run.pending_question and run.pending_question.related_question_id in cancelled_question_ids:
            question = run.pending_question
            cancel = getattr(self.scientific_port, "cancel_question", None)
            if cancel:
                cancel(question.session_id, question.id, "Escalated question ended with its original child work")
            self._settle_user_wait(run)
            run.pending_question = None
        run.child_questions = {key: question for key, question in run.child_questions.items() if question.task_id not in child_ids}
        if run.pending_question is not None and run.pending_question.task_id in child_ids:
            self._settle_user_wait(run)
            run.pending_question = None
        if run.pending_compilation is not None and run.pending_compilation.work_request_id == work.id:
            run.pending_compilation = None
        if work.status != WorkRequestStatus.CANCELLED:
            _transition_work_request(work, WorkRequestStatus.CANCELLED, error=error)
        work.outcome = self.interpreter.interpret(work, tasks=run.workflow.tasks if run.workflow else [], artifacts=run.artifacts)
        if work.owner_task_id == "task_root":
            for ref in work.outcome.artifacts:
                if ref.id not in run.scientific_artifact_ids:
                    run.scientific_artifact_ids.append(ref.id)
        notice = {"event_id": "cancel:" + work.id, "type": "work_interrupted",
            "recipient_session_id": work.owner_session_id,
            "work_request_id": work.id, "reason": reason,
            "stop_confirmed": not (run.terminal_error and run.terminal_error.details.get("stop_confirmed") is False),
            "files_retained": True, "tasks": [t.model_dump(mode="json") for t in work.outcome.tasks],
            "artifact_ids": [ref.id for ref in work.outcome.artifacts],
            "file_retention_note": "Files were retained without rollback; partial files are not validated deliveries.",
            "reusable_workspace_ids": list(dict.fromkeys(t.workspace_id for t in run.workflow.tasks
                if t.work_request_id == work.id and t.workspace_id)) if run.workflow else []}
        if not any(item["event_id"] == notice["event_id"] for item in run.runtime_notices):
            run.runtime_notices.append(notice)
        if work.owner_task_id is not None and work.owner_task_id not in run.attention_task_ids:
            run.attention_task_ids.append(work.owner_task_id)
        run.parent_attention = True

    @staticmethod
    def _has_pending_terminal_result(run):
        """Known execution is retained for receiving; stopping it is not cancelling business work."""
        terminal = {ModuleStatus.COMPLETED, ModuleStatus.COMPLETED_WITH_WARNINGS,
                    ModuleStatus.FAILED, ModuleStatus.BLOCKED}
        results = [pending.received_result or pending.result
                   for pending in run.pending_module_results.values()]
        # A known failure to stop must retain the existing FAILED/cleanup path.
        if any(result.error and result.error.details.get("stop_confirmed") is False for result in results):
            return False
        return any(result.status in terminal for result in results)

    def _pause_interrupted(self, run_id, reason):
        # Dispatch has unwound and all managed operations have confirmed stop.
        # Short durable pause/report settlement must not cancel itself again.
        with bind_cancellation(None):
            return self._settle_interrupt(run_id, reason)

    def _settle_interrupt(self, run_id, reason):
        run = self.scheduler.store.load(run_id)
        self._consume_controls(run)
        if run.pending_cancellation is None:
            work = None if self._has_pending_terminal_result(run) else self._active_work_request(run)
            self._settle_subtree_results(run, work)
            self._set_cancellation(run, work, reason, source="interrupt",
                operation_id="interrupt:" + uuid4().hex)
            self._save(run)
        return self._finish_pending_cancellation(run)

    def _apply_work_update(self, run, request, signal):
        payload = signal.work_update
        key = f"session:{signal.source_session_id}:turn:{signal.source_turn}:call:{signal.source_call_id}"
        if key in run.received_controls:
            if run.received_controls[key] != payload["work_request_id"]:
                raise ValueError("control identity was already received for a different target")
            return
        work = run.active_child(request.task_id, request.attempt_number)
        if work is None or work.id != payload["work_request_id"] or work.owner_session_id != signal.source_session_id:
            raise ValueError("update target is not this Attempt's active delegation")
        if work.status not in {WorkRequestStatus.REQUESTED, WorkRequestStatus.COMPILING, WorkRequestStatus.EXECUTING}:
            raise ValueError("terminal work cannot receive execution inputs")
        provided = payload.get("provided_input_ids", [])
        if provided != request.input_ids or provided != run.input_ids(request.task_id, request.attempt_number):
            raise ValueError("parent update was prepared before new direct input; refresh first")
        allowed = {task.id for task in run.workflow.tasks if task.work_request_id == work.id
            and task.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.NEEDS_USER_INPUT, TaskStatus.WAITING_FOR_WORK}}
        # Validate the whole batch before changing any answer or input record.
        prepared = []
        for index, item in enumerate(payload.get("inputs", [])):
            targets = item["task_ids"]
            if not targets or len(targets) != len(set(targets)) or not set(targets) <= allowed:
                raise ValueError("input targets must be live direct child Tasks")
            original = None
            source_id = item.get("source_input_id")
            if source_id is not None:
                original = next((event for event in request.upstream_inputs if event.event_id == source_id), None)
                if original is None:
                    raise ValueError("forwarded input is not available to this parent")
            if (original is None) == (item.get("text") is None):
                raise ValueError("provide new text or a source input identity")
            record = (original.model_copy(update={"original_event_id": original.original_event_id or original.event_id})
                if original is not None else BusinessInput(event_id=f"{key}:input:{index}",
                    text=item["text"], source="parent", created_at=datetime.now(UTC),
                    source_session_id=signal.source_session_id))
            prepared.append((targets, record))
        replies = []
        seen = set()
        for item in payload.get("answers", []):
            question = run.child_questions.get(item["question_id"])
            if item["task_id"] not in allowed or question is None or question.task_id != item["task_id"] or question.recipient_session_id != signal.source_session_id:
                raise ValueError("answer does not bind a live direct child business question")
            if question.action is not None or question.id in seen:
                raise ValueError("update_work answers only distinct business questions")
            seen.add(question.id)
            original = None
            if item.get("source_answer_id") is not None:
                original = next((answer for answer in request.user_answers
                    if answer.question_id == item["source_answer_id"]), None)
                if original is None:
                    raise ValueError("source answer is not available to this parent")
            values = original.values if original else item.get("values")
            if values is None:
                raise ValueError("answer values are required")
            answer = UserAnswer(question_id=question.id, values=values,
                answered_at=original.answered_at if original else datetime.now(UTC), comment=original.comment if original else None)
            _validate_answer(question, answer)
            target = run.task(question.task_id)
            if target.attempts[-1].number != question.attempt_number or target.attempts[-1].session.id != question.session_id:
                raise ValueError("answer Attempt/Session no longer matches the question")
            replies.append(RecordedAnswer(**answer.model_dump(), question_text=question.text,
                requested_fields=question.requested_fields, options=question.options,
                run_id=run.run_id, session_id=question.session_id, task_id=question.task_id,
                attempt_number=question.attempt_number, source=original.source if original else "parent",
                source_session_id=signal.source_session_id,
                original_question_id=(original.original_question_id or original.question_id) if original else None))
        for targets, record in prepared:
            for task_id in targets:
                run.append_input(task_id, record)
        for answer in replies:
            run.append_answer(answer)
            del run.child_questions[answer.question_id]
        work.processed_input_ids = list(provided)
        run.received_controls[key] = work.id
        run.attention_task_ids = [identity for identity in run.attention_task_ids if identity != request.task_id]
        for task in run.workflow.tasks:
            if task.work_request_id == work.id and task.status == TaskStatus.NEEDS_USER_INPUT and not any(
                    question.task_id == task.id for question in run.child_questions.values()) and not (
                    run.pending_question and run.pending_question.task_id == task.id):
                task.status = TaskStatus.PENDING
        run.parent_attention = bool(run.attention_task_ids)

    def _receive_task_control(self, run, task, attempt, request, result):
        from .handoffs import control_key
        key = control_key(result.control)
        target = (result.control.question.id if result.control.action == "ask_user" else
            result.control.work_request.id if result.control.action == "request_work" else
            result.control.work_update["work_request_id"] if result.control.action == "update_work" else
            result.control.abandon_work_id)
        if key in run.received_controls:
            if run.received_controls[key] != target:
                raise ValueError("control identity was already received for a different target")
            return
        if result.status in {ModuleStatus.UPDATE_WORK, ModuleStatus.REQUEST_WORK, ModuleStatus.ABANDON_WORK} and request.input_ids != run.input_ids(task.id, attempt.number):
            # This exact batch was saved before newer parent input arrived.
            # Consume its identity without applying any part of the old batch.
            run.received_controls[key] = target
            if task.id not in run.attention_task_ids:
                run.attention_task_ids.append(task.id)
            run.runtime_notices.append({"event_id": "stale_control:" + key,
                "type": "work_control_not_applied", "recipient_session_id": result.session.id,
                "action": result.control.action, "applied": False,
                "reason": "New direct input arrived after this control was prepared; it was not applied. Review the input before issuing new work control."})
            run.parent_attention = True
            return
        if task.id in run.attention_task_ids:
            run.attention_task_ids.remove(task.id)
        if result.status == ModuleStatus.UPDATE_WORK:
            self._apply_work_update(run, request, result.control)
        elif result.status == ModuleStatus.ABANDON_WORK:
            from .handoffs import control_key
            key = control_key(result.control)
            if key in run.received_controls:
                return
            work = run.active_child(task.id, attempt.number)
            if work is None or work.id != result.control.abandon_work_id:
                raise ValueError("cannot end a foreign delegation")
            self._cancel_work(run, work, result.control.reason or "Parent ended delegation",
                source="abandon", operation_id=key)
            from .handoffs import control_key
            run.received_controls[control_key(result.control)] = work.id
        else:
            if result.status == ModuleStatus.REQUEST_WORK:
                containing = run.work(task.work_request_id)
                binding = self.scheduler.bindings[task.workflow_agent_kind]
                if not binding.can_delegate or containing.depth >= run.request.execution_limits.max_work_depth:
                    raise ValueError("delegation capability or depth is not granted by this Run")
                authorized = {ref.id for ref in request.input_artifacts} | set(attempt.artifact_ids)
                if not {identity for item in result.control.work_request.request.tasks
                        for identity in item.input_artifact_ids} <= authorized:
                    raise ValueError("child work references unauthorized artifacts")
            receive_control(run, request, result)
        run.parent_attention = bool(run.attention_task_ids)

    def _input_boundary(self, request):
        run = self.scheduler.store.load(request.run_id)
        if self.control_inbox and self.control_inbox.pending(run.run_id, run.consumed_control_ids):
            return True
        return (run.pending_cancellation is not None
                or run.input_ids(request.task_id, request.attempt_number) != request.input_ids
                or run.owner_chain_blocked(request.task_id))

    def _enrich_request(self, run, task, request):
        if task.id != "task_root":
            return request
        context = dict(request.task_context)
        context.update({"conclusion_requirements": read_json(run.conclusion_requirements_ref),
            "run_permissions": run.request.permissions.model_dump(mode="json", exclude={"schema_version"})})
        refs = {ref.id: ref for ref in request.input_artifacts}
        refs.update((ref.id, ref) for ref in self._authorized_artifacts(run))
        return request.model_copy(update={"input_artifacts": list(refs.values()),
            "read_workspaces": [*self._scientific_read_workspaces(run), *request.read_workspaces], "task_context": context})

    def _merge_registered_datasets(self, run):
        if self.dataset_ref_source is None:
            return
        by_id = {ref.dataset_id: ref for ref in run.dataset_refs}
        for ref in self.dataset_ref_source.references():
            if ref.dataset_id in by_id and by_id[ref.dataset_id] != ref:
                raise ValueError("dataset remapped during Run")
            by_id[ref.dataset_id] = ref
        if list(by_id.values()) != run.dataset_refs or run.dataset_catalog_ref is None:
            run.dataset_refs = list(by_id.values())
            run.dataset_catalog_ref = system_artifact(self.scheduler.artifact_registry, run, "dataset_catalog",
                                                     {"datasets": [ref.model_dump(mode="json") for ref in run.dataset_refs]})
            self._save(run)
            for key in list(run.pending_registrations):
                if key.startswith("system:dataset_catalog:"):
                    self.scheduler.artifact_registry.acknowledge_registration(key, run_id=run.run_id, run=run)

    def _settle_final_report_registrations(self, run):
        root = next((task for task in run.workflow.tasks if task.id == "task_root"), None) if run.workflow else None
        attempt = root.attempts[-1] if root and root.attempts else None
        pending = run.pending_module_results.get(f"task:task_root:attempt:{attempt.number}") if attempt else None
        active_key = None
        if (run.final_report_artifact_id is None and pending is not None
                and pending.result.status in {ModuleStatus.COMPLETED, ModuleStatus.COMPLETED_WITH_WARNINGS}):
            # Older snapshots used one fixed slot and carry no candidate key.
            active_key = pending.final_report_key or "final_report"
        self.scheduler.artifact_registry.settle_final_report_registrations(
            run_id=run.run_id, run=run, active_key=active_key)

    def _after_task_result(self, run, task, attempt, request, result):
        if result.status in {ModuleStatus.FAILED, ModuleStatus.BLOCKED}:
            child = run.active_child(task.id, attempt.number)
            if child is not None and child.status in {WorkRequestStatus.REQUESTED,
                    WorkRequestStatus.COMPILING, WorkRequestStatus.EXECUTING}:
                self._cancel_work(run, child, "Owning Task ended before its delegation completed",
                    source="owner_failure", operation_id=f"owner:{task.id}:attempt:{attempt.number}")
        if task.id != "task_root":
            return run
        run.scientific_session = result.session or run.scientific_session
        run.scientific_report = result.report
        run.scientific_artifact_ids = list(dict.fromkeys([*run.scientific_artifact_ids, *attempt.artifact_ids]))
        if task.id in run.attention_task_ids and result.status not in {
                ModuleStatus.YIELDED, ModuleStatus.REQUEST_WORK, ModuleStatus.UPDATE_WORK,
                ModuleStatus.ABANDON_WORK, ModuleStatus.NEEDS_USER_INPUT}:
            run.attention_task_ids.remove(task.id)
        # A terminal Run snapshot already committed the root receiving verdict.
        # Recovery may finish acknowledgements, but must not revalidate/publish it.
        if ((run.status == RunStatus.COMPLETED and run.final_report_artifact_id is not None)
                or (run.status == RunStatus.FAILED and attempt.report_artifact_id is not None)):
            return run
        if result.status in {ModuleStatus.COMPLETED, ModuleStatus.COMPLETED_WITH_WARNINGS}:
            pending = run.pending_module_results.get(f"task:{task.id}:attempt:{attempt.number}")
            previous = set(pending.previous_artifact_ids) if pending else set()
            explicit = {item.id for item in result.artifacts if hasattr(item, "id")}
            refs = [run.artifacts[identity] for identity in attempt.artifact_ids
                if identity not in previous or identity in explicit]
            # Validation and rendering may read substantial immutable evidence.
            # Only the final inbox read and commit need the cross-process fence.
            validation = self.gate.validate(run, result, refs)
            rendered = self.report_renderer.render(validation.report) if validation.ok else None
            boundary = self.control_inbox.settlement(run.run_id, run.consumed_control_ids) if self.control_inbox else nullcontext([])
            with boundary as records:
                interrupted = self._consume_controls(run, records=records) if records else False
                if interrupted:
                    # Unwind before the caller acknowledges this uncommitted
                    # final result. Resume still applies the original input gate.
                    raise OperationCancelled("User interrupted before root publication")
                if records or request.input_ids != run.input_ids(task.id, attempt.number):
                    defer = getattr(self.scientific_port, "defer_completion", None)
                    if defer is None or result.session is None:
                        raise ValueError("root port cannot defer completion for pending input")
                    defer(result.session.id, "New user input arrived before root completion committed")
                    task.status = TaskStatus.PENDING
                    attempt.status = AttemptStatus.NEEDS_USER_INPUT
                    attempt.finished_at = None
                    attempt.error = None
                    if attempt.report_artifact_id is not None:
                        self.scheduler.artifact_registry.acknowledge_registration(
                            f"task:{task.id}:attempt:{attempt.number}:report", run_id=run.run_id, run=run)
                    attempt.report_artifact_id = None
                    run.pending_module_results.pop(f"task:{task.id}:attempt:{attempt.number}", None)
                    attempt.session = result.session.model_copy(update={"status": SessionStatus.PAUSED})
                    run.scientific_session = attempt.session
                    if task.id not in run.attention_task_ids:
                        run.attention_task_ids.append(task.id)
                    run.parent_attention = True
                    self._save(run)
                    self._settle_final_report_registrations(run)
                    return self._finish_pending_cancellation(run) if interrupted else run
                self.scheduler._persist_attempt_report(run, task, attempt, acknowledge=False)
                if not validation.ok:
                    run.completion_violations = list(validation.violations)
                    return self._fail_run(run, ModuleError(code=ErrorCode.CONTRACT_ERROR,
                        message="Scientific completion was rejected", retryable=False))
                if pending.final_report_key is None:
                    pending.final_report_key = ("final_report" if "final_report" in run.pending_registrations
                                                else f"final_report:{uuid4().hex}")
                    self._save(run)
                try:
                    report = self.scheduler.artifact_registry.register_final_report(rendered.candidate,
                        rendered.content, run_id=run.run_id, run=run, operation_key=pending.final_report_key)
                except DeadlineExceededError:
                    raise
                except OSError as error:
                    # Allocation/freezing may already have persisted material.
                    # Preserve those authoritative facts before marking failure.
                    current = self.scheduler.store.load(run.run_id)
                    for name in type(run).model_fields:
                        setattr(run, name, getattr(current, name))
                    return self._fail_run(run, ModuleError(code=ErrorCode.ARTIFACT_MISSING,
                        message=str(error).strip() or type(error).__name__, retryable=False))
                run.artifacts[report.id] = report
                run.final_opinion = validation.report.opinion
                run.final_report_artifact_id = report.id
                run.status = RunStatus.COMPLETED
                work = run.work(task.work_request_id)
                work.status = WorkRequestStatus.STABLE
                work.outcome = self.interpreter.interpret(work, tasks=run.workflow.tasks, artifacts=run.artifacts)
                work.status = WorkRequestStatus.CONSUMED
                work.result_delivered = True
                self._save(run)
            self._settle_final_report_registrations(run)
            return run
        if result.status in {ModuleStatus.FAILED, ModuleStatus.BLOCKED}:
            return self._fail_run(run, result.error)
        self._save(run)
        return run

    def _fail_run(self, run, error):
        run.status = RunStatus.FAILED
        run.terminal_error = error
        if not run.scientific_report:
            run.scientific_report = error.message
        self._save(run)
        return run

    def _execute_work_request(self, run_id, work_id):
        run = self.scheduler.store.load(run_id)
        active = run.work(work_id)
        try:
            if active.status == WorkRequestStatus.REQUESTED:
                _transition_work_request(active, WorkRequestStatus.COMPILING)
                self._save(run)
            existing = [task for task in run.workflow.tasks if task.work_request_id == active.id]
            if not existing:
                remaining = run.request.execution_limits.max_tasks - len(run.workflow.tasks)
                if len(active.request.tasks) > remaining:
                    failure = ModuleError(code=ErrorCode.BUDGET_EXHAUSTED,
                        message="No remaining task slots" if remaining <= 0 else "Explicit work exceeds remaining task slots", retryable=False)
                    _transition_work_request(active, WorkRequestStatus.FAILED, error=failure)
                    run.pending_compilation = None
                    self._save(run)
                    return run
                if run.pending_compilation is None:
                    with execution_budget(max_llm_calls=run.request.budget.max_llm_calls-run.llm_calls_used,
                            timeout_seconds=run.remaining_timeout_seconds(datetime.now(UTC)), usage=RunUsagePort(run, self.scheduler.store)):
                        compilation = self.compiler.compile(active, current=run.workflow,
                            registry=WorkflowAgentRegistry(execution_definitions=run.execution_definitions, presets=run.execution_presets),
                            limits=run.request.execution_limits, workspaces=self._workspace_descriptors(run))
                    check_cancelled()
                    run.pending_compilation = compilation.output
                    self._save(run)
                self.scheduler.apply_patch(run_id, run.pending_compilation)
                run = self.scheduler.store.load(run_id)
                active = run.work(work_id)
            _transition_work_request(active, WorkRequestStatus.EXECUTING, workflow_revision=run.workflow.revision)
            run.pending_compilation = None
            self._save(run)
            return run
        except OperationCancelled:
            raise
        except Exception as error:
            run = self.scheduler.store.load(run_id)
            active = run.work(work_id)
            code = (ErrorCode.TIMEOUT if isinstance(error, DeadlineExceededError) else
                ErrorCode.BUDGET_EXHAUSTED if isinstance(error, BudgetExhaustedError) else ErrorCode.CONTRACT_ERROR)
            failure = ModuleError(code=code, message=str(error).strip() or type(error).__name__, retryable=False,
                details={"compiler_usage_known": isinstance(error, CompilationError),
                         **(error.details if isinstance(error, CompilationError) else {})})
            _transition_work_request(active, WorkRequestStatus.FAILED, error=failure)
            run.pending_compilation = None
            self._save(run)
            return run

    @staticmethod
    def _active_work_request(run):
        return run.active_child("task_root")

    def _scientific_read_workspaces(self, run):
        # File paths in child questions are locators, never grants.
        grants = [WorkspaceGrant(root=record.root, source=record.source.source_kind,
            access=WorkspaceAccess(read_paths=record.source.access.read_paths,
                                   denied_paths=record.source.access.denied_paths))
            for record in run.workspaces.values()]
        return grants

    def _authorized_artifacts(self, run):
        imports = [ref.id for ref in run.artifacts.values() if ref.metadata.get("source_type") == "import"]
        system = [ref.id for ref in (run.conclusion_requirements_ref, run.dataset_catalog_ref) if ref is not None]
        ids = list(dict.fromkeys([*imports, *system, *run.scientific_artifact_ids]))
        refs = {key: run.artifacts[key] for key in ids}
        return list(refs.values())

    def _unresolved_tasks(self, run):
        if not run.workflow:
            return []
        result = []
        for work in run.work_requests:
            if work.outcome:
                result.extend(item for item in work.outcome.tasks if item.status != "completed")
        return result

    def _workspace_descriptors(self, run):
        return [WorkspaceDescriptor(workspace_id=key, source_kind=record.source.source_kind)
                for key, record in run.workspaces.items()]

    def _save(self, run):
        run.updated_at = datetime.now(UTC)
        self.scheduler.store.save(ResearchRun.model_validate(run.model_dump()))
