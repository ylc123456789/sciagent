"""WorkRequest configuration binding, without task rewriting or model calls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from sciagent_contracts import (
    FutureArtifactBinding, ExecutionDefinition, ExecutionLimits, TaskProposal, Workflow,
    WorkflowAgentKind, WorkflowAgentRegistry, WorkflowPatch, WorkflowProposal,
    WorkRequest, WorkspaceDescriptor,
)
from .workflow_validation import validate_workflow_candidate


@dataclass(frozen=True)
class CompilationResult:
    output: WorkflowProposal | WorkflowPatch
    llm_calls: int = 0


class CompilationError(ValueError):
    """A rejected binding, preserving its exact task and deterministic category."""
    def __init__(self, message: str, *, category="invalid_binding", task_key=None, details=None, llm_calls=0):
        super().__init__(message)
        self.llm_calls = llm_calls
        self.details = {"category": category, **(details or {})}
        if task_key is not None:
            self.details["task_key"] = task_key


class UnrouteableTaskError(CompilationError):
    """The parent explicitly reported that available presets do not fit."""


class WorkflowCompiler(Protocol):
    def compile(self, request: WorkRequest, *, current: Workflow | None,
                registry: WorkflowAgentRegistry, limits: ExecutionLimits,
                workspaces: list[WorkspaceDescriptor] | None = None) -> CompilationResult: ...


class DeterministicWorkflowCompiler:
    """Inject a trusted graph in orchestration fixtures without an LLM."""

    def __init__(self, proposal: WorkflowProposal, patch: WorkflowPatch | None = None) -> None:
        self._proposal = proposal
        self._patch = patch

    def compile(self, request, *, current, registry, limits, workspaces=None):
        if current is None:
            proposal = self._proposal.model_copy(deep=True)
            proposal.work_request_id = request.id
            for task in proposal.tasks:
                task.work_request_id = request.id
            return CompilationResult(proposal)
        # The uniform root graph exists before its first child request.
        # Reuse the trusted fixture proposal as that first patch; subsequent
        # requests still require an explicitly configured patch.
        prior_children = [task for task in current.tasks
            if task.workflow_agent_kind != WorkflowAgentKind.SCIENTIFIC]
        if not prior_children:
            patch = WorkflowPatch(work_request_id=request.id, based_on_revision=current.revision,
                add_tasks=[task.model_copy(deep=True, update={"work_request_id": request.id})
                    for task in self._proposal.tasks])
        elif self._patch is None:
            raise CompilationError("no patch configured for an existing workflow")
        else:
            patch = self._patch.model_copy(deep=True)
        patch.work_request_id = request.id
        for task in patch.add_tasks:
            task.work_request_id = request.id
        return CompilationResult(patch)


def _validate_request_resources(request, current, limits, workspaces):
    used = len(current.tasks) if current else 0
    if len(request.request.tasks) > limits.max_tasks - used:
        raise CompilationError("explicit tasks exceed remaining task limits")
    workspace_ids = {w.workspace_id for w in workspaces}
    for task in request.request.tasks:
        if task.workspace_id is None and len(workspace_ids) > 1:
            raise CompilationError("workspace_id required with multiple workspaces")
        if task.workspace_id is not None and task.workspace_id not in workspace_ids:
            raise CompilationError("task selected an undeclared workspace")


class FixedWorkflowCompiler:
    """Resolve a parent's explicit choices and bind the original tasks atomically."""
    def __init__(self, validate_definition: Callable[[ExecutionDefinition], None] | None = None):
        self.validate_definition = validate_definition

    def compile(self, request, *, current, registry, limits, workspaces=None):
        try:
            request = WorkRequest.model_validate(request)
            registry = WorkflowAgentRegistry.model_validate(registry)
        except ValueError as error:
            raise CompilationError(str(error), category="invalid_input") from error
        workspaces = list(workspaces or [])
        _validate_request_resources(request, current, limits, workspaces)
        presets = {item.preset_ref: item for item in registry.presets}
        by_key = {}
        unmatched = [{"task_key": item.key, "reason": item.selection.reason}
                     for item in request.request.tasks if item.selection.kind == "no_match"]
        if unmatched:
            raise UnrouteableTaskError(
                "; ".join(f"task {item['task_key']} cannot be routed: {item['reason']}" for item in unmatched),
                category="no_match", details={"unmatched": unmatched})
        for item in request.request.tasks:
            choice = item.selection
            preset = presets.get(choice.preset_ref)
            if preset is None:
                raise CompilationError(f"Unknown execution preset: {choice.preset_ref}",
                    category="unknown_preset", task_key=item.key,
                    details={"preset_ref": choice.preset_ref, "selection_reason": choice.reason})
            definition = registry.execution_definitions[preset.execution_ref]
            if self.validate_definition is not None:
                try:
                    self.validate_definition(definition)
                except (ValueError, KeyError) as error:
                    raise CompilationError(str(error), category="unsupported_configuration",
                        task_key=item.key, details={"execution_ref": definition.ref, "selection_reason": choice.reason}) from error
            by_key[item.key] = definition.ref
        return CompilationResult(_materialize(request=request, current=current, by_key=by_key,
                                               limits=limits, workspaces=workspaces))


def _materialize(*, request, current, by_key, limits, workspaces):
    keys = [task.key for task in request.request.tasks]
    existing = {task.id for task in current.tasks} if current else set()
    mapping = {}
    for key in keys:
        base = f"task_{key}"
        candidate = base
        counter = 1
        while candidate in existing:
            candidate = f"{base}_{request.id.removeprefix('work_')}_{counter}"
            counter += 1
        existing.add(candidate)
        mapping[key] = candidate
    workspace_ids = [w.workspace_id for w in workspaces]
    tasks = []
    for item in request.request.tasks:
        workspace = item.workspace_id
        if workspace is None and len(workspace_ids) == 1:
            workspace = workspace_ids[0]
        tasks.append(TaskProposal(
            id=mapping[item.key], work_request_id=request.id,
            workflow_agent_kind=WorkflowAgentKind.EXECUTOR,
            execution_ref=by_key[item.key],
            instruction=item.instruction,
            depends_on=[mapping[key] for key in item.depends_on], workspace_id=workspace,
            input_artifacts=list(item.input_artifact_ids),
            input_artifact_bindings=[FutureArtifactBinding(
                source_task=mapping[binding.source_task], output_selector=binding.output_selector,
            ) for binding in item.input_artifact_bindings],
            output_names=list(item.output_names),
        ))
    if current is None:
        candidate = WorkflowProposal(work_request_id=request.id, tasks=tasks)
    else:
        candidate = WorkflowPatch(work_request_id=request.id, based_on_revision=current.revision, add_tasks=tasks)
    validate_workflow_candidate(candidate)
    return candidate
