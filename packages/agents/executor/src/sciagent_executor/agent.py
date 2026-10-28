"""One native executor assembled from a frozen trusted execution definition."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sciagent_contracts import (
    AgentOwner, AgentRequest, AgentResult, AgentCheckpoint, ErrorCode,
    ExecutionDefinition, ModuleError, ModuleStatus, task_session_id,
)
from sciagent_components.read_scope import request_read_boundary, tool_result_directory
from sciagent_components.execution_environment import ExecutionEnvironment
from sciagent_capabilities.delegation import delegation_tools
from sciagent_capabilities import (
    CreateFileTool, DeletePathTool, GitDiffTool, ListFilesTool, ReadArtifactTool,
    ReadFileTool, ReplaceTextTool, RunShellTool, SearchTextTool,
)
from sciagent_components import (
    ArtifactReadError, DatasetResolutionError, GitBaseline, GitWorkspace, GitWorkspaceError,
    ProcessRunner, RegisteredArtifactReader, RepoMaterializer, RepoMaterializerError,
    ResourceLayout, WorkspaceBoundary, WorkspacePermissionError,
    dataset_env_overrides, request_dataset_refs, resolve_dataset_refs, preserve_execution_record,
)
from sciagent_runtime import (
    AgentDefinition, AgentLoop, AskUserTool, FinishTool, InMemorySessionStore,
    LLMClient, OperationReviewer, SessionStore,
)
from sciagent_capabilities.permissions import OperationPermissionPolicy
from sciagent_runtime.budget import DeadlineExceededError, execution_budget
from sciagent_runtime.cancellation import OperationCancelled

from .completion import ExecutionCompletionCheck, ExecutionFinishInput, git_patch_candidate
from .configuration import GIT_POLICY, validate_execution_definition
from .context import build_context


class NativeExecutorAgent:
    """Execute with registered components; a config is capability, not permission."""

    def __init__(self, model_resolver: Callable[[str], LLMClient], *,
                 store: SessionStore | None = None,
                 resource_layout: ResourceLayout | None = None,
                 operation_reviewer: OperationReviewer | None = None) -> None:
        self.model_resolver = model_resolver
        self.loop = AgentLoop(store=store or InMemorySessionStore())
        self.resource_layout = resource_layout or ResourceLayout.from_env()
        self.operation_reviewer = operation_reviewer

    @staticmethod
    def _failure(message: str, *, blocked: bool = False) -> AgentResult:
        return AgentResult(status=ModuleStatus.BLOCKED if blocked else ModuleStatus.FAILED,
            report=message, error=ModuleError(code=ErrorCode.INVALID_INPUT, message=message, retryable=False))

    def cancel_question(self, session_id: str, question_id: str, reason: str) -> None:
        self.loop.cancel_question(session_id, question_id, reason)

    def defer_completion(self, session_id: str, reason: str) -> None:
        self.loop.defer_completion(session_id, reason)

    def terminate_session(self, session_id: str, reason: str) -> None:
        self.loop.terminate_session(session_id, reason)

    def read_checkpoint_result(self, *, run_id: str, task_id: str, attempt_number: int) -> AgentCheckpoint | None:
        return self.loop.read_checkpoint_result(
            session_id=task_session_id(run_id, task_id, attempt_number), run_id=run_id,
            task_id=task_id, attempt_number=attempt_number, owner=AgentOwner.EXECUTOR)

    def read_terminal_result(self, *, run_id: str, task_id: str, attempt_number: int) -> AgentResult | None:
        return self.loop.read_terminal_result(
            session_id=task_session_id(run_id, task_id, attempt_number), run_id=run_id,
            task_id=task_id, attempt_number=attempt_number, owner=AgentOwner.EXECUTOR)

    def invoke(self, request: AgentRequest) -> AgentResult:
        try:
            request = AgentRequest.model_validate(request)
        except ValueError as error:
            return self._failure(str(error))
        with execution_budget(max_llm_calls=request.budget.max_llm_calls,
                              timeout_seconds=request.budget.timeout_seconds):
            try:
                return self._invoke(request)
            except DeadlineExceededError as error:
                return AgentResult(status=ModuleStatus.FAILED, report=str(error),
                    error=ModuleError(code=ErrorCode.TIMEOUT, message=str(error), retryable=False))

    def _invoke(self, request: AgentRequest) -> AgentResult:
        if request.agent != AgentOwner.EXECUTOR:
            return self._failure("NativeExecutorAgent received a non-Executor request")
        if request.execution_definition is None:
            return self._failure("Executor requires a frozen execution definition", blocked=True)
        try:
            definition = validate_execution_definition(request.execution_definition)
        except (ValueError, KeyError) as error:
            return self._failure(str(error), blocked=True)
        try:
            self.loop.validate_resume_binding(request, owner=AgentOwner.EXECUTOR)
        except (OSError, ValueError, KeyError) as error:
            return AgentResult(status=ModuleStatus.FAILED, report=str(error),
                error=ModuleError(code=ErrorCode.CONTRACT_ERROR, message=str(error), retryable=False))
        git_tracking = definition.config.preparation == GIT_POLICY[0]
        if git_tracking and request.output_dir is None:
            return self._failure(
                "Git-tracked execution requires a trusted output_dir for generated patch artifacts",
                blocked=True)
        try:
            client = self.model_resolver(definition.config.model_profile)
            if client is None:
                raise ValueError(f"Model profile has no registered client: {definition.config.model_profile}")
        except (ValueError, KeyError) as error:
            return self._failure(str(error), blocked=True)
        if request.workspace is None:
            return self._failure("Executor requires a WorkspaceGrant", blocked=True)
        try:
            if request.workspace_spec is not None:
                materialized = RepoMaterializer().materialize(
                    workspace=Path(request.workspace.root), source=request.workspace_spec)
                if materialized.repo_path.resolve() != Path(request.workspace.root).resolve():
                    return self._failure("Materialized repository is outside the granted root", blocked=True)
            boundary = WorkspaceBoundary(request.workspace)
            repository = GitWorkspace(boundary) if git_tracking else None
            datasets = resolve_dataset_refs(self.resource_layout.dataset_root, request_dataset_refs(request))
        except DeadlineExceededError:
            raise
        except (OSError, ValueError, GitWorkspaceError, WorkspacePermissionError,
                RepoMaterializerError, DatasetResolutionError, ArtifactReadError) as error:
            return self._failure(str(error), blocked=True)
        prior = self.loop.store.load(request.parent_session_id) if request.parent_session_id else None
        baseline = None
        initial_memory = {}
        if git_tracking:
            if request.parent_session_id is not None:
                try:
                    baseline = GitBaseline.from_memory(prior.memory.get("workspace_snapshot"))
                except (OSError, ValueError, KeyError):
                    return self._failure("Resumed execution attempt has no persisted Git baseline", blocked=True)
            else:
                try:
                    baseline = repository.snapshot()
                except (OSError, GitWorkspaceError) as error:
                    return self._failure(str(error), blocked=True)
                initial_memory["workspace_snapshot"] = baseline.to_memory()

        output_root = request.output_dir or str(boundary.root / ".sciagent" / request.task_id)
        environment = None
        shell_env = dataset_env_overrides(self.resource_layout.dataset_root, datasets)
        if ("run_shell" in definition.config.tools and request.permissions.execute_commands
                and boundary.grant.access.unrestricted
                and (prior is None or prior.status.value not in {"completed", "failed"})):
            try:
                environment = (ExecutionEnvironment.from_memory(
                    prior.memory.get("execution_environment"), workspace_root=boundary.root)
                    if prior is not None else ExecutionEnvironment.allocate(
                        workspace_root=boundary.root, output_dir=request.output_dir))
                shell_env.update(environment.env_overrides())
            except (OSError, ValueError) as error:
                return self._failure(str(error), blocked=True)
            initial_memory["execution_environment"] = environment.to_memory()
        tool_result_dir = tool_result_directory(output_root,
            task_session_id(request.run_id, request.task_id, request.attempt_number),
            previous=prior.tool_result_dir if prior else None)
        material_boundary = request_read_boundary(request, command_log_dir=f"{output_root}/commands",
                                                  tool_result_dir=tool_result_dir)
        read_boundary = (request_read_boundary(request, command_log_dir=f"{output_root}/commands",
            tool_result_dir=tool_result_dir, execution_environment_dir=str(environment.root))
            if environment is not None else material_boundary)
        runner = ProcessRunner(boundary)
        reader = RegisteredArtifactReader(request.input_artifacts, run_id=request.run_id)
        factories = {
            "list_files": lambda: ListFilesTool(read_boundary),
            "read_file": lambda: ReadFileTool(read_boundary),
            "search_text": lambda: SearchTextTool(boundary),
            "read_artifact": lambda: ReadArtifactTool(reader),
            "git_diff": lambda: GitDiffTool(repository, baseline=baseline),
            "create_file": lambda: CreateFileTool(boundary),
            "replace_text": lambda: ReplaceTextTool(boundary),
            "delete_path": lambda: DeletePathTool(boundary),
            "run_shell": lambda: RunShellTool(runner, repository=repository,
                timeout_seconds=request.budget.timeout_seconds,
                extra_env=shell_env, execution_environment=environment,
                log_dir=f"{output_root}/commands", log_read_boundary=read_boundary,
                allowed=request.permissions.execute_commands),
        }
        names = definition.config.tools
        tools = [factories[name]() for name in names if name != "request_work"]
        tools.extend((AskUserTool(), FinishTool(ExecutionFinishInput)))
        tools.extend(tool for tool in delegation_tools(request, reader)
                     if tool.name != "request_work" or "request_work" in names)
        tools = tuple(tools)
        runtime_definition = AgentDefinition(
            name="executor", owner=AgentOwner.EXECUTOR,
            tool_result_dir=tool_result_dir if "read_file" in definition.config.tools else None,
            execution_ref=definition.ref, execution_config_hash=definition.config_hash,
            system_prompt=definition.config.system_prompt, tools=tools, llm_client=client,
            context_builder=lambda req, state, limit: build_context(req, state,
                datasets=datasets, reader=reader, read_boundary=material_boundary,
                execution_environment=environment),
            permission_policy=OperationPermissionPolicy(tools, boundary=boundary, request=request),
            operation_reviewer=self.operation_reviewer,
            completion_check=ExecutionCompletionCheck(boundary, repository=repository,
                baseline=baseline, output_dir=request.output_dir),
            max_context_tokens=definition.config.max_context_tokens,
            finalize_result=lambda result, state: self._finalize_result(
                result, repository, baseline, output_dir=output_root, state=state),
        )
        result = self.loop.run(runtime_definition, request,
            session_id=task_session_id(request.run_id, request.task_id, request.attempt_number),
            initial_memory=initial_memory)
        if result.session is not None and result.session.status.value in {"completed", "failed"}:
            return result
        return self._finalize_result(result, repository, baseline, output_dir=output_root)

    def _finalize_result(self, result, repository, baseline, *, output_dir, state=None):
        result = preserve_execution_record(result, self.loop.store, state=state)
        if (repository is None or result.status not in {ModuleStatus.FAILED, ModuleStatus.BLOCKED}
                or any(item.kind == "code_patch" for item in result.artifacts)):
            return result
        try:
            if not repository.changed_paths_since(baseline):
                return result
            patch = git_patch_candidate(repository, baseline, output_dir, diagnostic=True)
        except OperationCancelled as error:
            details = {**error.details, **result.error.details,
                       "diagnostic_patch_error": str(error),
                       "diagnostic_patch_details": {**error.details, "stop_confirmed": error.stopped},
                       "stop_confirmed": error.stopped and result.error.details.get("stop_confirmed") is not False}
            return result.model_copy(update={"error": result.error.model_copy(update={
                "retryable": False, "details": details})})
        except Exception as error:
            # Optional diagnostics must never replace the primary failure or
            # prevent its execution facts and terminal result from being saved.
            return result.model_copy(update={"error": result.error.model_copy(update={
                "retryable": False,
                "details": {**result.error.details, "diagnostic_patch_error": str(error)}})})
        if patch:
            error = result.error.model_copy(update={"retryable": False}) if result.error else None
            return result.model_copy(update={"artifacts": [*result.artifacts, patch], "error": error})
        return result
