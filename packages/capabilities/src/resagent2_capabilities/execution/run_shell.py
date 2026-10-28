"""General Bash execution through the shared process component."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from resagent2_components.git import GitWorkspace
from resagent2_components.execution_environment import ExecutionEnvironment
from resagent2_components.process import (
    DEFAULT_OUTPUT_PREVIEW_CHARS, ProcessRunner, output_preview, output_tail,
)
from resagent2_components.workspace import ReadOnlyWorkspaceBoundary, WorkspaceBoundary
from resagent2_components.python_source_evidence import (
    collect_python_source_evidence, validate_review_inspections,
)
from resagent2_runtime import AgentState, ToolObservation
from resagent2_runtime.models import RuntimeModel
from resagent2_runtime.cancellation import OperationCancelled


class RunShellInput(RuntimeModel):
    """One Bash script, preserving whitespace and here-document delimiters."""

    command: str = Field(min_length=1)

    @field_validator("command")
    @classmethod
    def nonempty_command(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must contain non-whitespace text")
        if "\x00" in value:
            raise ValueError("command must not contain NUL bytes")
        return value


class RunShellTool:
    """Execute an approved Bash script and preserve its actual outcomes."""

    name = "run_shell"
    input_model = RunShellInput
    model_guidance = (
        "Run a Linux Bash script in the workspace for installation, checks, training "
        "or evaluation. Inspect, select and manage environments with explicit commands; "
        "use the intended interpreter or activate and execute in the same script. The runtime "
        "checks the exact script under the configured approval mode and requests human "
        "approval when required. Approval never expands existing grants. Keep work in the foreground; "
        "background jobs are not managed. Each call "
        "starts a fresh, non-persistent shell with errexit and pipefail: unhandled "
        "failures stop execution under Bash's errexit rules. Handle expected failures "
        "explicitly with if or ||; preserve the main command's status when running "
        "diagnostics or cleanup afterward. A failing command used as a condition or "
        "inside a command substitution may not stop the script. "
        "Activation, exported variables and directory changes do not persist across calls. "
        "This tool is not suitable for displaying large file bodies. This is trusted host execution, "
        "not a sandbox. A zero exit code alone does not establish sufficient validation. "
        "Receipts preserve stdout/stderr at exact log paths, report byte sizes and "
        "whether the preview is complete or shows bounded beginning/end portions. "
        "Log access and UTF-8 text/size limits are reported separately; a path alone "
        "does not guarantee readable text. Inspect retained output within its access "
        "and format limits instead of rerunning the command to obtain output."
    )

    def __init__(
        self,
        runner: ProcessRunner,
        *,
        timeout_seconds: int,
        log_dir: str,
        extra_env: dict[str, str] | None = None,
        execution_environment: ExecutionEnvironment | None = None,
        allowed: bool = True,
        repository: GitWorkspace | None = None,
        output_preview_chars: int = DEFAULT_OUTPUT_PREVIEW_CHARS,
        log_read_boundary: WorkspaceBoundary | ReadOnlyWorkspaceBoundary | None = None,
    ) -> None:
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self.log_dir = log_dir
        self.extra_env = extra_env
        self.execution_environment = execution_environment
        self.allowed = allowed
        self.repository = repository
        if (isinstance(output_preview_chars, bool)
                or not isinstance(output_preview_chars, int) or output_preview_chars < 1):
            raise ValueError("output_preview_chars must be a positive integer")
        self.output_preview_chars = output_preview_chars
        self.log_read_boundary = log_read_boundary or runner.boundary

    def _output_receipt(self, path: str) -> dict:
        absolute = self.runner.boundary.root / path
        value = output_preview(absolute, limit=self.output_preview_chars)
        boundary = self.log_read_boundary
        read_path = str(absolute)
        try:
            if isinstance(boundary, WorkspaceBoundary):
                read_path = absolute.relative_to(boundary.root).as_posix()
            boundary.resolve_read_file(read_path)
        except (PermissionError, ValueError) as error:
            value.update(read_access="not_authorized", read_error=str(error)[:500])
        except OSError as error:
            value.update(read_access="unavailable", read_error=str(error)[:500])
        else:
            value.update(read_access="authorized", read_path=read_path)
        return value

    def approval_context(self) -> dict:
        """Describe the same shell startup configuration used by the runner."""
        context = self.runner.shell_context(extra_env=self.extra_env)
        if self.execution_environment is not None:
            context["execution_environment"] = self.execution_environment.context()
        return context

    def prepare(self, arguments: BaseModel, *, inspections: list[dict] | None = None) -> dict:
        """Bind the launch and any actual review inspections to one source identity."""
        args = cast(RunShellInput, arguments)
        evidence = collect_python_source_evidence(
            args.command, self.runner.boundary, inspections=inspections,
        )
        prepared = {**self.approval_context(), "source_fingerprint": evidence.fingerprint}
        if inspections:
            prepared["review_inspections"] = [dict(item) for item in inspections]
        return prepared

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        return self._execute(state, arguments)

    def execute_prepared(self, state: AgentState, arguments: BaseModel, prepared: dict) -> ToolObservation:
        """Use the approved source and original startup configuration."""
        startup = dict(prepared)
        environment_context = startup.pop("execution_environment", None)
        try:
            current_environment = self.execution_environment.context() if self.execution_environment else None
            if environment_context != current_environment:
                raise ValueError("Execution environment changed after approval; script was not executed")
        except (OSError, ValueError) as error:
            return ToolObservation(summary=str(error), ok=False, value={"execution_status": "not_executed"})
        fingerprint = startup.pop("source_fingerprint", None)
        inspections = startup.pop("review_inspections", None)
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            return ToolObservation(
                summary="The prepared source fingerprint is missing or invalid; script was not executed",
                ok=False, value={"execution_status": "not_executed"},
            )
        try:
            if inspections is not None:
                inspections = validate_review_inspections(inspections)
        except ValueError as error:
            return ToolObservation(
                summary=str(error), ok=False, value={"execution_status": "not_executed"},
            )
        return self._execute(
            state, arguments, expected_context=startup, expected_source_fingerprint=fingerprint,
            source_inspections=inspections,
        )

    def _execute(
        self, state: AgentState, arguments: BaseModel, *,
        expected_context: dict | None = None,
        expected_source_fingerprint: str | None = None,
        source_inspections: list[dict] | None = None,
    ) -> ToolObservation:
        args = cast(RunShellInput, arguments)
        if not self.allowed:
            raise PermissionError("Process execution is not authorized")
        if not self.runner.boundary.grant.access.unrestricted:
            raise PermissionError("Shell execution requires an unrestricted trusted workspace")
        environment_context = self.execution_environment.context() if self.execution_environment else None
        before = self.repository.snapshot() if self.repository is not None else None
        try:
            if expected_source_fingerprint is not None:
                evidence = collect_python_source_evidence(
                    args.command, self.runner.boundary, inspections=source_inspections,
                )
                if evidence.fingerprint != expected_source_fingerprint:
                    raise PermissionError("Referenced source changed after approval; script was not executed")
            result = self.runner.run_shell(
                args.command,
                log_dir=str(Path(self.log_dir) / uuid4().hex),
                index=1,
                timeout_seconds=self.timeout_seconds,
                extra_env=self.extra_env,
                expected_context=expected_context,
            )
        except (PermissionError, ValueError) as error:
            return ToolObservation(summary=str(error), ok=False,
                                   value={"execution_status": "not_executed"})
        except OperationCancelled as error:
            if environment_context is not None:
                error.details["execution_environment"] = environment_context
            if before is not None:
                error.details["code_before"] = before.tree_hash
            raise
        value = result.model_dump(mode="json")
        if environment_context is not None:
            value["execution_environment"] = environment_context
        value["stdout_output"] = self._output_receipt(result.stdout_path)
        value["stderr_output"] = self._output_receipt(result.stderr_path)
        value["stdout_tail"] = output_tail(self.runner.boundary.root / result.stdout_path)
        value["stderr_tail"] = output_tail(self.runner.boundary.root / result.stderr_path)
        if self.repository is not None:
            value["code_before"] = before.tree_hash
            try:
                changed = self.repository.changed_paths_since(before)
                value["code_after"] = self.repository.snapshot().tree_hash
            except OperationCancelled as error:
                # The script completed, but stopping a later Git diagnostic may
                # be unconfirmed. Preserve both facts through the original call.
                error.details["completed_command"] = value
                raise
            except Exception as error:
                # The command already ran; a later diagnostic must not erase its receipt.
                value["workspace_change_error"] = f"{type(error).__name__}: {str(error)[:500]}"
            else:
                value["changed_paths"] = changed
        return ToolObservation(
            summary=f"Shell command exited with code {result.exit_code}",
            value=value,
            ok=(result.exit_code == 0 and not result.timed_out),
        )
