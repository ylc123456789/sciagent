"""Freeze command facts from the owning Agent's original tool receipts."""

from __future__ import annotations

import json

from resagent2_contracts import ArtifactCandidate, AgentResult, ModuleStatus, VerificationResult, WarningRecord
from resagent2_runtime import AgentState, SessionStore


_STARTUP_FIELDS = ("cwd", "shell", "shell_args", "pipefail", "environment_fingerprint", "execution_environment")
_CODE_FIELDS = ("code_before", "code_after", "changed_paths", "workspace_change_error")
_INTERRUPTED_FIELDS = (
    "execution_status", "stop_confirmed", "operation", "command", "pid", "exit_code", "timed_out",
    "stdout_path", "stderr_path", "duration_seconds", "files_retained",
    "unconfirmed_processes", "cleanup_error", "stdout_tail", "stderr_tail",
) + _STARTUP_FIELDS + _CODE_FIELDS
_HANDOFF_OUTCOMES = frozenset({
    ModuleStatus.COMPLETED, ModuleStatus.COMPLETED_WITH_WARNINGS,
    ModuleStatus.FAILED, ModuleStatus.BLOCKED, ModuleStatus.INTERRUPTED,
    ModuleStatus.NEEDS_USER_INPUT, ModuleStatus.YIELDED,
})


def execution_record(state: AgentState) -> ArtifactCandidate | None:
    """Preserve completed and interrupted outcomes without inferring success.

    Original turns remain complete even when their model projection is compacted.
    Reading only receipts also excludes proposed, denied and unstarted commands.
    Interruption facts may lack a known exit code or log path; keep the actual
    fields rather than constructing a successful VerificationResult for them.
    """
    records = []
    for turn in state.tool_turns:
        for call in turn.tool_calls:
            if call.name != "run_shell" or call.id not in turn.tool_results:
                continue
            receipt = json.loads(turn.tool_results[call.id])
            value = receipt.get("value")
            if not isinstance(value, dict):
                continue
            completed = value.get("completed_command", value)
            if isinstance(completed, dict) and "execution_status" not in completed and "exit_code" in completed:
                fields = {key: completed[key] for key in VerificationResult.model_fields if key in completed}
                records.append({
                    **VerificationResult.model_validate(fields).model_dump(mode="json"),
                    **{key: completed[key] for key in (*_STARTUP_FIELDS, *_CODE_FIELDS) if key in completed},
                })
            if (value.get("execution_status") in {"interrupted", "unknown"}
                    and value.get("operation") == "process" and "command" in value):
                records.append({key: value[key] for key in _INTERRUPTED_FIELDS if key in value})
    if not records:
        return None
    return ArtifactCandidate(
        kind="execution_record", path="execution_record.json", media_type="application/json",
        summary="Recorded command outcomes; interrupted files are retained, not validated deliveries",
        content=json.dumps({"results": records}, ensure_ascii=False),
    )


def preserve_execution_record(
    result: AgentResult, store: SessionStore, *, state: AgentState | None = None,
) -> AgentResult:
    """Attach actual command evidence through the usual public output.

    Completion checks already include the record. Failure, interruption and a
    business/approval wait preserve the same cumulative snapshot. That lets
    cancellation while waiting use already accepted artifacts, without an
    upper layer inspecting a private Session. Reception owns snapshot reuse.
    Diagnostics must not overwrite an execution error or stop status.
    """
    if (result.status not in _HANDOFF_OUTCOMES or result.session is None
            or any(item.kind == "execution_record" for item in result.artifacts)):
        return result
    try:
        record = execution_record(state if state is not None else store.load(result.session.id))
    except (OSError, ValueError, KeyError, TypeError) as error:
        if result.error is not None:
            return result.model_copy(update={"error": result.error.model_copy(update={
                "details": {**result.error.details, "execution_record_error": str(error)},
            })})
        warning = WarningRecord(code="execution_record_unavailable", message=str(error) or type(error).__name__)
        status = (ModuleStatus.COMPLETED_WITH_WARNINGS
                  if result.status == ModuleStatus.COMPLETED else result.status)
        return result.model_copy(update={"status": status, "warnings": [*result.warnings, warning]})
    return result if record is None else result.model_copy(update={"artifacts": [record, *result.artifacts]})
