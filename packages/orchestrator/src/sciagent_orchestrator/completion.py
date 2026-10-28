"""Check final artifact evidence and render the accepted research report."""

from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field
from sciagent_contracts import (
    AgentOwner, ArtifactCandidate, ArtifactRef, AttemptStatus, ConclusionRequirements,
    ModuleStatus, ModuleError, ScientificOpinion, SessionStatus, TaskStatus,
    WorkRequestStatus, WorkTaskResult, WorkflowAgentKind, missing_required_evidence_kinds,
)
from sciagent_components.artifacts import RegisteredArtifactReader
from .handoffs import read_json, check_acceptance
from .models import CompletionViolation, CompletionViolationCode


class FinalReportData(BaseModel):
    """Validated inputs to deterministic final-report rendering."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    goal: str
    opinion: ScientificOpinion
    evidence: list[ArtifactRef]
    execution_issues: list[WorkTaskResult] = Field(default_factory=list)
    work_errors: dict[str, ModuleError] = Field(default_factory=dict)


@dataclass(frozen=True)
class CompletionValidation:
    """Completion violations or the accepted report data."""
    violations: tuple[CompletionViolation, ...] = ()
    report: FinalReportData | None = None

    @property
    def ok(self):
        return not self.violations and self.report is not None


@dataclass(frozen=True)
class RenderedFinalReport:
    """Rendered report bytes and registration metadata."""
    candidate: ArtifactCandidate
    content: str


class ScientificCompletionValidator:
    """Check evidence provenance and terminal execution state."""
    def __init__(self, artifact_registry):
        self._artifact_registry = artifact_registry

    def validate(self, run, result, refs):
        violations = []

        def reject(code, message, ids=(), *, subject=None):
            violations.append(CompletionViolation(
                code=code, message=message, subject=subject, related_ids=list(ids),
            ))

        if result.status not in {ModuleStatus.COMPLETED, ModuleStatus.COMPLETED_WITH_WARNINGS}:
            reject(CompletionViolationCode.INVALID_OPINION, "Scientific has not completed")
        session = result.session
        if (session is None or session.module != AgentOwner.SCIENTIFIC or session.status != SessionStatus.COMPLETED
                or run.scientific_session is None or session.id != run.scientific_session.id):
            reject(CompletionViolationCode.INVALID_SESSION, "completed result requires its bound Scientific session")
        if run.pending_question or run.child_questions or any(work.status not in {WorkRequestStatus.CONSUMED, WorkRequestStatus.FAILED, WorkRequestStatus.CANCELLED} for work in run.work_requests if work.owner_task_id is not None):
            reject(CompletionViolationCode.ACTIVE_CONTROL_STATE, "active control state prevents completion")
        if any(task.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.NEEDS_USER_INPUT, TaskStatus.WAITING_FOR_WORK} for task in (run.workflow.tasks if run.workflow else [])):
            reject(CompletionViolationCode.ACTIVE_CONTROL_STATE, "non-terminal tasks prevent completion")
        try:
            opinions = [ref for ref in refs if ref.kind == "scientific_opinion"]
            if len(opinions) != 1:
                raise ValueError("completion requires one scientific opinion")
            for ref in opinions:
                if run.artifacts.get(ref.id) != ref or ref.producer != AgentOwner.SCIENTIFIC or not session or ref.session_id != session.id:
                    raise ValueError("Scientific completion artifact has invalid ownership")
            opinion = read_json(opinions[0], ScientificOpinion)
            cited = set(opinion.evidence_artifact_ids)
            if len(cited) != len(opinion.evidence_artifact_ids):
                raise ValueError("duplicate opinion citations")
            reader = RegisteredArtifactReader(list(run.artifacts.values()), run_id=run.run_id)
            for artifact_id in cited:
                if reader.resolve_ref(artifact_id) is None:
                    raise ValueError(f"citation is not registered in this Run: {artifact_id}")
                reader.verify(artifact_id)
            requirement = run.conclusion_requirements_ref
            if requirement is None or run.artifacts.get(requirement.id) != requirement or requirement.kind != "conclusion_requirements" or requirement.run_id != run.run_id:
                raise ValueError("Run conclusion requirement binding missing or invalid")
            requirements = read_json(requirement, ConclusionRequirements)
            for name in self._artifact_registry.missing_required_artifacts(
                requirements.required_artifacts, run_id=run.run_id, artifacts=run.artifacts,
            ):
                reject(CompletionViolationCode.REQUIRED_ARTIFACT_MISSING,
                       "required artifact was not produced", subject=name)
            missing = missing_required_evidence_kinds(requirements.required_evidence_kinds, run_id=run.run_id, artifacts=run.artifacts.values(),
                                                      cited_artifact_ids=cited)
            if missing:
                reject(CompletionViolationCode.MISSING_EVIDENCE_KIND, "missing registered and cited evidence kinds", missing)
        except (ValueError, OSError, KeyError) as error:
            reject(CompletionViolationCode.INVALID_OPINION, str(error))
            return CompletionValidation(tuple(violations))
        for task in run.workflow.tasks if run.workflow else []:
            if task.id == "task_root":
                continue
            if task.status != TaskStatus.COMPLETED:
                continue
            last = task.attempts[-1] if task.attempts else None
            if (task.workflow_agent_kind != WorkflowAgentKind.EXECUTOR
                    or task.execution_ref not in run.execution_definitions) or last is None or last.status not in {AttemptStatus.COMPLETED, AttemptStatus.COMPLETED_WITH_WARNINGS} or last.finished_at is None or last.error:
                reject(CompletionViolationCode.INCONSISTENT_TASK_RESULT, "completed task lacks valid execution history", [task.id])
                continue
            try:
                outputs = [run.artifacts[key] for key in last.artifact_ids]
                if any(ref.run_id != run.run_id or ref.task_id != task.id or ref.attempt_number != last.number
                       or ref.producer.value != task.workflow_agent_kind.value for ref in outputs
                       if ref.kind != "module_report"):
                    raise ValueError("completed task has foreign output artifacts")
                check_acceptance(run, task, last, outputs)
            except (ValueError, OSError, KeyError) as error:
                reject(CompletionViolationCode.INCONSISTENT_TASK_RESULT, str(error), [task.id])
        issues = [item for work in run.work_requests if work.outcome for item in work.outcome.tasks if item.status != "completed"]
        work_errors = {work.id: work.error for work in run.work_requests
                       if work.status in {WorkRequestStatus.FAILED, WorkRequestStatus.CANCELLED} and work.error is not None}
        if (issues or work_errors) and not opinion.limitations:
            reject(CompletionViolationCode.MISSING_LIMITATIONS, "failed execution work requires stated limitations")
        if violations:
            return CompletionValidation(tuple(violations))
        return CompletionValidation(report=FinalReportData(run_id=run.run_id, goal=run.request.goal, opinion=opinion,
                                                          evidence=[run.artifacts[key] for key in opinion.evidence_artifact_ids],
                                                          execution_issues=issues, work_errors=work_errors))


class FinalReportRenderer:
    """Render an accepted opinion and its provenance as Markdown."""
    def render(self, data):
        opinion = data.opinion
        lines = ["# Research Run Final Report", "", f"- Run: `{data.run_id}`", f"- Verdict: `{opinion.verdict.value}`",
                 "", "## Goal", "", data.goal, "", "## Scientific opinion", "", opinion.statement, "", "## Evidence", ""]
        for ref in data.evidence:
            lines.extend([f"- `{ref.id}`: {ref.summary}", f"  kind: `{ref.kind}`; producer: `{ref.producer.value}`",
                          f"  sha256: `{ref.sha256}`", f"  uri: `{ref.uri}`"])
        if not data.evidence:
            lines.append("- No evidence artifacts cited.")
        for title, values in [("Limitations", opinion.limitations), ("Unresolved questions", opinion.unresolved_questions),
                              ("Recommended next steps", opinion.recommended_next_steps)]:
            lines.extend(["", "## " + title, "", *["- " + item for item in values or ["None."]]])
        lines.extend(["", "## Non-completed delegated work", "",
                      "This section lists non-completed task outcomes and recorded errors from failed or cancelled WorkRequests. "
                      "It does not summarize individual tool failures or permission decisions.", ""])
        lines.extend(f"- WorkRequest `{work_id}`: {error.message}" for work_id, error in data.work_errors.items())
        lines.extend(f"- `{issue.task_id}` ({issue.status.value}): {issue.error.message if issue.error else 'No detailed error recorded'}" for issue in data.execution_issues)
        if not data.execution_issues and not data.work_errors:
            lines.append("- None in this scope.")
        content = "\n".join(lines) + "\n"
        return RenderedFinalReport(ArtifactCandidate(kind="final_report", path="final_report.md", media_type="text/markdown",
                                                     summary=f"Final report for {data.run_id}", metadata={"source_type": "final_report"}), content)
