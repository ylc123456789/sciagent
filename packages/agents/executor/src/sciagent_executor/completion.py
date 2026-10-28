"""Composable actual execution and optional Git-change delivery."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from pydantic import Field

from sciagent_contracts import ArtifactCandidate, ArtifactRef, SYSTEM_GENERATED_ARTIFACT_KINDS
from sciagent_components import GitBaseline, GitWorkspace, WorkspaceBoundary, execution_record
from sciagent_components.artifacts import ArtifactCandidateError, check_task_output_artifacts
from sciagent_runtime import AgentState, CompletionDecision, FinishCandidate
from sciagent_runtime.tools import FinishInput


def git_patch_candidate(repository, baseline, output_dir, *, diagnostic=False):
    """Write exact patch bytes under the trusted Attempt output root."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # A fresh directory avoids overwriting submitted outputs or following a
    # workspace-selected changes.patch symlink.
    directory = root / f"git-patch-{uuid4().hex}"
    directory.mkdir(exist_ok=False)
    destination = directory / ("failed_changes.patch" if diagnostic else "changes.patch")
    repository.write_patch_since(baseline, destination)
    return ArtifactCandidate(
        kind="code_patch", path=destination.relative_to(root).as_posix(),
        media_type="text/x-diff",
        summary=("Diagnostic patch from failed execution attempt" if diagnostic
                 else "Git patch from this execution attempt"),
        metadata={"diagnostic": True} if diagnostic else {},
    )


class ExecutionFinishInput(FinishInput):
    """Executors propose new files; existing evidence keeps its original owner."""

    artifacts: list[ArtifactCandidate] = Field(default_factory=list, description=(
        "Explicitly selected deliverable files or inline content produced for this Task. "
        "Changed workspace files are not automatically delivered individually; submit "
        "an ArtifactCandidate for each file that should be a separate output. Do not resubmit "
        "registered ArtifactRefs or child/input artifacts as new outputs; reference "
        "their existing IDs in report. The Registry assigns output IDs and provenance."
    ))


class ExecutionCompletionCheck:
    def __init__(self, boundary: WorkspaceBoundary, *, repository: GitWorkspace | None = None,
                 baseline: GitBaseline | None = None, output_dir: str | None = None) -> None:
        if (repository is None) != (baseline is None):
            raise ValueError("Git completion needs both repository and Attempt baseline")
        if repository is not None and output_dir is None:
            raise ValueError("Git completion requires a trusted output_dir for generated patch artifacts")
        self.boundary = boundary
        self.repository = repository
        self.baseline = baseline
        self.output_dir = output_dir

    def evaluate(self, state: AgentState, candidate: FinishCandidate | None) -> CompletionDecision:
        if candidate is None:
            return CompletionDecision(complete=False)
        if any(isinstance(item, ArtifactRef) for item in candidate.artifacts):
            return CompletionDecision(complete=False, report=(
                "existing_artifact_as_output: Executor finish accepts new ArtifactCandidates, "
                "not registered ArtifactRefs. Keep child/input evidence under its original "
                "identity and reference its ID in report; submit only new file/content outputs."
            ))
        changed = self.repository.changed_paths_since(self.baseline) if self.repository is not None else []
        for path in changed:
            self.boundary.resolve_write_file(path)
        artifacts = list(candidate.artifacts)
        if any(item.kind in SYSTEM_GENERATED_ARTIFACT_KINDS for item in artifacts):
            return CompletionDecision(complete=False,
                report="Patches and execution records are generated from actual changes and command observations")
        record = execution_record(state)
        if record is not None:
            artifacts.insert(0, record)
        if changed:
            artifacts.append(git_patch_candidate(self.repository, self.baseline, self.output_dir))
        try:
            check_task_output_artifacts(artifacts, grant=self.boundary.grant, output_dir=self.output_dir)
        except ArtifactCandidateError as error:
            return CompletionDecision(complete=False, report=f"{error.code}: {error}")
        return CompletionDecision(complete=True, report=candidate.report, artifacts=artifacts)
