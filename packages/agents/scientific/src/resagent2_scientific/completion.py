"""Validate registered citations and required Scientific deliverables."""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import ValidationError

from resagent2_contracts import (
    AgentOwner, ArtifactCandidate, ArtifactRef, ScientificOpinion, WorkTaskResult,
    missing_required_evidence_kinds, SCIENTIFIC_ARTIFACT_KINDS,
    SYSTEM_GENERATED_ARTIFACT_KINDS,
)
from resagent2_components import RegisteredArtifactReader, read_artifact_json
from resagent2_components.artifacts import (
    ArtifactCandidateError, check_output_names, missing_required_artifacts,
)
from resagent2_runtime import AgentState, CompletionDecision, FinishCandidate


SCIENTIFIC_FINISH_ARTIFACT_KINDS = SCIENTIFIC_ARTIFACT_KINDS - SYSTEM_GENERATED_ARTIFACT_KINDS


def _observed_artifact_ids(state: AgentState) -> list[str]:
    """Historical access log only; never evidence or completion authority."""
    return list(dict.fromkeys(state.memory.get("read_artifact_ids", [])))


def cited_artifacts(cited_ids: list[str], *, run_id: str,
                    reader: RegisteredArtifactReader | None) -> list[ArtifactRef]:
    """Check registered source identity and bytes without recording a read."""
    refs = []
    for artifact_id in cited_ids:
        ref = reader.resolve_ref(artifact_id) if reader is not None else None
        if ref is None or ref.run_id != run_id:
            raise ValueError(f"Citation is not an authorized artifact of this Run: {artifact_id}")
        reader.verify(artifact_id)
        refs.append(ref)
    return refs


class ScientificCompletionCheck:
    def __init__(
        self, unresolved_task_outcomes: list[WorkTaskResult],
        required_evidence_kinds: list[str] | None = None, *,
        required_artifacts: list[str] | None = None,
        resolve_artifact: Callable[[str], ArtifactRef | None] | None = None,
        reader: RegisteredArtifactReader | None = None,
        input_artifact_ids: list[str] | None = None,
        registered_artifacts: Callable[[], list[ArtifactRef]] | None = None,
        failed_work: bool = False,
    ) -> None:
        self._has_unresolved_work = bool(unresolved_task_outcomes) or failed_work
        self._required_evidence_kinds = required_evidence_kinds or []
        self._required_artifacts = required_artifacts or []
        self._resolve_artifact = resolve_artifact
        self._reader = reader
        self._input_artifact_ids = frozenset(input_artifact_ids or [])
        self._registered_artifacts = registered_artifacts

    def evaluate(self, state: AgentState, candidate: FinishCandidate | None) -> CompletionDecision:
        if candidate is None:
            return CompletionDecision(complete=False)
        if candidate.status == "failed":
            return CompletionDecision(
                complete=False,
                report="Scientific finish requires a completed opinion. Use an inconclusive "
                "verdict when evidence is insufficient, request more work, or ask the user; "
                "required evidence and delivery checks still apply.",
            )
        output_error = self._output_error(state, candidate)
        if output_error is not None:
            return CompletionDecision(complete=False, report=output_error)
        opinions = [item for item in candidate.artifacts if item.kind == "scientific_opinion"]
        if len(opinions) != 1:
            return CompletionDecision(
                complete=False, report="Submit exactly one scientific_opinion JSON artifact",
            )
        try:
            item = opinions[0]
            if isinstance(item, ArtifactCandidate):
                if item.content is None:
                    raise ValueError("scientific_opinion must provide JSON content")
                opinion = ScientificOpinion.model_validate_json(item.content)
            elif self._reader is not None:
                opinion = read_artifact_json(self._reader, item.id, ScientificOpinion)
            else:
                raise ValueError("Registered opinion has no authorized reader")
        except (ValueError, ValidationError) as error:
            return CompletionDecision(complete=False, report=f"Opinion artifact is invalid: {error}")
        try:
            artifacts = cited_artifacts(
                opinion.evidence_artifact_ids, run_id=state.run_id, reader=self._reader,
            )
        except (ValueError, OSError) as error:
            return CompletionDecision(complete=False, report=f"Invalid evidence citation: {error}")
        missing = missing_required_evidence_kinds(
            self._required_evidence_kinds, run_id=state.run_id, artifacts=artifacts,
            cited_artifact_ids=opinion.evidence_artifact_ids,
        )
        if missing:
            return CompletionDecision(
                complete=False,
                report="Still missing required evidence of kind " + ", ".join(missing)
                + "; cite registered artifacts of those kinds",
            )
        if self._required_artifacts:
            if self._reader is None:
                raise ValueError("Required artifacts have no authorized registry reader")
            registered = self._registered_artifacts() if self._registered_artifacts else []
            available_ids = [
                *sorted(self._input_artifact_ids),
                *[item.id for item in registered if item.run_id == state.run_id],
                *state.authorized_artifact_ids,
                *opinion.evidence_artifact_ids,
                *[item.id for item in candidate.artifacts if isinstance(item, ArtifactRef)],
            ]
            # This pre-registration boundary may propose its own valid outputs.
            # The receiving gate must still register and independently verify them.
            pending_names = {
                item.output_name for item in candidate.artifacts
                if isinstance(item, ArtifactCandidate) and item.output_name is not None
            }
            missing_outputs = missing_required_artifacts(
                [name for name in self._required_artifacts if name not in pending_names],
                artifact_ids=available_ids, reader=self._reader,
            )
            if missing_outputs:
                return CompletionDecision(
                    complete=False,
                    report="\n".join(
                        f"required_artifact_missing: required artifact was not produced; subject={name}"
                        for name in missing_outputs
                    ) + "\nRequest the missing work or ask the user how to proceed; "
                    "delivery requires a registered artifact with that exact output_name.",
                )
        if self._has_unresolved_work and not opinion.limitations:
            return CompletionDecision(
                complete=False,
                report="State at least one limitation because failed or blocked requested work remains",
            )
        return CompletionDecision(
            complete=True, report=candidate.report, artifacts=list(candidate.artifacts),
        )

    def _output_error(self, state: AgentState, candidate: FinishCandidate) -> str | None:
        returned_ids = set()
        try:
            check_output_names(candidate.artifacts)
        except ArtifactCandidateError as error:
            return f"{error.code}: {error}"
        for item in candidate.artifacts:
            if isinstance(item, ArtifactRef):
                registered = self._resolve_artifact(item.id) if self._resolve_artifact else None
                if (
                    registered != item or item.run_id != state.run_id
                    or item.producer != AgentOwner.SCIENTIFIC
                    or item.session_id != state.session_id
                    or item.id in self._input_artifact_ids
                ):
                    return (
                        "Input, foreign or unregistered artifacts cannot be returned as new outputs; "
                        "cite existing evidence IDs in scientific_opinion.evidence_artifact_ids"
                    )
                if item.id in returned_ids:
                    return "Return each registered artifact only once"
                returned_ids.add(item.id)
            elif item.kind not in SCIENTIFIC_FINISH_ARTIFACT_KINDS:
                return (
                    f"Unsupported Scientific finish artifact kind: {item.kind}. "
                    "New artifact kinds allowed: " + ", ".join(sorted(SCIENTIFIC_FINISH_ARTIFACT_KINDS))
                    + ". Cite existing evidence IDs in scientific_opinion.evidence_artifact_ids; "
                    "tool and system records are returned automatically"
                )
        return None
