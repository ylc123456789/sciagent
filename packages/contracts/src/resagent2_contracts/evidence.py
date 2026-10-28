"""Pure evidence requirements shared by the Agent and orchestration boundary."""

from collections.abc import Iterable
import shlex

from .models import ArtifactRef, RunId, VerificationResult


def latest_command_results(results: Iterable[VerificationResult]) -> list[VerificationResult]:
    """Keep the latest outcome of each argv within one Attempt, in event order.

    Only a retry of the same command supersedes its failure. Whitespace and
    quoting are ignored; different executables or arguments are not guessed
    to be equivalent. Callers retain the complete original execution record.
    """
    latest: dict[tuple[str, ...], VerificationResult] = {}
    for result in results:
        key = tuple(shlex.split(result.command))
        if not key:
            raise ValueError("execution record command cannot be empty")
        latest.pop(key, None)
        latest[key] = result
    return list(latest.values())


def missing_required_evidence_kinds(
    required_kinds: Iterable[str],
    *,
    run_id: RunId,
    artifacts: Iterable[ArtifactRef],
    cited_artifact_ids: Iterable[str],
) -> list[str]:
    """Find required kinds not backed by registered, cited evidence.

    Callers supply trusted ArtifactRefs, not model-reported metadata. Imported
    evidence and newly produced evidence follow the same rule; a requirement
    names an artifact kind, not the tool that must run to produce it.
    """
    used = set(cited_artifact_ids)
    present = {
        artifact.kind
        for artifact in artifacts
        if artifact.run_id == run_id and artifact.id in used
    }
    return sorted(set(required_kinds) - present)
