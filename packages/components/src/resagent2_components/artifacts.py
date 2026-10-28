"""Shared output facts and read-only access to registered ArtifactRefs."""

from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse
from urllib.request import url2pathname

from resagent2_contracts import ArtifactCandidate, ArtifactRef, RunId, SessionId, WorkspaceGrant

from resagent2_runtime.budget import current_budget
from resagent2_runtime.cancellation import check_cancelled

from .workspace import WorkspaceBoundary, WorkspacePermissionError
from .text import MAX_READ_CHARS, decode_text_chunks, slice_text_chunks, validate_text_window


class ArtifactCandidateError(ValueError):
    """A submitted output can be corrected by its author, without changing policy."""

    def __init__(self, code: str, message: str, *, subject: str | None = None):
        self.code = code
        self.subject = subject
        super().__init__(message)


def check_output_names(artifacts) -> None:
    """Logical names must identify exactly one output."""
    names = set()
    for item in artifacts:
        if item.output_name is not None:
            if item.output_name in names:
                raise ArtifactCandidateError(
                    "duplicate_output_name", f"Use a unique output_name for each output artifact: {item.output_name}",
                    subject=item.output_name,
                )
            names.add(item.output_name)


def resolve_artifact_source(
    path: str, *, grant: WorkspaceGrant | None, output_dir: str | None = None,
) -> tuple[str, Path, Path]:
    """Resolve one file under the same authorized roots at finish and registration.

    Missing or ambiguous submitted files are correctable. Authorization errors
    and other IO failures propagate; they are never converted into acceptance.
    """
    if grant is None and output_dir is None:
        raise WorkspacePermissionError("workspace-file ArtifactCandidate requires a workspace grant")
    roots = []
    if grant is not None:
        roots.append(("workspace", Path(grant.root).resolve(strict=True)))
    if output_dir is not None:
        roots.append(("output_dir", Path(output_dir).resolve()))
    matches = [(label, root, (root / path).resolve())
               for label, root in roots if (root / path).is_file()]
    if len(matches) != 1:
        raise ArtifactCandidateError(
            "artifact_path_missing" if not matches else "artifact_path_ambiguous",
            f"artifact path is missing or ambiguous across authorized roots: {path}",
            subject=path,
        )
    label, root, source = matches[0]
    if not source.is_file() or not source.is_relative_to(root):
        raise WorkspacePermissionError("artifact path is outside workspace or not a file")
    if label == "workspace":
        source = WorkspaceBoundary(grant).resolve_read_file(path)
    return label, root, source


def check_task_output_artifacts(artifacts, *, grant, output_dir=None) -> None:
    """Check submitted names and file facts; registration supplies provenance."""
    check_output_names(artifacts)
    for item in artifacts:
        if isinstance(item, ArtifactCandidate) and item.content is None:
            resolve_artifact_source(item.path, grant=grant, output_dir=output_dir)


class ArtifactReadError(ValueError):
    """Raised when a registered ArtifactRef cannot be verified and read."""


class RegisteredArtifactReader:
    """Read explicitly granted or live-authorized refs within one Run only."""

    def __init__(
        self,
        artifacts: list[ArtifactRef],
        *,
        run_id: RunId,
        resolve: Callable[[str], ArtifactRef | None] | None = None,
    ) -> None:
        self._run_id = run_id
        self._artifacts = {artifact.id: artifact for artifact in artifacts}
        self._resolve = resolve

    def resolve_ref(self, artifact_id: str) -> ArtifactRef | None:
        """Return an authorized, Run-scoped Ref without reading its file."""
        artifact = self._artifacts.get(artifact_id)
        if artifact is None and self._resolve is not None:
            artifact = self._resolve(artifact_id)
        # Check provenance before touching a path, even if a resolver is buggy.
        if (
            artifact is None
            or artifact.id != artifact_id
            or artifact.run_id != self._run_id
        ):
            return None
        return artifact

    def _resolve_file(self, artifact_id: str) -> tuple[ArtifactRef, Path]:
        artifact = self.resolve_ref(artifact_id)
        if artifact is None:
            raise ArtifactReadError(f"unknown artifact id: {artifact_id}")
        parsed = urlparse(artifact.uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ArtifactReadError("only local file ArtifactRefs are readable")
        path = Path(url2pathname(parsed.path))
        if not path.is_file():
            raise ArtifactReadError("artifact file is missing")
        return artifact, path

    def verify(self, artifact_id: str) -> None:
        """Verify frozen bytes in bounded memory, without decoding or observing."""
        artifact, path = self._resolve_file(artifact_id)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != artifact.sha256:
            raise ArtifactReadError("artifact sha256 does not match frozen content")

    def verified_path(self, artifact_id: str) -> Path:
        """Resolve a local frozen source for parsers without granting read status."""
        self.verify(artifact_id)
        return self._resolve_file(artifact_id)[1]

    def read_text(
        self,
        artifact_id: str,
        *,
        max_chars: int = MAX_READ_CHARS,
        start_line: int | None = None,
        end_line: int | None = None,
        start_char: int = 0,
        end_char: int | None = None,
    ) -> dict:
        """Verify frozen bytes, select physical lines, then a character window."""
        validate_text_window(
            start_line=start_line, end_line=end_line,
            start_char=start_char, end_char=end_char,
        )
        artifact, path = self._resolve_file(artifact_id)

        def verified_chunks():
            digest = hashlib.sha256()
            budget = current_budget()
            with path.open("rb") as handle:
                while True:
                    check_cancelled()
                    if budget is not None:
                        budget.remaining_timeout()
                    chunk = handle.read(64 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    yield chunk
            if digest.hexdigest() != artifact.sha256:
                raise ArtifactReadError("artifact sha256 does not match frozen content")

        try:
            window = slice_text_chunks(
                decode_text_chunks(verified_chunks()),
                start_line=start_line, end_line=end_line,
                start_char=start_char, end_char=end_char, max_chars=max_chars,
            )
        except ArtifactReadError:
            raise
        except ValueError as error:
            raise ArtifactReadError(str(error)) from error
        return {
            "artifact_id": artifact.id,
            "kind": artifact.kind,
            "summary": artifact.summary,
            "provenance": artifact.model_dump(
                mode="json", include={"producer", "task_id", "attempt_number", "session_id"},
                exclude_none=True,
            ),
            **window,
        }


def missing_required_artifacts(
    required_names: Iterable[str], *, artifact_ids: Iterable[str],
    reader: RegisteredArtifactReader,
) -> list[str]:
    """Check exact output names against authorized, frozen Run artifacts.

    Only registered refs count; filenames, metadata and finish candidates do
    not establish delivery. The reader verifies frozen bytes in bounded memory,
    without decoding or marking them as observed by the Scientific Agent.
    Corrupt or unreadable registered files remain errors, not missing outputs.
    """
    required = list(dict.fromkeys(required_names))
    if not required:
        return []
    present = set()
    for artifact_id in dict.fromkeys(artifact_ids):
        ref = reader.resolve_ref(artifact_id)
        if ref is not None and ref.output_name in required:
            reader.verify(artifact_id)
            present.add(ref.output_name)
    return [name for name in required if name not in present]


def media_type_for(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


class ArtifactRegistrationPort(Protocol):
    """Scientific Tool seam for freezing results via the ResAgent Registry.

    The composition root adapts the orchestrator ArtifactRegistry to this shape;
    capabilities must not import the orchestrator.
    """

    def register_scientific(
        self,
        candidate: ArtifactCandidate,
        *,
        run_id: RunId,
        session_id: SessionId,
        content_bytes: bytes | None = None,
        operation_key: str,
        output_position: int,
    ) -> ArtifactRef:
        """Freeze trusted tool output; binary content is never an LLM parameter."""

    def list_artifacts(self, *, run_id: RunId, session_id: SessionId) -> list[ArtifactRef]:
        """List only material produced by this Scientific Session in this Run."""

    def resolve(self, artifact_id: str, *, run_id: RunId, session_id: SessionId) -> ArtifactRef | None:
        """Return a live-authorized artifact of this Run, or ``None``.

        This lets the Scientific Agent's ``read_artifact`` see an artifact
        (e.g. a literature search) registered earlier in the same turn.
        An artifact registered for another Run must never be returned.
        """

    def acknowledge_registration(self, *, run_id: RunId, operation_key: str) -> None:
        """Clear recovery records after the Session durably saves its receipt."""
