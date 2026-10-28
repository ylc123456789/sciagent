"""Immutable artifact registration with durable UUID allocation before freezing."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from sciagent_components.artifacts import (
    ArtifactCandidateError, RegisteredArtifactReader, missing_required_artifacts,
    resolve_artifact_source,
)
from sciagent_contracts import (
    AgentOwner, ArtifactCandidate, ArtifactImport, ArtifactRef,
    SYSTEM_ARTIFACT_PROVENANCE, SCIENTIFIC_ARTIFACT_KINDS, new_artifact_id,
)
from sciagent_runtime.budget import current_budget
from sciagent_runtime.cancellation import check_cancelled

from .models import PendingRegistration
from .store import RunStore


class ArtifactRegistrationError(ValueError):
    """A candidate cannot become a provenance-safe immutable ArtifactRef."""


def _checkpoint(checkpoint=None):
    check_cancelled()
    budget = current_budget()
    if budget is not None:
        budget.remaining_timeout()
    if checkpoint is not None:
        checkpoint()


def _sha256(path: Path, *, checkpoint=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            _checkpoint(checkpoint)
            chunk = handle.read(64 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _file_digest(path, checkpoint=None):
    # Keep the single-path seam for existing callers and fault injection.
    return _sha256(path) if checkpoint is None else _sha256(path, checkpoint=checkpoint)


def _bytes_digest(encoded, checkpoint=None):
    digest = hashlib.sha256()
    for offset in range(0, len(encoded), 64 * 1024):
        _checkpoint(checkpoint)
        digest.update(memoryview(encoded)[offset:offset + 64 * 1024])
    _checkpoint(checkpoint)
    return digest.hexdigest()


def _copy_file(source, destination, *, checkpoint=None):
    with source.open("rb") as reader, destination.open("wb") as writer:
        while True:
            _checkpoint(checkpoint)
            chunk = reader.read(64 * 1024)
            if not chunk:
                return
            writer.write(chunk)


def _write_bytes(encoded, destination, *, checkpoint=None):
    with destination.open("wb") as writer:
        for offset in range(0, len(encoded), 64 * 1024):
            _checkpoint(checkpoint)
            writer.write(memoryview(encoded)[offset:offset + 64 * 1024])
    _checkpoint(checkpoint)


def _ref_path(ref):
    parsed = urlparse(ref.uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ArtifactRegistrationError("only registered local artifacts may be read")
    return Path(url2pathname(parsed.path))


@dataclass(frozen=True)
class _PreparedRegistration:
    ref: ArtifactRef
    operation_key: str | None = None
    allocation: PendingRegistration | None = None
    source: Path | None = None
    encoded: bytes | None = None


def _resolve_import_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
        return Path(url2pathname(parsed.path)).expanduser().resolve()
    if parsed.scheme == "":
        return Path(parsed.path).expanduser().resolve()
    raise ArtifactRegistrationError(f"unsupported import uri scheme: {parsed.scheme!r}")


class ArtifactRegistry:
    """Validate sources and freeze them using one content-independent identity."""

    def __init__(self, root: str | Path, store: RunStore | None = None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store

    def missing_required_artifacts(self, required_names, *, run_id, artifacts):
        reader = RegisteredArtifactReader(list(artifacts.values()), run_id=run_id)
        return missing_required_artifacts(required_names, artifact_ids=artifacts, reader=reader)

    def _run(self, run_id, run):
        if run is not None:
            if run.run_id != run_id:
                raise ArtifactRegistrationError("registration Run mismatch")
            return run
        return self.store.load(run_id) if self.store is not None else None

    def _save(self, run):
        if run is not None:
            if self.store is not None:
                self.store.save(run)

    def _verify_ref(self, ref, *, checkpoint=None):
        path = _ref_path(ref)
        if not path.is_file() or _file_digest(path, checkpoint) != ref.sha256:
            raise ArtifactRegistrationError("frozen artifact hash changed or file is missing")

    def _prepare_freeze(self, candidate, *, run_id, producer, operation_key=None, run=None,
                        source=None, encoded=None, source_sha256=None, metadata=None, task_id=None,
                        attempt_number=None, session_id=None, await_receipt=False, checkpoint=None):
        """Check one slot and allocate in memory; neither publish nor mutate its Run."""
        _checkpoint(checkpoint)
        if run is not None and operation_key is None:
            raise ArtifactRegistrationError("durable registration requires its logical operation key")
        facts = dict(metadata if metadata is not None else candidate.metadata)
        pending = run.pending_registrations.get(operation_key) if run is not None else None
        if pending is not None:
            ref = pending.ref
            if pending.candidate_path != candidate.path:
                raise ArtifactRegistrationError("registration recovery path changed")
            expected = (candidate.kind, producer, task_id, attempt_number, session_id,
                        candidate.media_type, candidate.summary, candidate.output_name, facts)
            actual = (ref.kind, ref.producer, ref.task_id, ref.attempt_number, ref.session_id,
                      ref.media_type, ref.summary, ref.output_name, ref.metadata)
            if expected != actual:
                raise ArtifactRegistrationError("registration recovery facts changed")
            if encoded is not None and _bytes_digest(encoded, checkpoint) != ref.sha256:
                raise ArtifactRegistrationError("registration slot has different content")
            if run.artifacts.get(ref.id) not in (None, ref):
                raise ArtifactRegistrationError("artifact identity conflicts with Run registration")
            destination = _ref_path(ref)
            # A frozen slot owns its snapshot even when its workspace source was
            # removed or changed. Only an unfinished slot needs the original source.
            if not destination.is_file() and encoded is None:
                if source is None and pending.source_path:
                    source = Path(pending.source_path)
                if source is None or _file_digest(source, checkpoint) != ref.sha256:
                    raise ArtifactRegistrationError("artifact source changed while freezing interrupted registration")
            return _PreparedRegistration(ref, operation_key, source=source, encoded=encoded)
        digest = _bytes_digest(encoded, checkpoint) if encoded is not None else (
            source_sha256 if source_sha256 is not None else _file_digest(source, checkpoint))
        artifact_id = new_artifact_id()
        filename = Path(candidate.path).name if encoded is not None else source.name
        destination = self.root / run_id / artifact_id / filename
        ref = ArtifactRef(id=artifact_id, kind=candidate.kind, producer=producer, run_id=run_id,
                          task_id=task_id, attempt_number=attempt_number, session_id=session_id,
                          uri=destination.as_uri(), sha256=digest, media_type=candidate.media_type,
                          summary=candidate.summary, output_name=candidate.output_name, metadata=facts)
        allocation = PendingRegistration(ref=ref, candidate_path=candidate.path,
            source_path=str(source) if source is not None else None, await_receipt=await_receipt)
        return _PreparedRegistration(ref, operation_key, allocation, source, encoded)

    def _freeze_prepared(self, prepared, *, checkpoint=None):
        """Atomically freeze one already allocated slot without saving its Run."""
        ref = prepared.ref
        destination = _ref_path(ref)
        if destination.is_file():
            self._verify_ref(ref, checkpoint=checkpoint)
            return ref
        _checkpoint(checkpoint)
        destination.parent.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=destination.parent.parent, prefix=f".{ref.id}."))
        try:
            staged_file = staging / destination.name
            if prepared.encoded is not None:
                _write_bytes(prepared.encoded, staged_file, checkpoint=checkpoint)
            else:
                _copy_file(prepared.source, staged_file, checkpoint=checkpoint)
            if _file_digest(staged_file, checkpoint) != ref.sha256:
                raise ArtifactRegistrationError("artifact source changed while freezing")
            _checkpoint(checkpoint)
            # Do not interrupt a rename or a Run save halfway through. The next
            # checkpoint is after this Ref has entered the completed group.
            os.replace(staging, destination.parent)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return ref

    def _publish_prepared(self, prepared, *, run, checkpoint=None):
        """Persist the complete plan, freeze it, then index completed material once.

        Preparation failures publish nothing. A freeze failure or cooperative
        interruption preserves completed Refs and every original pending slot;
        the caller acknowledges only after saving its authoritative receipt.
        """
        allocations = {item.operation_key: item.allocation for item in prepared
                       if item.allocation is not None and run is not None}
        _checkpoint(checkpoint)
        if allocations:
            run.pending_registrations.update(allocations)
            self._save(run)
        completed = []
        failure = None
        try:
            for item in prepared:
                _checkpoint(checkpoint)
                completed.append(self._freeze_prepared(item, checkpoint=checkpoint))
        except BaseException as error:
            failure = error
            raise
        finally:
            if run is not None and completed:
                for ref in completed:
                    run.artifacts[ref.id] = ref
                # Preserve interruption identity if indexing also fails, so the
                # owner cannot mistake cancelled reception for Agent failure.
                try:
                    self._save(run)
                except Exception as save_error:
                    if failure is not None:
                        raise failure from save_error
                    raise
        _checkpoint(checkpoint)
        return completed

    def _freeze(self, candidate, *, run_id, producer, operation_key=None, run=None,
                source=None, encoded=None, source_sha256=None, metadata=None, task_id=None,
                attempt_number=None, session_id=None, await_receipt=False, checkpoint=None):
        run = self._run(run_id, run)
        prepared = self._prepare_freeze(candidate, run_id=run_id, producer=producer,
            operation_key=operation_key, run=run, source=source, encoded=encoded,
            source_sha256=source_sha256, metadata=metadata, task_id=task_id,
            attempt_number=attempt_number, session_id=session_id,
            await_receipt=await_receipt, checkpoint=checkpoint)
        return self._publish_prepared([prepared], run=run, checkpoint=checkpoint)[0]

    def acknowledge_registrations(self, operation_keys, *, run_id, run=None, abort_unfrozen=False):
        """Acknowledge exact slots in one save after their authoritative receipt.

        A failed receiver may explicitly abandon unfinished ordinary outputs only
        after its FAILED receipt is durable. Native Scientific receipt slots and
        material already on disk never qualify for that abandonment.
        """
        run = self._run(run_id, run)
        if run is None:
            return
        keys = list(dict.fromkeys(operation_keys))
        present = []
        for key in keys:
            pending = run.pending_registrations.get(key)
            if pending is None:
                continue
            if run.artifacts.get(pending.ref.id) != pending.ref:
                absent = pending.ref.id not in run.artifacts and not _ref_path(pending.ref).exists()
                if not (abort_unfrozen and absent and not pending.await_receipt):
                    raise ArtifactRegistrationError("cannot acknowledge an uncommitted artifact")
            present.append(key)
        if present:
            for key in present:
                del run.pending_registrations[key]
            self._save(run)

    def acknowledge_registration(self, operation_key, *, run_id, run=None):
        """Commit caller association only after its authoritative receipt is saved."""
        self.acknowledge_registrations([operation_key], run_id=run_id, run=run)

    def _prepare_task(self, candidate, *, grant, producer, run_id, task_id, attempt_number,
                      index, output_dir=None, operation_key=None, run=None, checkpoint=None,
                      reusable_refs=()):
        key = operation_key or f"task:{task_id}:attempt:{attempt_number}:output:{index}"
        pending = run.pending_registrations.get(key) if run is not None else None
        metadata = dict(candidate.metadata)
        metadata.pop("source_path", None)
        metadata.pop("source_root", None)
        source = None
        if candidate.content is None:
            if pending is not None:
                source = Path(pending.source_path) if pending.source_path else None
                metadata.update({name: pending.ref.metadata[name] for name in ("source_path", "source_root")})
            else:
                try:
                    label, workspace, source = resolve_artifact_source(candidate.path, grant=grant, output_dir=output_dir)
                except (ArtifactCandidateError, OSError) as error:
                    raise ArtifactRegistrationError(str(error)) from error
                metadata.update(source_path=source.relative_to(workspace).as_posix(), source_root=label)
        elif candidate.kind == "execution_record" and pending is None and run is not None:
            digest = _bytes_digest(candidate.content.encode("utf-8"), checkpoint)
            expected = (candidate.kind, producer, run_id, task_id, attempt_number, None, digest,
                        candidate.media_type, candidate.output_name, metadata)
            for ref in (*run.artifacts.values(), *reusable_refs):
                if (ref.kind, ref.producer, ref.run_id, ref.task_id, ref.attempt_number, ref.session_id, ref.sha256,
                        ref.media_type, ref.output_name, ref.metadata) != expected:
                    continue
                if ref.id in run.artifacts:
                    self._verify_ref(ref, checkpoint=checkpoint)
                return _PreparedRegistration(ref)
        return self._prepare_freeze(candidate, run_id=run_id, producer=producer, operation_key=key, run=run,
            source=source, encoded=candidate.content.encode("utf-8") if candidate.content is not None else None,
            metadata=metadata, task_id=task_id, attempt_number=attempt_number, checkpoint=checkpoint)

    def register(self, candidate, *, grant, producer, run_id, task_id, attempt_number,
                 index, output_dir=None, operation_key=None, run=None, checkpoint=None):
        run = self._run(run_id, run)
        prepared = self._prepare_task(candidate, grant=grant, producer=producer, run_id=run_id,
            task_id=task_id, attempt_number=attempt_number, index=index, output_dir=output_dir,
            operation_key=operation_key, run=run, checkpoint=checkpoint)
        return self._publish_prepared([prepared], run=run, checkpoint=checkpoint)[0]

    def register_many(self, outputs, *, grant, producer, run_id, task_id=None, attempt_number=None,
                      start_index=1, output_dir=None, operation_keys, session_id=None, run=None,
                      checkpoint=None):
        """Prepare an entire received result before saving allocations or freezing.

        Stable caller keys recover old partially registered Runs without changing
        UUIDs, sources or expected hashes. No new Ref is published on prepare failure.
        """
        run = self._run(run_id, run)
        outputs, keys = list(outputs), list(operation_keys)
        if len(outputs) != len(keys) or len(set(keys)) != len(keys):
            raise ArtifactRegistrationError("registration group requires one unique key per output")
        prepared = []
        ids, names = set(), set()
        for index, (output, key) in enumerate(zip(outputs, keys)):
            _checkpoint(checkpoint)
            if output.output_name is not None:
                if output.output_name in names:
                    raise ArtifactRegistrationError("duplicate logical output name")
                names.add(output.output_name)
            if isinstance(output, ArtifactRef):
                if run is None or run.artifacts.get(output.id) != output or output.run_id != run_id:
                    raise ArtifactRegistrationError("returned Ref is not in the trusted registration record")
                self._verify_ref(output, checkpoint=checkpoint)
                item = _PreparedRegistration(output)
            elif producer == AgentOwner.SCIENTIFIC:
                if session_id is None:
                    raise ArtifactRegistrationError("Scientific output requires its owning session")
                item = self._prepare_scientific(output, run_id=run_id, session_id=session_id,
                    operation_key=key, run=run, checkpoint=checkpoint)
            else:
                item = self._prepare_task(output, grant=grant, producer=producer, run_id=run_id,
                    task_id=task_id, attempt_number=attempt_number, index=start_index + index,
                    output_dir=output_dir, operation_key=key, run=run, checkpoint=checkpoint,
                    reusable_refs=(item.ref for item in prepared))
            if item.ref.id in ids:
                raise ArtifactRegistrationError("duplicate returned artifact reference")
            ids.add(item.ref.id)
            prepared.append(item)
        return self._publish_prepared(prepared, run=run, checkpoint=checkpoint)

    def register_import(self, spec: ArtifactImport, *, run_id, operation_key=None, run=None):
        source = _resolve_import_uri(spec.uri)
        pending_run = self._run(run_id, run)
        pending = pending_run.pending_registrations.get(operation_key) if pending_run is not None else None
        if pending is None and not source.is_file():
            raise ArtifactRegistrationError(f"import uri is not a readable file: {spec.uri}")
        # Bind allocation and every later freeze to the first checked snapshot.
        # Recomputing it inside _freeze could accept a different source version.
        digest = pending.ref.sha256 if pending is not None else _sha256(source)
        if spec.expected_sha256 is not None and spec.expected_sha256 != digest:
            raise ArtifactRegistrationError(f"import sha256 mismatch for {spec.uri}")
        metadata = {**spec.metadata, "source_type": "import"}
        # Import reuse is content and provenance equality, independent of UUID generation.
        if pending is None and pending_run is not None:
            for ref in pending_run.artifacts.values():
                if (ref.kind, ref.sha256, ref.media_type, ref.metadata) == (spec.kind, digest, spec.media_type, metadata):
                    self._verify_ref(ref)
                    return ref
        candidate = ArtifactCandidate(kind=spec.kind, path=source.name, media_type=spec.media_type,
                                      summary=spec.summary, metadata=metadata)
        return self._freeze(candidate, run_id=run_id, producer=AgentOwner.ORCHESTRATOR,
                            operation_key=operation_key, run=pending_run, source=source,
                            source_sha256=digest, metadata=metadata)

    def _prepare_scientific(self, candidate, *, run_id, session_id, content_bytes=None,
                            operation_key=None, run=None, await_receipt=False, checkpoint=None):
        if candidate.kind not in SCIENTIFIC_ARTIFACT_KINDS:
            raise ArtifactRegistrationError(f"unsupported scientific artifact kind: {candidate.kind}")
        if content_bytes is not None and candidate.content is not None:
            raise ArtifactRegistrationError("supply text or binary content, not both")
        encoded = content_bytes if content_bytes is not None else (
            candidate.content if candidate.content is not None else json.dumps(candidate.metadata, sort_keys=True, ensure_ascii=False, indent=2)
        ).encode("utf-8")
        run = self._run(run_id, run)
        task_id = attempt_number = None
        if run is not None and run.workflow is not None:
            for task in run.workflow.tasks:
                for attempt in task.attempts:
                    if attempt.session is not None and attempt.session.id == session_id:
                        task_id, attempt_number = task.id, attempt.number
        return self._prepare_freeze(candidate, run_id=run_id, producer=AgentOwner.SCIENTIFIC,
                            operation_key=operation_key, run=run, encoded=encoded,
                            task_id=task_id, attempt_number=attempt_number,
                            session_id=session_id, await_receipt=await_receipt, checkpoint=checkpoint)

    def register_scientific(self, candidate, *, run_id, session_id, content_bytes=None,
                            operation_key=None, run=None, await_receipt=False, checkpoint=None):
        run = self._run(run_id, run)
        prepared = self._prepare_scientific(candidate, run_id=run_id, session_id=session_id,
            content_bytes=content_bytes, operation_key=operation_key, run=run,
            await_receipt=await_receipt, checkpoint=checkpoint)
        return self._publish_prepared([prepared], run=run, checkpoint=checkpoint)[0]

    def register_system_artifact(self, candidate, *, run_id, source_type, session_id=None,
                                 task_id=None, attempt_number=None, operation_key=None, run=None):
        if candidate.kind not in SYSTEM_ARTIFACT_PROVENANCE:
            raise ArtifactRegistrationError(f"unsupported system artifact kind: {candidate.kind}")
        encoded = (candidate.content if candidate.content is not None else json.dumps(
            candidate.metadata, sort_keys=True, ensure_ascii=False, indent=2)).encode("utf-8")
        return self._freeze(candidate, run_id=run_id, producer=AgentOwner.ORCHESTRATOR,
                            operation_key=operation_key, run=run, encoded=encoded,
                            metadata={**candidate.metadata, "source_type": source_type},
                            session_id=session_id, task_id=task_id, attempt_number=attempt_number)

    def settle_final_report_registrations(self, *, run_id, run=None, active_key=None):
        """Retire allocations only after completion or deferral is durable.

        Frozen superseded reports remain immutable evidence, including a file
        renamed before its Ref was indexed. An unused, unfrozen allocation has
        no evidence to retain. The active candidate keeps its recovery slot.
        """
        run = self._run(run_id, run)
        if run is None:
            return
        keys = [key for key, item in run.pending_registrations.items()
                if key != active_key and item.ref.kind == "final_report"
                and item.ref.producer == AgentOwner.ORCHESTRATOR
                and item.ref.metadata.get("source_type") == "final_report"
                and (item.ref.task_id, item.ref.attempt_number, item.ref.session_id) == (None, None, None)]
        recovered = []
        for key in keys:
            ref = run.pending_registrations[key].ref
            if run.artifacts.get(ref.id) not in (None, ref):
                raise ArtifactRegistrationError("artifact identity conflicts with Run registration")
            if _ref_path(ref).exists() or ref.id in run.artifacts:
                self._verify_ref(ref)
                recovered.append(ref)
        if keys:
            for ref in recovered:
                run.artifacts[ref.id] = ref
            for key in keys:
                del run.pending_registrations[key]
            self._save(run)

    def register_final_report(self, candidate, content, *, run_id, run=None, operation_key="final_report"):
        if (candidate.kind, candidate.media_type, candidate.metadata.get("source_type"), candidate.path) != (
                "final_report", "text/markdown", "final_report", "final_report.md"):
            raise ArtifactRegistrationError("invalid final report candidate")
        run = self._run(run_id, run)
        if run is not None and run.final_report_artifact_id is not None:
            ref = run.artifacts[run.final_report_artifact_id]
            if ref.sha256 != _bytes_digest(content.encode("utf-8")):
                raise ArtifactRegistrationError("final report differs from committed content")
            self._verify_ref(ref)
            return ref
        return self._freeze(candidate, run_id=run_id, producer=AgentOwner.ORCHESTRATOR,
                            operation_key=operation_key, run=run, encoded=content.encode("utf-8"))


class ScientificArtifactRegistration:
    """Run authority port; callers maintain their own explicit authorization set."""

    def __init__(self, registry: ArtifactRegistry, store: RunStore):
        self._registry = registry
        self._store = store
        self._registry.store = store

    def register_scientific(self, candidate, *, run_id, session_id, operation_key, output_position, content_bytes=None):
        run = self._store.load(run_id)
        ref = self._registry.register_scientific(candidate, run_id=run_id, session_id=session_id,
                                                 content_bytes=content_bytes, operation_key=f"{operation_key}:output:{output_position}",
                                                 run=run, await_receipt=True)
        return ref

    def acknowledge_registration(self, *, run_id, operation_key):
        """Consume allocations only after the Session saved their unique receipt."""
        run = self._store.load(run_id)
        keys = [key for key, pending in run.pending_registrations.items()
                if key.startswith(operation_key + ":output:") and pending.await_receipt]
        for key in keys:
            ref = run.pending_registrations[key].ref
            frozen = Path(url2pathname(urlparse(ref.uri).path))
            if ref.id not in run.artifacts and not frozen.exists():
                # No bytes were frozen before interruption. The receipt records
                # that the call is unknown; this unused allocation grants nothing.
                del run.pending_registrations[key]
                self._store.save(run)
                continue
            RegisteredArtifactReader([ref], run_id=run_id).verify(ref.id)
            if run.artifacts.get(ref.id) != ref:
                raise ArtifactRegistrationError("receipt artifact is not in the trusted registration record")
            if ref.id not in run.scientific_artifact_ids:
                run.scientific_artifact_ids.append(ref.id)
            self._registry.acknowledge_registration(key, run_id=run_id, run=run)

    def recover_registration(self, *, run_id, operation_key):
        """Return truthful committed material refs for an interrupted native receipt."""
        run = self._store.load(run_id)
        refs = []
        for key, pending in sorted(run.pending_registrations.items()):
            if not key.startswith(operation_key + ":output:") or not pending.await_receipt:
                continue
            ref = pending.ref
            frozen = Path(url2pathname(urlparse(ref.uri).path))
            if ref.id not in run.artifacts and not frozen.exists():
                continue
            reader = RegisteredArtifactReader([ref], run_id=run_id)
            reader.verify(ref.id)
            if run.artifacts.get(ref.id) not in (None, ref):
                raise ArtifactRegistrationError("recovery artifact conflicts with registry")
            run.artifacts[ref.id] = ref
            refs.append(ref)
        self._store.save(run)
        return refs

    def resolve(self, artifact_id, *, run_id, session_id):
        if not self._store.exists(run_id):
            return None
        run = self._store.load(run_id)
        ref = run.artifacts.get(artifact_id)
        return ref if ref is not None and ref.session_id == session_id and artifact_id in run.scientific_artifact_ids else None

    def list_artifacts(self, *, run_id, session_id):
        run = self._store.load(run_id)
        return [run.artifacts[key] for key in run.scientific_artifact_ids
                if key in run.artifacts and run.artifacts[key].session_id == session_id]
