"""Import allocation, expected hashes, and recovery describe one frozen snapshot."""

import hashlib
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from resagent2_contracts import ArtifactImport, ResearchRequest, RunBudget, RunPermissions
from resagent2_orchestrator import (
    ArtifactRegistrationError, ArtifactRegistry, JsonRunStore, ResearchRun,
)
from resagent2_orchestrator import artifacts as implementation


RUN_ID = "run_import_hash"
OPERATION_KEY = "initialization:import:0"
ORIGINAL = b"version A"
REPLACEMENT = b"version B"
ORIGINAL_HASH = hashlib.sha256(ORIGINAL).hexdigest()


def setup_import(tmp_path, *, durable=True, expected=True):
    source = tmp_path / "source.txt"
    source.write_bytes(ORIGINAL)
    store = None
    if durable:
        now = datetime.now(UTC)
        store = JsonRunStore(tmp_path / "state")
        store.save(ResearchRun(
            run_id=RUN_ID, request=ResearchRequest(
                goal="Import original input", permissions=RunPermissions(),
                budget=RunBudget(max_llm_calls=10, timeout_seconds=60),
            ), status="running", created_at=now, updated_at=now,
        ))
    registry = ArtifactRegistry(tmp_path / "artifacts", store)
    spec = ArtifactImport(
        uri=str(source), kind="data", media_type="text/plain", summary="Original input",
        expected_sha256=ORIGINAL_HASH if expected else None,
    )
    return source, store, registry, spec


def register(registry, spec):
    return registry.register_import(spec, run_id=RUN_ID, operation_key=OPERATION_KEY)


def frozen_path(ref):
    return Path(url2pathname(urlparse(ref.uri).path))


def assert_no_frozen_files(registry):
    run_root = registry.root / RUN_ID
    assert not run_root.exists() or not list(run_root.iterdir())


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("expected", [False, True])
def test_source_change_after_first_hash_cannot_replace_import_snapshot(tmp_path, monkeypatch, durable, expected):
    source, store, registry, spec = setup_import(tmp_path, durable=durable, expected=expected)
    original_hash = implementation._sha256
    source_hashes = []

    def change_after_hash(path):
        digest = original_hash(path)
        if path == source:
            source_hashes.append(digest)
            source.write_bytes(REPLACEMENT)
        return digest

    monkeypatch.setattr(implementation, "_sha256", change_after_hash)
    with pytest.raises(ArtifactRegistrationError, match="changed while freezing"):
        register(registry, spec)

    assert source_hashes == [ORIGINAL_HASH]
    assert_no_frozen_files(registry)
    if store is not None:
        run = store.load(RUN_ID)
        assert run.pending_registrations[OPERATION_KEY].ref.sha256 == ORIGINAL_HASH
        assert run.artifacts == {}


@pytest.mark.parametrize("expected", [False, True])
def test_interrupted_copy_keeps_first_digest_and_identity_across_retries(tmp_path, monkeypatch, expected):
    source, store, registry, spec = setup_import(tmp_path, expected=expected)

    with monkeypatch.context() as patch:
        def fail_copy(*args, **kwargs):
            raise OSError("copy interrupted")
        patch.setattr(implementation, "_copy_file", fail_copy)
        with pytest.raises(OSError, match="copy interrupted"):
            register(registry, spec)

    allocated = store.load(RUN_ID).pending_registrations[OPERATION_KEY]
    assert allocated.ref.sha256 == ORIGINAL_HASH
    assert_no_frozen_files(registry)
    source.write_bytes(REPLACEMENT)
    reopened = ArtifactRegistry(registry.root, JsonRunStore(store.root))
    with pytest.raises(ArtifactRegistrationError, match="changed while freezing"):
        register(reopened, spec)
    run = store.load(RUN_ID)
    assert run.pending_registrations[OPERATION_KEY] == allocated
    assert run.artifacts == {}
    assert_no_frozen_files(registry)

    # Even an explicitly revised expected hash cannot repurpose this operation.
    changed_spec = spec.model_copy(update={"expected_sha256": hashlib.sha256(REPLACEMENT).hexdigest()})
    with pytest.raises(ArtifactRegistrationError, match="sha256 mismatch"):
        register(reopened, changed_spec)
    assert store.load(RUN_ID).pending_registrations[OPERATION_KEY] == allocated

    source.write_bytes(ORIGINAL)
    recovered = register(reopened, spec)
    assert recovered == allocated.ref
    assert frozen_path(recovered).read_bytes() == ORIGINAL
    assert store.load(RUN_ID).artifacts == {recovered.id: recovered}


@pytest.mark.parametrize("expected", [False, True])
def test_staged_bytes_are_checked_even_if_source_is_restored_after_copy(tmp_path, monkeypatch, expected):
    source, store, registry, spec = setup_import(tmp_path, expected=expected)
    original_copy = implementation._copy_file

    def copy_changed_version(original, destination, **kwargs):
        source.write_bytes(REPLACEMENT)
        try:
            return original_copy(original, destination, **kwargs)
        finally:
            source.write_bytes(ORIGINAL)

    monkeypatch.setattr(implementation, "_copy_file", copy_changed_version)
    with pytest.raises(ArtifactRegistrationError, match="changed while freezing"):
        register(registry, spec)
    assert source.read_bytes() == ORIGINAL
    assert store.load(RUN_ID).artifacts == {}
    assert store.load(RUN_ID).pending_registrations[OPERATION_KEY].ref.sha256 == ORIGINAL_HASH
    assert_no_frozen_files(registry)


@pytest.mark.parametrize("source_state", ["changed", "deleted"])
def test_frozen_pending_import_recovers_original_without_source_read(tmp_path, monkeypatch, source_state):
    source, store, registry, spec = setup_import(tmp_path)
    original_save = store.save

    with monkeypatch.context() as patch:
        def fail_before_index_save(run):
            if run.artifacts:
                raise OSError("index save interrupted")
            original_save(run)
        patch.setattr(store, "save", fail_before_index_save)
        with pytest.raises(OSError, match="index save interrupted"):
            register(registry, spec)

    allocated = store.load(RUN_ID).pending_registrations[OPERATION_KEY]
    assert store.load(RUN_ID).artifacts == {}
    assert frozen_path(allocated.ref).read_bytes() == ORIGINAL
    source.unlink() if source_state == "deleted" else source.write_bytes(REPLACEMENT)
    original_hash = implementation._sha256

    def frozen_hash_only(path):
        assert path != source, "pending recovery must use its saved snapshot"
        return original_hash(path)

    monkeypatch.setattr(implementation, "_sha256", frozen_hash_only)
    recovered = register(ArtifactRegistry(registry.root, JsonRunStore(store.root)), spec)
    assert recovered == allocated.ref
    assert store.load(RUN_ID).artifacts == {recovered.id: recovered}


def test_existing_import_reuse_returns_checked_frozen_version_if_source_changes(tmp_path, monkeypatch):
    source, store, registry, spec = setup_import(tmp_path)
    first = register(registry, spec)
    registry.acknowledge_registration(OPERATION_KEY, run_id=RUN_ID)
    original_hash = implementation._sha256

    def change_after_hash(path):
        digest = original_hash(path)
        if path == source:
            source.write_bytes(REPLACEMENT)
        return digest

    def unexpected_copy(*args, **kwargs):
        pytest.fail("matching frozen import should be reused")

    monkeypatch.setattr(implementation, "_sha256", change_after_hash)
    monkeypatch.setattr(implementation, "_copy_file", unexpected_copy)
    reused = registry.register_import(spec, run_id=RUN_ID, operation_key="initialization:import:1")
    assert reused == first
    assert frozen_path(reused).read_bytes() == ORIGINAL
    assert len(store.load(RUN_ID).artifacts) == 1


def test_import_hashes_source_once_with_bounded_memory_for_large_files(tmp_path, monkeypatch):
    source, _, registry, spec = setup_import(tmp_path, durable=False, expected=False)
    original_hash = implementation._sha256
    hashes = []

    def count_hash(path):
        hashes.append(path)
        return original_hash(path)

    monkeypatch.setattr(implementation, "_sha256", count_hash)
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("whole-file allocation"))
    peaks = []
    for megabytes in (1, 16):
        with source.open("wb") as handle:
            for _ in range(megabytes * 16):
                handle.write(b"x" * 65536)
        hashes.clear()
        tracemalloc.start()
        try:
            ref = register(registry, spec)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert hashes.count(source) == 1
        assert len(hashes) == 2  # initial source and complete staged copy
        assert original_hash(frozen_path(ref)) == ref.sha256
        peaks.append(peak)
    assert max(peaks) < 2 * 1024 * 1024
    assert peaks[1] <= peaks[0] + 512 * 1024
