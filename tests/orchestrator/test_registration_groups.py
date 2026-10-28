"""Whole-result allocation, bounded finalization and original-snapshot recovery."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from sciagent_contracts import (
    AgentOwner, AgentResult, ArtifactCandidate, ResearchRequest, RunBudget,
    RunPermissions, WorkspaceGrant,
)
from sciagent_orchestrator import ArtifactRegistrationError, ArtifactRegistry, JsonRunStore, ResearchRun
from sciagent_orchestrator import artifacts as implementation
from sciagent_orchestrator.handoffs import receive_artifacts
from sciagent_runtime.budget import DeadlineExceededError, execution_budget
from sciagent_runtime.cancellation import CancellationToken, OperationCancelled, bind_cancellation


RUN_ID = "run_registration_group"


def setup_group(tmp_path):
    now = datetime.now(UTC)
    store = JsonRunStore(tmp_path / "state")
    run = ResearchRun(run_id=RUN_ID, request=ResearchRequest(
        goal="Receive exactly the original outputs", permissions=RunPermissions(),
        budget=RunBudget(max_llm_calls=1, timeout_seconds=60)),
        status="running", created_at=now, updated_at=now)
    store.save(run)
    registry = ArtifactRegistry(tmp_path / "artifacts", store)
    request = SimpleNamespace(task_id="task_result", attempt_number=1,
        agent=AgentOwner.EXECUTOR, input_artifacts=[], output_dir=None,
        workspace=WorkspaceGrant(root=str(tmp_path),
            access={"read_paths": ["."], "write_paths": ["."]}, source="local"))
    return store, run, registry, request


def candidate(index, *, inline=True):
    return ArtifactCandidate(kind="experiment_result", path=f"result_{index}.txt",
        media_type="text/plain", summary=f"Result {index}", output_name=f"output_{index}",
        content=str(index) if inline else None)


def result(outputs):
    return AgentResult(status="completed", report="Original result", artifacts=outputs)


def frozen(ref):
    return Path(url2pathname(urlparse(ref.uri).path))


def key(index):
    return f"task:task_result:attempt:1:output:{index}"


def test_two_hundred_outputs_save_one_complete_allocation_and_one_ref_group(tmp_path, monkeypatch):
    store, run, registry, request = setup_group(tmp_path)
    snapshots = []
    save = store.save
    def observe(current):
        snapshots.append(current.model_copy(deep=True))
        save(current)
    monkeypatch.setattr(store, "save", observe)
    outputs = [candidate(index) for index in range(200)]
    refs = receive_artifacts(registry, run, request, result(outputs))
    assert len(snapshots) == 2
    allocation, published = snapshots
    assert allocation.artifacts == {} and len(allocation.pending_registrations) == 200
    assert published.artifacts == {ref.id: ref for ref in refs}
    for index, ref in enumerate(refs):
        assert allocation.pending_registrations[key(index + 1)].ref == ref
        assert allocation.pending_registrations[key(index + 1)].source_path is None
        assert ref.sha256 == hashlib.sha256(str(index).encode()).hexdigest()
        assert frozen(ref).read_text() == str(index)
    assert published.pending_registrations == allocation.pending_registrations
    snapshots.clear()
    registry.acknowledge_registrations([key(index + 1) for index in range(200)], run_id=RUN_ID, run=run)
    assert len(snapshots) == 1 and snapshots[0].pending_registrations == {}


@pytest.mark.parametrize("failure", ["missing_file", "duplicate_name", "foreign_ref"])
def test_prepare_failure_publishes_no_allocation_or_new_ref(tmp_path, failure):
    store, run, registry, request = setup_group(tmp_path)
    outputs = [candidate(1)]
    if failure == "missing_file":
        outputs.append(candidate(2, inline=False))
    elif failure == "duplicate_name":
        outputs.append(candidate(2).model_copy(update={"output_name": "output_1"}))
    else:
        foreign = registry.register(candidate(9), grant=request.workspace,
            producer=request.agent, run_id=RUN_ID, task_id="task_foreign",
            attempt_number=1, index=1, run=run)
        outputs.append(foreign)
    before = run.model_copy(deep=True)
    with pytest.raises(ArtifactRegistrationError):
        receive_artifacts(registry, run, request, result(outputs))
    assert run == before and store.load(RUN_ID) == before
    assert {path for path in registry.root.rglob("*.txt")} == {frozen(ref) for ref in before.artifacts.values()}


def test_freeze_failure_retains_all_slots_and_publishes_only_frozen_refs(tmp_path, monkeypatch):
    store, run, registry, request = setup_group(tmp_path)
    outputs = [candidate(index, inline=False) for index in range(3)]
    for index in range(3):
        (tmp_path / f"result_{index}.txt").write_text(str(index))
    copy = implementation._copy_file
    with monkeypatch.context() as patch:
        def fail_second(source, destination, **kwargs):
            if source.name == "result_1.txt":
                raise OSError("second file cannot be copied")
            return copy(source, destination, **kwargs)
        patch.setattr(implementation, "_copy_file", fail_second)
        with pytest.raises(OSError, match="second file"):
            receive_artifacts(registry, run, request, result(outputs))
    interrupted = store.load(RUN_ID)
    slots = interrupted.pending_registrations.copy()
    assert len(slots) == 3
    first = slots[key(1)].ref
    assert interrupted.artifacts == {first.id: first} and run == interrupted
    assert frozen(first).read_text() == "0"
    assert not frozen(slots[key(2)].ref).exists()
    # Recovery must not revisit a completed source, and must reject changed bytes
    # in an unfinished source without replacing any saved identity or hash.
    (tmp_path / "result_0.txt").unlink()
    (tmp_path / "result_1.txt").write_text("changed")
    reopened = ArtifactRegistry(registry.root, store)
    with pytest.raises(ArtifactRegistrationError, match="source changed"):
        receive_artifacts(reopened, interrupted, request, result(outputs))
    assert store.load(RUN_ID).pending_registrations == slots
    assert store.load(RUN_ID).artifacts == {first.id: first}
    (tmp_path / "result_1.txt").write_text("1")
    refs = receive_artifacts(reopened, interrupted, request, result(outputs))
    assert refs == [slots[key(index + 1)].ref for index in range(3)]
    assert store.load(RUN_ID).artifacts == {ref.id: ref for ref in refs}


@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
def test_copy_checkpoint_stops_within_chunks_and_preserves_frozen_prefix(tmp_path, monkeypatch, interruption):
    store, run, registry, request = setup_group(tmp_path)
    outputs = [candidate(0, inline=False), candidate(1, inline=False)]
    (tmp_path / "result_0.txt").write_text("0")
    (tmp_path / "result_1.txt").write_bytes(b"x" * (2 * 1024 * 1024))
    copy = implementation._copy_file
    copying_second = False
    chunks = 0
    observed_sizes = []
    def observing_copy(source, destination, **kwargs):
        nonlocal copying_second
        copying_second = source.name == "result_1.txt"
        return copy(source, destination, **kwargs)
    def checkpoint():
        nonlocal chunks
        if copying_second:
            chunks += 1
            observed_sizes.extend(path.stat().st_size for path in registry.root.rglob("result_1.txt"))
            if chunks == 3:
                if interruption == "cancel":
                    raise OperationCancelled("stop during copy")
                raise DeadlineExceededError("deadline during copy")
    monkeypatch.setattr(implementation, "_copy_file", observing_copy)
    expected = OperationCancelled if interruption == "cancel" else DeadlineExceededError
    with pytest.raises(expected):
        receive_artifacts(registry, run, request, result(outputs), checkpoint=checkpoint)
    interrupted = store.load(RUN_ID)
    refs = [interrupted.pending_registrations[key(index + 1)].ref for index in range(2)]
    assert interrupted.artifacts == {refs[0].id: refs[0]}
    assert max(observed_sizes) <= 2 * 64 * 1024
    assert not frozen(refs[1]).exists()
    assert not list((registry.root / RUN_ID).glob(".*"))
    assert set(interrupted.pending_registrations) == {key(1), key(2)}
    (tmp_path / "result_0.txt").unlink()
    recovered = receive_artifacts(ArtifactRegistry(registry.root, store), interrupted, request, result(outputs))
    assert recovered == refs


@pytest.mark.parametrize("interruption", ["token", "deadline"])
def test_source_hash_checks_bound_context_before_allocation(tmp_path, interruption):
    store, run, registry, request = setup_group(tmp_path)
    (tmp_path / "result_0.txt").write_bytes(b"x" * (2 * 1024 * 1024))
    if interruption == "token":
        token = CancellationToken()
        token.cancel()
        with bind_cancellation(token), pytest.raises(OperationCancelled):
            receive_artifacts(registry, run, request, result([candidate(0, inline=False)]))
    else:
        with execution_budget(max_llm_calls=0, timeout_seconds=0), pytest.raises(DeadlineExceededError):
            receive_artifacts(registry, run, request, result([candidate(0, inline=False)]))
    assert run.pending_registrations == {} and run.artifacts == {}
    assert store.load(RUN_ID) == run


def test_zero_remaining_model_calls_still_allows_artifact_delivery(tmp_path):
    store, run, registry, request = setup_group(tmp_path)
    with execution_budget(max_llm_calls=0, timeout_seconds=60):
        refs = receive_artifacts(registry, run, request, result([candidate(0)]))
    assert store.load(RUN_ID).artifacts == {refs[0].id: refs[0]}


def test_legacy_partial_registration_and_new_slots_share_one_plan(tmp_path, monkeypatch):
    store, run, registry, request = setup_group(tmp_path)
    outputs = [candidate(index) for index in range(3)]
    original = registry.register(outputs[0], grant=request.workspace, producer=request.agent,
        run_id=RUN_ID, task_id=request.task_id, attempt_number=1, index=1, run=run)
    saves = []
    save = store.save
    def observe(current):
        saves.append(current.model_copy(deep=True))
        save(current)
    monkeypatch.setattr(store, "save", observe)
    refs = receive_artifacts(registry, run, request, result(outputs))
    assert refs[0] == original and len(saves) == 2
    assert len(saves[0].pending_registrations) == 3 and saves[0].artifacts == {original.id: original}
    assert len(saves[1].artifacts) == 3


def test_failed_receipt_can_abandon_only_exact_unfrozen_ordinary_slots(tmp_path, monkeypatch):
    store, run, registry, request = setup_group(tmp_path)
    def unavailable(*args, **kwargs):
        raise OSError("cannot start freezing")
    with monkeypatch.context() as patch:
        patch.setattr(implementation.tempfile, "mkdtemp", unavailable)
        with pytest.raises(OSError):
            receive_artifacts(registry, run, request, result([candidate(0), candidate(1)]))
        with pytest.raises(OSError):
            registry.register_scientific(ArtifactCandidate(kind="literature_paper", path="paper.txt",
                media_type="text/plain", summary="Native tool paper", content="paper"),
                run_id=RUN_ID, session_id="session_native", operation_key="native:output:1",
                run=run, await_receipt=True)
    with pytest.raises(ArtifactRegistrationError, match="uncommitted"):
        registry.acknowledge_registrations([key(1), key(2)], run_id=RUN_ID, run=run)
    before = run.model_copy(deep=True)
    with pytest.raises(ArtifactRegistrationError, match="uncommitted"):
        registry.acknowledge_registrations([key(1), "native:output:1"], run_id=RUN_ID, run=run,
            abort_unfrozen=True)
    assert run == before
    registry.acknowledge_registrations([key(1), key(2)], run_id=RUN_ID, run=run, abort_unfrozen=True)
    assert list(store.load(RUN_ID).pending_registrations) == ["native:output:1"]


@pytest.mark.parametrize("phase", ["source", "frozen"])
@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
def test_hash_and_verify_checkpoint_interrupt_inside_large_files(tmp_path, monkeypatch, phase, interruption):
    store, run, registry, request = setup_group(tmp_path)
    source = tmp_path / "result_0.txt"
    source.write_bytes(b"x" * (2 * 1024 * 1024))
    output = candidate(0, inline=False)
    if phase == "frozen":
        output = registry.register(output, grant=request.workspace, producer=request.agent,
            run_id=RUN_ID, task_id=request.task_id, attempt_number=1, index=1, run=run)
    target = source if phase == "source" else frozen(output)
    active = None
    checks = 0
    digest = implementation._sha256
    def observing_hash(path, **kwargs):
        nonlocal active
        active = path
        try:
            return digest(path, **kwargs)
        finally:
            active = None
    def checkpoint():
        nonlocal checks
        if active == target:
            checks += 1
            if checks == 3:
                if interruption == "cancel":
                    raise OperationCancelled("stop hashing")
                raise DeadlineExceededError("deadline while hashing")
    monkeypatch.setattr(implementation, "_sha256", observing_hash)
    before = run.model_copy(deep=True)
    expected = OperationCancelled if interruption == "cancel" else DeadlineExceededError
    with pytest.raises(expected):
        receive_artifacts(registry, run, request, result([output]), checkpoint=checkpoint)
    assert checks == 3 and run == before and store.load(RUN_ID) == before


@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
def test_index_save_failure_cannot_convert_interruption_into_agent_failure(tmp_path, monkeypatch, interruption):
    store, run, registry, request = setup_group(tmp_path)
    outputs = [candidate(0), candidate(1)]
    freeze = registry._freeze_prepared
    save = store.save
    error = (OperationCancelled("stop receiving") if interruption == "cancel"
             else DeadlineExceededError("deadline receiving"))
    def stop_second(prepared, **kwargs):
        if prepared.ref.output_name == "output_1":
            raise error
        return freeze(prepared, **kwargs)
    def fail_index(current):
        if current.artifacts:
            raise OSError("cannot persist the frozen prefix")
        return save(current)
    with monkeypatch.context() as patch:
        patch.setattr(registry, "_freeze_prepared", stop_second)
        patch.setattr(store, "save", fail_index)
        with pytest.raises(type(error)) as caught:
            receive_artifacts(registry, run, request, result(outputs))
    assert caught.value is error and isinstance(error.__cause__, OSError)
    interrupted = store.load(RUN_ID)
    assert len(interrupted.pending_registrations) == 2 and interrupted.artifacts == {}
    original_refs = [interrupted.pending_registrations[key(index + 1)].ref for index in range(2)]
    assert frozen(original_refs[0]).is_file()
    recovered = receive_artifacts(ArtifactRegistry(registry.root, store), interrupted, request, result(outputs))
    assert recovered == original_refs


def test_group_reuses_same_attempt_execution_record_identity_before_publication(tmp_path):
    store, run, registry, request = setup_group(tmp_path)
    record = ArtifactCandidate(kind="execution_record", path="record.json", media_type="application/json",
        summary="Exact stopped-command evidence", content='{"results": []}')
    renamed = record.model_copy(update={"path": "same_record.json"})
    with pytest.raises(ArtifactRegistrationError, match="duplicate returned artifact reference"):
        receive_artifacts(registry, run, request, result([record, renamed]))
    assert run.artifacts == {} and run.pending_registrations == {} and store.load(RUN_ID) == run
    # Existing frozen records still reuse equal bytes from the producing Attempt.
    original = registry.register(record, grant=request.workspace, producer=request.agent,
        run_id=RUN_ID, task_id=request.task_id, attempt_number=1, index=1, run=run)
    recovered = registry.register(renamed, grant=request.workspace, producer=request.agent,
        run_id=RUN_ID, task_id=request.task_id, attempt_number=1, index=2, run=run)
    assert recovered == original


@pytest.mark.parametrize("boundary", ["rename", "index_save"])
def test_interruption_during_last_atomic_action_is_observed_after_group_save(tmp_path, monkeypatch, boundary):
    store, run, registry, request = setup_group(tmp_path)
    token = CancellationToken()
    clock = [0.0]
    def interrupt():
        if boundary == "rename":
            token.cancel("stop after final rename")
        else:
            clock[0] = 2.0
    replace, save = implementation.os.replace, store.save
    if boundary == "rename":
        def after_rename(source, destination):
            replace(source, destination)
            if Path(destination).name.startswith("artifact_"):
                interrupt()
        monkeypatch.setattr(implementation.os, "replace", after_rename)
    else:
        def after_index(current):
            save(current)
            if current.artifacts:
                interrupt()
        monkeypatch.setattr(store, "save", after_index)
    expected = OperationCancelled if boundary == "rename" else DeadlineExceededError
    with bind_cancellation(token), execution_budget(max_llm_calls=0, timeout_seconds=1,
            clock=lambda: clock[0]), pytest.raises(expected):
        receive_artifacts(registry, run, request, result([candidate(0)]))
    interrupted = store.load(RUN_ID)
    ref = interrupted.pending_registrations[key(1)].ref
    assert interrupted.artifacts == {ref.id: ref} and frozen(ref).read_text() == "0"
