"""Receiving is bounded work: stop safely and replay saved results, never execution."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from e2e.native_fixtures import tool_turns
from resagent2_contracts import AgentResult, ArtifactCandidate, ModuleStatus, RunStatus, SessionRef
from resagent2_orchestrator import JsonRunStore
from resagent2_runtime import ScriptedLLMClient
from resagent2_runtime.cancellation import current_cancellation
from test_live_lifecycle import build, request, scientific_finish, work


class OutputPort:
    def __init__(self, count=100):
        self.calls = 0
        self.count = count

    def invoke(self, request):
        self.calls += 1
        now = datetime.now(UTC)
        return AgentResult(status="completed", report="Original computation already completed", session=SessionRef(
            id="session_original_outputs", module="executor", state_uri="session://session_original_outputs",
            status="completed", created_at=now, updated_at=now), artifacts=[ArtifactCandidate(
                kind="experiment_result", path=f"result_{i}.txt", media_type="text/plain",
                summary=f"Explicit result {i}", content=str(i)) for i in range(self.count)])


def controller_with_disk(tmp_path, client, child):
    controller = build(tmp_path, client, child)
    store = JsonRunStore(tmp_path / "state")
    controller.scheduler.store = store
    controller.scheduler.artifact_registry.store = store
    return controller


def interrupt_during_freeze(controller, run_id, original, ref, *, text=""):
    value = original(ref)
    controller.control_inbox.submit(run_id, "interrupt", text)
    token = current_cancellation()
    assert token is not None
    # Exercise the actual inbox watcher, rather than faking a paused Run.
    token.wait(3)
    raise AssertionError("the owner did not observe the interrupt")


def test_interrupt_large_result_restarts_only_receiving_with_stable_ids(tmp_path, monkeypatch):
    child = OutputPort()
    client = ScriptedLLMClient(tool_turns([work(), scientific_finish()]))
    controller = controller_with_disk(tmp_path, client, child)
    registry = controller.scheduler.artifact_registry
    original = registry._freeze_prepared
    frozen = []

    def stop_at_third(prepared, *, checkpoint=None):
        frozen.append(prepared.ref.id)
        if prepared.ref.kind == "experiment_result" and len(frozen) == 3:
            return interrupt_during_freeze(controller, "run_receiving_pause",
                lambda item: original(item, checkpoint=checkpoint), prepared)
        return original(prepared, checkpoint=checkpoint)

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_freeze_prepared", stop_at_third)
        paused = controller.create_run("run_receiving_pause", request())
    assert paused.status == RunStatus.PAUSED
    assert paused.manually_paused_at is not None
    assert child.calls == 1 and len(frozen) == 3
    pending = next(iter(paused.pending_module_results.values()))
    assert pending.result.status == ModuleStatus.COMPLETED and pending.received_result is None
    child_task = next(task for task in paused.workflow.tasks if task.id != "task_root")
    assert child_task.status == "running" and child_task.attempts[0].session.status == "completed"
    assert paused.active_child("task_root").status == "executing"
    allocations = {item.ref.id for item in paused.pending_registrations.values()}
    assert len(allocations) == child.count

    # Re-read disk state through a fresh scheduler/controller owner, retaining
    # native root Session storage; no execution result is requested again.
    from resagent2_orchestrator import ResearchController, WorkflowScheduler
    scheduler = WorkflowScheduler(bindings=controller.scheduler.bindings,
        store=JsonRunStore(tmp_path / "state"), artifact_root=registry.root, data_root=tmp_path / "data")
    fresh = ResearchController(scientific_port=controller.scientific_port,
        scheduler=scheduler, compiler=controller.compiler, registry=controller.registry,
        interpreter=controller.interpreter, control_inbox=controller.control_inbox)
    final = fresh.resume_run(paused.run_id)
    assert final.status == RunStatus.COMPLETED, final.terminal_error
    assert child.calls == 1 and len(client.contexts) == 2
    assert allocations <= set(final.artifacts)
    assert final.pending_module_results == {} and final.pending_registrations == {}
    assert all(len(task.attempts) == 1 for task in final.workflow.tasks)


@pytest.mark.parametrize("stage", ["output", "gate", "final_report"])
def test_root_receiving_pause_preserves_unpublished_result(tmp_path, monkeypatch, stage):
    client = ScriptedLLMClient(tool_turns([scientific_finish()]))
    controller = controller_with_disk(tmp_path, client, OutputPort(0))
    registry = controller.scheduler.artifact_registry
    sent = False
    original = registry._freeze_prepared

    def freeze(prepared, *, checkpoint=None):
        nonlocal sent
        target = "scientific_opinion" if stage == "output" else "final_report"
        if not sent and stage != "gate" and prepared.ref.kind == target:
            sent = True
            if stage == "final_report":
                # The publication lock fences external input. Signal the owner
                # token rather than acquiring that lock again in its own thread.
                value = original(prepared, checkpoint=checkpoint)
                current_cancellation().cancel("Interrupted during final report freeze")
                current_cancellation().raise_if_cancelled()
                return value
            return interrupt_during_freeze(controller, "run_root_receiving",
                lambda item: original(item, checkpoint=checkpoint), prepared)
        return original(prepared, checkpoint=checkpoint)

    gate = controller.gate

    class Gate:
        def validate(self, run, result, refs):
            nonlocal sent
            if not sent and stage == "gate":
                sent = True
                controller.control_inbox.submit(run.run_id, "interrupt")
            return gate.validate(run, result, refs)

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_freeze_prepared", freeze)
        controller.gate = Gate()
        paused = controller.create_run("run_root_receiving", request())
    assert sent and paused.status == RunStatus.PAUSED
    assert paused.final_report_artifact_id is None and len(client.contexts) == 1
    assert paused.pending_module_results
    controller.gate = gate
    original_final_id = next((item.ref.id for item in paused.pending_registrations.values()
        if item.ref.kind == "final_report"), None)
    final = controller.resume_run(paused.run_id)
    assert final.status == RunStatus.COMPLETED, final.terminal_error
    if original_final_id is not None:
        assert final.final_report_artifact_id == original_final_id
    assert len(client.contexts) == 1
    assert sum(ref.kind == "final_report" for ref in final.artifacts.values()) == 1
    assert not final.pending_module_results and not final.pending_registrations


def test_new_input_during_root_receiving_pause_reaches_original_input_gate(tmp_path, monkeypatch):
    first, second = scientific_finish(), scientific_finish()
    second["arguments"]["artifacts"][0]["content"]["statement"] = "Reconsidered after new requirement"
    client = ScriptedLLMClient(tool_turns([first, second]))
    controller = controller_with_disk(tmp_path, client, OutputPort(0))
    gate = controller.gate

    class Gate:
        def validate(self, run, result, refs):
            controller.control_inbox.submit(run.run_id, "interrupt", "Reconsider the new requirement")
            return gate.validate(run, result, refs)

    controller.gate = Gate()
    paused = controller.create_run("run_new_input_pause", request())
    assert paused.status == RunStatus.PAUSED and len(client.contexts) == 1
    assert paused.pending_module_results
    controller.gate = gate
    final = controller.resume_run(paused.run_id)
    assert final.status == RunStatus.COMPLETED
    assert len(client.contexts) == 2
    assert "Reconsider the new requirement" in client.contexts[-1].task_context
    assert final.final_opinion.statement == "Reconsidered after new requirement"


def test_reception_deadline_keeps_original_result_without_retry(tmp_path, monkeypatch):
    from resagent2_orchestrator import controller as controller_module, scheduler as scheduler_module

    class Clock:
        value = datetime.now(UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.value

    monkeypatch.setattr(controller_module, "datetime", Clock)
    monkeypatch.setattr(scheduler_module, "datetime", Clock)
    child = OutputPort(20)
    controller = controller_with_disk(tmp_path, ScriptedLLMClient(tool_turns([work()])), child)
    registry = controller.scheduler.artifact_registry
    original = registry._freeze_prepared
    copied = []

    def expire(prepared, *, checkpoint=None):
        ref = original(prepared, checkpoint=checkpoint)
        if ref.kind == "experiment_result":
            copied.append(ref.id)
            Clock.value += timedelta(seconds=120)
        return ref

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_freeze_prepared", expire)
        failed = controller.create_run("run_reception_timeout", request())
    assert failed.status == RunStatus.FAILED
    assert failed.terminal_error.code == "timeout" and not failed.terminal_error.retryable
    assert len(copied) == 1 and child.calls == 1
    pending = next(iter(failed.pending_module_results.values()))
    assert pending.result.status == ModuleStatus.COMPLETED
    assert pending.result.report == "Original computation already completed"
    assert set(copied) <= set(failed.artifacts)
    before = controller.scheduler.store.load(failed.run_id).model_dump()
    assert controller.run_until_stable(failed.run_id).model_dump() == before
    assert child.calls == 1


def test_root_publication_deadline_retains_timeout_not_missing_artifact(tmp_path, monkeypatch):
    from resagent2_runtime.budget import DeadlineExceededError
    client = ScriptedLLMClient(tool_turns([scientific_finish()]))
    controller = controller_with_disk(tmp_path, client, OutputPort(0))
    registry = controller.scheduler.artifact_registry
    original = registry._freeze_prepared

    def expire(prepared, *, checkpoint=None):
        if prepared.ref.kind == "final_report":
            raise DeadlineExceededError("Final report reception expired")
        return original(prepared, checkpoint=checkpoint)

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_freeze_prepared", expire)
        failed = controller.create_run("run_final_timeout", request())
    assert failed.status == RunStatus.FAILED
    assert failed.terminal_error.code == "timeout"
    assert failed.final_report_artifact_id is None and len(client.contexts) == 1
    assert failed.pending_module_results
    assert controller.run_until_stable(failed.run_id).model_dump() == failed.model_dump()
    assert len(client.contexts) == 1


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("retirement_crash", [False, True])
def test_frozen_final_report_is_superseded_after_new_input(tmp_path, monkeypatch, frozen, legacy, retirement_crash):
    from resagent2_orchestrator import ResearchController, WorkflowScheduler
    from resagent2_components.artifacts import RegisteredArtifactReader

    first, second = scientific_finish(), scientific_finish()
    second["arguments"]["artifacts"][0]["content"]["statement"] = "Reconsidered after new requirement"
    client = ScriptedLLMClient(tool_turns([first, second]))
    controller = controller_with_disk(tmp_path, client, OutputPort(0))
    registry = controller.scheduler.artifact_registry
    original = registry._freeze_prepared

    def pause_final(prepared, *, checkpoint=None):
        if prepared.ref.kind == "final_report":
            if frozen:
                original(prepared, checkpoint=checkpoint)
            current_cancellation().cancel("Interrupted before final report association")
            current_cancellation().raise_if_cancelled()
        return original(prepared, checkpoint=checkpoint)

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_freeze_prepared", pause_final)
        paused = controller.create_run("run_revised_final", request())
    assert paused.status == RunStatus.PAUSED and paused.final_report_artifact_id is None
    old = next(item.ref for item in paused.pending_registrations.values()
        if item.ref.kind == "final_report")
    assert old.id not in paused.artifacts
    if legacy:
        # Load an actual pre-fix JSON snapshot with the old fixed slot and no
        # optional field, rather than relying on an in-memory fallback.
        import json
        snapshot = tmp_path / "state" / f"{paused.run_id}.json"
        value = json.loads(snapshot.read_text())
        pending = value["pending_module_results"]["task:task_root:attempt:1"]
        key = pending.pop("final_report_key")
        value["pending_registrations"]["final_report"] = value["pending_registrations"].pop(key)
        snapshot.write_text(json.dumps(value))
    controller.control_inbox.submit(paused.run_id, "input", "Reconsider the new requirement")

    # A fresh owner reloads the Run; only the native Session store is retained.
    scheduler = WorkflowScheduler(bindings=controller.scheduler.bindings,
        store=JsonRunStore(tmp_path / "state"), artifact_root=registry.root, data_root=tmp_path / "data")
    fresh = ResearchController(scientific_port=controller.scientific_port,
        scheduler=scheduler, compiler=controller.compiler, registry=controller.registry,
        interpreter=controller.interpreter, control_inbox=controller.control_inbox)
    if retirement_crash:
        class ProcessCrash(BaseException):
            pass
        original_settle = scheduler.artifact_registry.settle_final_report_registrations
        def stop_before_retirement(**kwargs):
            if kwargs["active_key"] is None:
                raise ProcessCrash()
            return original_settle(**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(scheduler.artifact_registry, "settle_final_report_registrations", stop_before_retirement)
            with pytest.raises(ProcessCrash):
                fresh.resume_run(paused.run_id)
        saved = scheduler.store.load(paused.run_id)
        assert not saved.pending_module_results and len(client.contexts) == 1
        assert any(item.ref.id == old.id for item in saved.pending_registrations.values())
        final = fresh.run_until_stable(paused.run_id)
    else:
        final = fresh.resume_run(paused.run_id)
    assert final.status == RunStatus.COMPLETED, final.terminal_error
    assert len(client.contexts) == 2
    assert "Reconsider the new requirement" in client.contexts[-1].task_context
    assert final.final_opinion.statement == "Reconsidered after new requirement"
    assert final.final_report_artifact_id != old.id
    assert not final.pending_module_results and not final.pending_registrations
    reader = RegisteredArtifactReader(list(final.artifacts.values()), run_id=final.run_id)
    reader.verify(final.final_report_artifact_id)
    if frozen:
        assert final.artifacts[old.id] == old
        reader.verify(old.id)
    else:
        assert old.id not in final.artifacts
    assert fresh.run_until_stable(final.run_id).model_dump() == final.model_dump()
    assert len(client.contexts) == 2


def test_committed_final_report_only_finishes_acknowledgment_after_crash(tmp_path, monkeypatch):
    from resagent2_orchestrator import ResearchController, WorkflowScheduler
    class ProcessCrash(BaseException):
        pass

    client = ScriptedLLMClient(tool_turns([scientific_finish()]))
    controller = controller_with_disk(tmp_path, client, OutputPort(0))
    registry = controller.scheduler.artifact_registry
    original = registry.settle_final_report_registrations
    def stop_after_commit(**kwargs):
        if kwargs["run"].status == RunStatus.COMPLETED:
            raise ProcessCrash()
        return original(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(registry, "settle_final_report_registrations", stop_after_commit)
        with pytest.raises(ProcessCrash):
            controller.create_run("run_committed_final", request())
    saved = controller.scheduler.store.load("run_committed_final")
    report_id, opinion = saved.final_report_artifact_id, saved.final_opinion
    assert saved.status == RunStatus.COMPLETED and saved.pending_registrations

    scheduler = WorkflowScheduler(bindings=controller.scheduler.bindings,
        store=JsonRunStore(tmp_path / "state"), artifact_root=registry.root, data_root=tmp_path / "data")
    fresh = ResearchController(scientific_port=controller.scientific_port,
        scheduler=scheduler, compiler=controller.compiler, registry=controller.registry,
        interpreter=controller.interpreter, control_inbox=controller.control_inbox)
    final = fresh.run_until_stable(saved.run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.final_report_artifact_id == report_id and final.final_opinion == opinion
    assert final.artifacts == saved.artifacts and len(client.contexts) == 1
    assert not final.pending_module_results and not final.pending_registrations
    assert fresh.run_until_stable(final.run_id).model_dump() == final.model_dump()
