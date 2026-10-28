"""Exercise durable allocation and returned-result recovery at actual write boundaries."""
from resagent2_executor import default_execution_registry
from e2e.scheduler_fixtures import executor_bindings

from scheduler_fixtures import accept_proposal, add_root_owner, execution_tasks
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from resagent2_contracts import (
    AgentOwner, AgentResult, ArtifactCandidate, ArtifactImport, ResearchRequest,
    RunBudget, RunPermissions, TaskProposal, WorkflowProposal, WorkspaceGrant,
    WorkflowAgentRegistry,
    WorkRequest, WorkRequestDraft,
)
from resagent2_orchestrator import (
    ArtifactRegistry, JsonRunStore, ModuleBinding, ResearchRun, WorkflowScheduler,
    ScientificArtifactRegistration,
    ResearchController, DeterministicWorkInterpreter,
)
from resagent2_orchestrator import artifacts as artifacts_module


class ProcessCrash(BaseException):
    pass


def setup_run(tmp_path):
    now = datetime.now(UTC)
    store = JsonRunStore(tmp_path / "state")
    run = ResearchRun(
        run_id="run_recovery", request=ResearchRequest(
            goal="Recover original material", permissions=RunPermissions(),
            budget=RunBudget(max_llm_calls=10, timeout_seconds=60),
        ), status="running", created_at=now, updated_at=now,
    )
    run.execution_definitions = default_execution_registry().execution_definitions
    run.execution_presets = default_execution_registry().presets
    store.save(run)
    return store, ArtifactRegistry(tmp_path / "artifacts", store)


@pytest.mark.parametrize("entry", ["task", "import", "scientific", "system", "final"])
@pytest.mark.parametrize("boundary", ["allocated", "frozen", "registered"])
def test_interrupted_registration_reuses_preallocated_identity(tmp_path, monkeypatch, entry, boundary):
    store, registry = setup_run(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("Exact original bytes.\n", encoding="utf-8")

    def register(current):
        if entry == "task":
            return current.register(
                ArtifactCandidate(kind="experiment_result", path=source.name,
                                  media_type="text/plain", summary="Original result"),
                grant=WorkspaceGrant(root=str(tmp_path), access={"read_paths": ["."], "write_paths": ["."]}, source="local"),
                producer=AgentOwner.EXECUTOR, run_id="run_recovery", task_id="task_result",
                attempt_number=1, index=1, operation_key="original",
            )
        if entry == "import":
            return current.register_import(ArtifactImport(uri=str(source), kind="literature_paper",
                media_type="text/plain", summary="Original paper"), run_id="run_recovery", operation_key="original")
        if entry == "scientific":
            return current.register_scientific(ArtifactCandidate(kind="literature_paper", path="paper.md",
                media_type="text/plain", summary="Original paper", content="Exact original bytes.\n"),
                run_id="run_recovery", session_id="session_scientific", operation_key="original")
        if entry == "system":
            return current.register_system_artifact(ArtifactCandidate(kind="conclusion_requirements", path="requirements.json",
                media_type="application/json", summary="Original requirements", content="Exact original bytes.\n"),
                run_id="run_recovery", source_type="conclusion_requirement", operation_key="original")
        return current.register_final_report(ArtifactCandidate(kind="final_report", path="final_report.md",
            media_type="text/markdown", summary="Final report", metadata={"source_type": "final_report"}),
            "Exact original bytes.\n", run_id="run_recovery")

    with monkeypatch.context() as patch:
        if boundary == "allocated":
            patch.setattr(artifacts_module.tempfile, "mkdtemp", lambda **kwargs: (_ for _ in ()).throw(ProcessCrash()))
        elif boundary == "frozen":
            original = artifacts_module.os.replace
            def freeze_then_crash(source, destination):
                original(source, destination)
                if Path(destination).name.startswith("artifact_"):
                    raise ProcessCrash()
            patch.setattr(artifacts_module.os, "replace", freeze_then_crash)
        else:
            original = store.save
            def register_then_crash(run):
                original(run)
                if run.artifacts:
                    raise ProcessCrash()
            patch.setattr(store, "save", register_then_crash)
        with pytest.raises(ProcessCrash):
            register(registry)

    pending = next(iter(store.load("run_recovery").pending_registrations.values()))
    recovered = register(ArtifactRegistry(registry.root, store))
    assert recovered == pending.ref
    assert store.load("run_recovery").artifacts == {recovered.id: recovered}
    assert Path(url2pathname(urlparse(recovered.uri).path)).read_bytes() == b"Exact original bytes.\n"


@pytest.mark.parametrize("deleted", [False, True])
def test_registered_file_recovery_uses_frozen_snapshot_when_source_changes(tmp_path, deleted):
    store, registry = setup_run(tmp_path)
    source = tmp_path / "result.txt"
    source.write_text("Original", encoding="utf-8")
    candidate = ArtifactCandidate(kind="experiment_result", path=source.name,
                                  media_type="text/plain", summary="Original result")
    kwargs = dict(grant=WorkspaceGrant(root=str(tmp_path), access={"read_paths": ["."], "write_paths": ["."]}, source="local"),
                  producer=AgentOwner.EXECUTOR, run_id="run_recovery", task_id="task_result",
                  attempt_number=1, index=1, operation_key="original")
    original = registry.register(candidate, **kwargs)
    source.unlink() if deleted else source.write_text("Changed", encoding="utf-8")
    recovered = ArtifactRegistry(registry.root, store).register(candidate, **kwargs)
    assert recovered == original
    assert Path(url2pathname(urlparse(recovered.uri).path)).read_text() == "Original"


@pytest.mark.parametrize("boundary", ["before_report", "after_report_binding", "after_group_ack"])
def test_returned_result_recovery_does_not_call_agent_or_allocate_outputs_again(tmp_path, monkeypatch, boundary):
    store, registry = setup_run(tmp_path)
    class Port:
        calls = 0
        def invoke(self, request):
            self.calls += 1
            return AgentResult(status="completed", report="Exact original report.  \n", artifacts=[
                ArtifactCandidate(kind="experiment_result", path=f"result_{index}.txt", media_type="text/plain",
                                  summary=f"Result {index}", content=str(index), output_name=f"result_{index}")
                for index in range(2)
            ])
    port = Port()
    scheduler = WorkflowScheduler(bindings=executor_bindings({"experiment": ModuleBinding(owner=AgentOwner.EXECUTOR, port=port)}),
                                  store=store, artifact_root=registry.root, data_root=tmp_path / "data")
    accept_proposal(scheduler, "run_recovery", WorkflowProposal(work_request_id="work_original", tasks=[
        TaskProposal(execution_ref="experiment@1", id="task_result", work_request_id="work_original", workflow_agent_kind="executor",
                     instruction="Return both results"),
    ]))
    with monkeypatch.context() as patch:
        if boundary == "before_report":
            patch.setattr(scheduler, "_persist_attempt_report", lambda *args, **kwargs: (_ for _ in ()).throw(ProcessCrash()))
        elif boundary == "after_report_binding":
            original = scheduler._save
            def save_then_crash(run):
                original(run)
                if run.workflow.tasks[0].attempts[0].report_artifact_id:
                    raise ProcessCrash()
            patch.setattr(scheduler, "_save", save_then_crash)
        else:
            original = scheduler.artifact_registry.acknowledge_registrations
            def acknowledge_then_crash(keys, **kwargs):
                original(keys, **kwargs)
                if any(":output:" in key for key in keys):
                    raise ProcessCrash()
            patch.setattr(scheduler.artifact_registry, "acknowledge_registrations", acknowledge_then_crash)
        with pytest.raises(ProcessCrash):
            scheduler.execute_task("run_recovery", "task_result")
    interrupted = store.load("run_recovery")
    outputs = interrupted.workflow.tasks[0].attempts[0].artifact_ids.copy()
    assert len(outputs) == 2
    scheduler._recover_interrupted_attempts_in_place(interrupted)
    recovered = store.load("run_recovery")
    attempt = recovered.workflow.tasks[0].attempts[0]
    assert port.calls == 1 and attempt.artifact_ids == outputs
    assert attempt.report == "Exact original report.  \n" and attempt.report_artifact_id
    assert len(recovered.artifacts) == 3 and recovered.pending_module_results == {}
    assert recovered.pending_registrations == {}


@pytest.mark.parametrize("boundary", ["frozen", "registered"])
def test_interrupted_execution_record_registration_recovery_reuses_identity(tmp_path, monkeypatch, boundary):
    import json
    from resagent2_contracts import AttemptStatus, ModuleError, SessionRef, TaskStatus

    store, registry = setup_run(tmp_path)
    facts = {"results": [{"operation": "process", "command": "python train.py", "pid": 123,
        "execution_status": "interrupted", "stop_confirmed": True, "files_retained": True}]}
    class Port:
        calls = 0
        def invoke(self, request):
            self.calls += 1
            now = datetime.now(UTC)
            return AgentResult(status="interrupted", report="Command interrupted; files retained", session=SessionRef(
                id="session_stopped", module="executor", state_uri="session://session_stopped",
                status="paused", created_at=now, updated_at=now), artifacts=[ArtifactCandidate(
                    kind="execution_record", path="execution_record.json", media_type="application/json",
                    summary="Actual execution", content=json.dumps(facts))])

    port = Port()
    scheduler = WorkflowScheduler(bindings=executor_bindings({"experiment": ModuleBinding(owner=AgentOwner.EXECUTOR, port=port)}),
                                  store=store, artifact_root=registry.root, data_root=tmp_path / "data")
    accept_proposal(scheduler, "run_recovery", WorkflowProposal(work_request_id="work_original", tasks=[
        TaskProposal(execution_ref="experiment@1", id="task_result", work_request_id="work_original", workflow_agent_kind="executor",
                     instruction="Preserve stopped command evidence")]))
    with monkeypatch.context() as patch:
        if boundary == "frozen":
            original = artifacts_module.os.replace
            def crash_after_freeze(source, destination):
                original(source, destination)
                if Path(destination).name.startswith("artifact_"):
                    raise ProcessCrash()
            patch.setattr(artifacts_module.os, "replace", crash_after_freeze)
        else:
            original = scheduler.artifact_registry.register_many
            def crash_after_registration(*args, **kwargs):
                original(*args, **kwargs)
                raise ProcessCrash()
            patch.setattr(scheduler.artifact_registry, "register_many", crash_after_registration)
        with pytest.raises(ProcessCrash):
            scheduler.execute_task("run_recovery", "task_result")
    interrupted = store.load("run_recovery")
    allocated = next(iter(interrupted.pending_registrations.values())).ref.id
    assert scheduler._recover_interrupted_attempts_in_place(interrupted)
    recovered = store.load("run_recovery")
    task = recovered.workflow.tasks[0]
    attempt = task.attempts[0]
    assert port.calls == 1 and attempt.artifact_ids == [allocated]
    assert json.loads(Path(url2pathname(urlparse(recovered.artifacts[allocated].uri).path)).read_text()) == facts
    # Controller's ordinary cancellation finalizes this Attempt and freezes the
    # same report+artifact association, without inventing a successful delivery.
    task.status = TaskStatus.CANCELLED
    attempt.status = AttemptStatus.CANCELLED
    attempt.finished_at = datetime.now(UTC)
    attempt.error = ModuleError(code="interrupted", message="User stopped work", retryable=False)
    scheduler._persist_attempt_report(recovered, task, attempt)
    scheduler._save(recovered)
    scheduler._recover_interrupted_attempts_in_place(recovered)
    again = store.load("run_recovery")
    assert port.calls == 1 and sum(ref.kind == "execution_record" for ref in again.artifacts.values()) == 1
    assert again.artifacts[attempt.report_artifact_id].metadata["output_artifact_ids"] == [allocated]
    assert not again.pending_registrations and not again.pending_module_results


def test_execution_record_snapshots_reuse_only_equal_bytes_and_producing_attempt(tmp_path):
    import json
    from resagent2_orchestrator import ArtifactRegistrationError

    store, registry = setup_run(tmp_path)
    first = {"command": "python prepare.py", "exit_code": 0, "timed_out": False,
             "stdout_path": "first.stdout", "stderr_path": "first.stderr", "duration_seconds": 0.1}
    candidate = ArtifactCandidate(kind="execution_record", path="execution_record.json", media_type="application/json",
        summary="Cumulative execution snapshot", content=json.dumps({"results": [first]}))
    common = dict(grant=None, producer=AgentOwner.EXECUTOR, run_id="run_recovery", task_id="task_result",
                  attempt_number=1, index=1)
    original = registry.register(candidate, operation_key="pause:1", **common)
    registry.acknowledge_registration("pause:1", run_id="run_recovery")
    assert registry.register(candidate, operation_key="pause:2", **common) == original
    assert registry.register(candidate, operation_key="finish", **common) == original
    more = candidate.model_copy(update={"content": json.dumps({"results": [first, {**first, "command": "python analyze.py"}]})})
    latest = registry.register(more, operation_key="pause:3", **common)
    assert latest.id != original.id and latest.sha256 != original.sha256
    another_attempt = registry.register(candidate, operation_key="attempt:2", **{**common, "attempt_number": 2})
    assert another_attempt.id not in {original.id, latest.id}
    assert len(store.load("run_recovery").artifacts) == 3
    Path(url2pathname(urlparse(original.uri).path)).write_text("corrupted frozen execution record")
    with pytest.raises(ArtifactRegistrationError):
        registry.register(candidate, operation_key="after_corruption", **common)
    assert len(store.load("run_recovery").artifacts) == 3


def test_scheduler_keeps_artifacts_registered_during_agent_invocation(tmp_path):
    store, registry = setup_run(tmp_path)
    class Port:
        def invoke(self, request):
            ref = registry.register(ArtifactCandidate(kind="experiment_result", path="live.txt", media_type="text/plain",
                summary="Tool registered result", content="Live material"), grant=None, producer=request.agent,
                run_id=request.run_id, task_id=request.task_id, attempt_number=request.attempt_number,
                index=1, operation_key="live:output")
            return AgentResult(status="completed", report="Original report", artifacts=[ref])
    scheduler = WorkflowScheduler(bindings=executor_bindings({"experiment": ModuleBinding(owner=AgentOwner.EXECUTOR, port=Port())}),
                                  store=store, artifact_root=registry.root, data_root=tmp_path / "data")
    accept_proposal(scheduler, "run_recovery", WorkflowProposal(work_request_id="work_original", tasks=[
        TaskProposal(execution_ref="experiment@1", id="task_result", work_request_id="work_original", workflow_agent_kind="executor",
                     instruction="Keep live tool registration"),
    ]))
    result = scheduler.execute_task("run_recovery", "task_result")
    attempt = result.workflow.tasks[0].attempts[0]
    assert attempt.status == "completed" and len(attempt.artifact_ids) == 1
    assert result.artifacts[attempt.artifact_ids[0]].summary == "Tool registered result"


def test_scientific_unfrozen_allocation_grants_no_ref_and_is_consumed_after_receipt(tmp_path, monkeypatch):
    store, registry = setup_run(tmp_path)
    port = ScientificArtifactRegistration(registry, store)
    with monkeypatch.context() as patch:
        patch.setattr(artifacts_module.tempfile, "mkdtemp", lambda **kwargs: (_ for _ in ()).throw(ProcessCrash()))
        with pytest.raises(ProcessCrash):
            port.register_scientific(ArtifactCandidate(kind="literature_paper", path="paper.md", media_type="text/plain",
                summary="Paper", content="Exact bytes"), run_id="run_recovery", session_id="session_scientific",
                operation_key="session:session_scientific:turn:0:call:fetch", output_position=1)
    allocated = next(iter(store.load("run_recovery").pending_registrations.values())).ref.id
    assert port.recover_registration(run_id="run_recovery", operation_key="session:session_scientific:turn:0:call:fetch") == []
    assert port.resolve(allocated, run_id="run_recovery", session_id="session_scientific") is None
    port.acknowledge_registration(run_id="run_recovery", operation_key="session:session_scientific:turn:0:call:fetch")
    run = store.load("run_recovery")
    assert run.pending_registrations == {} and run.artifacts == {} and run.scientific_artifact_ids == []


@pytest.mark.parametrize("corruption", ["changed", "deleted"])
def test_registered_material_corruption_is_not_treated_as_unused_allocation(tmp_path, corruption):
    store, registry = setup_run(tmp_path)
    port = ScientificArtifactRegistration(registry, store)
    ref = port.register_scientific(ArtifactCandidate(kind="literature_paper", path="paper.md", media_type="text/plain",
        summary="Paper", content="Exact bytes"), run_id="run_recovery", session_id="session_scientific",
        operation_key="session:session_scientific:turn:0:call:fetch", output_position=1)
    path = Path(url2pathname(urlparse(ref.uri).path))
    path.unlink() if corruption == "deleted" else path.write_text("Changed", encoding="utf-8")
    for operation in (port.recover_registration, port.acknowledge_registration):
        with pytest.raises(ValueError, match="missing|sha256"):
            operation(run_id="run_recovery", operation_key="session:session_scientific:turn:0:call:fetch")
    assert store.load("run_recovery").pending_registrations
    assert port.resolve(ref.id, run_id="run_recovery", session_id="session_scientific") is None


@pytest.mark.parametrize("entry", ["import", "paper"])
def test_initialization_recovers_original_identity_before_first_agent_call(tmp_path, monkeypatch, entry):
    from resagent2_components.literature import LiteraturePaper, PreparedLiteratureImport
    class StopAfterPreparation(BaseException):
        pass
    class Port:
        requests = []
        def invoke(self, request):
            self.requests.append(request)
            raise StopAfterPreparation()
    port = Port()
    store = JsonRunStore(tmp_path / "state")
    scheduler = WorkflowScheduler(bindings=executor_bindings({}), store=store, artifact_root=tmp_path / "artifacts")
    controller = ResearchController(scientific_port=port, compiler=None, scheduler=scheduler,
        registry=default_execution_registry(),
        interpreter=DeterministicWorkInterpreter())
    source = tmp_path / "original.md"
    source.write_text("Original imported bytes", encoding="utf-8")
    request = ResearchRequest(goal="Recover preparation", permissions=RunPermissions(), budget=RunBudget(max_llm_calls=10, timeout_seconds=60),
        input_artifacts=[ArtifactImport(uri=str(source), kind="data", media_type="text/plain", summary="Original input")] if entry == "import" else [])
    literature = [PreparedLiteratureImport(paper=LiteraturePaper(paper_id="arxiv:1706.03762v7", title="Original paper",
        source_url="https://arxiv.org/abs/1706.03762v7", abstract="Original abstract"))] if entry == "paper" else []
    original_replace = artifacts_module.os.replace
    with monkeypatch.context() as patch:
        def interrupt_before_freeze(source, destination):
            if Path(destination).name.startswith("artifact_"):
                raise ProcessCrash()
            return original_replace(source, destination)
        patch.setattr(artifacts_module.os, "replace", interrupt_before_freeze)
        with pytest.raises(ProcessCrash):
            controller.create_run("run_initialization", request, literature=literature)
    interrupted = store.load("run_initialization")
    ref = next(iter(interrupted.pending_registrations.values())).ref
    assert port.requests == [] and not interrupted.initialization.complete
    with pytest.raises(StopAfterPreparation):
        controller.run_until_stable("run_initialization")
    recovered = store.load("run_initialization")
    assert recovered.initialization.complete and recovered.pending_registrations == {}
    assert recovered.artifacts[ref.id] == ref
    assert len(port.requests) == 1 and ref in port.requests[0].input_artifacts


@pytest.mark.parametrize("boundary", ["allocated", "frozen"])
def test_final_receiving_error_keeps_durable_registration_facts(tmp_path, monkeypatch, boundary):
    from e2e.native_fixtures import tool_turns
    from resagent2_runtime import JsonSessionStore, ScriptedLLMClient
    from resagent2_scientific import ScientificAgent
    store = JsonRunStore(tmp_path / "state")
    scheduler = WorkflowScheduler(bindings=executor_bindings({}), store=store, artifact_root=tmp_path / "artifacts")
    scientific = ScientificAgent(ScriptedLLMClient(tool_turns([{"tool": "finish", "arguments": {
        "report": "Original conclusion", "artifacts": [{"kind": "scientific_opinion", "path": "opinion.json",
            "media_type": "application/json", "summary": "Opinion",
            "content": {"verdict":"inconclusive","statement":"No research execution was needed"}}],
    }}])), store=JsonSessionStore(tmp_path / "sessions"))
    controller = ResearchController(scientific_port=scientific, compiler=None, scheduler=scheduler,
        registry=default_execution_registry(),
        interpreter=DeterministicWorkInterpreter())
    if boundary == "allocated":
        original = store.save
        def save_then_error(run):
            original(run)
            if any(item.ref.kind == "final_report" for item in run.pending_registrations.values()):
                raise OSError("Receiving interrupted after final allocation")
        monkeypatch.setattr(store, "save", save_then_error)
    else:
        original = scheduler.artifact_registry._freeze
        def freeze_then_error(candidate, **kwargs):
            ref = original(candidate, **kwargs)
            if candidate.kind == "final_report":
                raise OSError("Receiving interrupted after final freezing")
            return ref
        monkeypatch.setattr(scheduler.artifact_registry, "_freeze", freeze_then_error)
    # Allow the failure state itself to be saved after the one injected fault.
    if boundary == "allocated":
        triggered = False
        def fail_once(run):
            nonlocal triggered
            if triggered:
                return original(run)
            if any(item.ref.kind == "final_report" for item in run.pending_registrations.values()):
                triggered = True
            return save_then_error(run)
        monkeypatch.setattr(store, "save", fail_once)
    run = controller.create_run("run_final_receiving", ResearchRequest(goal="Preserve final registration",
        permissions=RunPermissions(), budget=RunBudget(max_llm_calls=5, timeout_seconds=60)))
    assert run.status == "failed"
    key = next(key for key, item in run.pending_registrations.items() if item.ref.kind == "final_report")
    pending = run.pending_registrations[key].ref
    assert (pending.id in run.artifacts) == (boundary == "frozen")
    assert run.final_report_artifact_id is None and run.final_opinion is None
    assert store.load(run.run_id).pending_registrations[key].ref == pending


@pytest.mark.parametrize("boundary", ["allocated", "frozen", "accepted"])
def test_compiler_result_is_reused_after_workflow_registration_interruption(tmp_path, monkeypatch, boundary):
    from resagent2_orchestrator import CompilationResult
    from resagent2_contracts import WorkflowPatch
    store, registry = setup_run(tmp_path)
    now = datetime.now(UTC)
    run = store.load("run_recovery")
    run.work_requests = [WorkRequest(owner_task_id="task_root", owner_attempt_number=1, depth=2, id="work_original", run_id=run.run_id, owner_session_id="session_scientific",
        request=WorkRequestDraft(objective='Deliver measurements', tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Deliver measurements'}]),
        status="requested", created_at=now, updated_at=now)]
    add_root_owner(run, "session_scientific")
    store.save(run)
    class Compiler:
        calls = 0
        def compile(self, request, **kwargs):
            self.calls += 1
            assert self.calls == 1, "Recover the original accepted compiler output without another model call"
            return CompilationResult(WorkflowPatch(work_request_id=request.id, based_on_revision=kwargs["current"].revision, add_tasks=[TaskProposal(execution_ref="experiment@1",
                id="task_original", work_request_id=request.id, workflow_agent_kind="executor",
                instruction="Deliver the measurement", output_names=["metrics"],
            )]))
    class Port:
        def invoke(self, request):
            return AgentResult(status="completed", report="Original measurement report", artifacts=[ArtifactCandidate(
                kind="experiment_result", path="result.json", media_type="application/json", summary="Measured result",
                output_name="metrics", content='{"accuracy": 0.8}',
            )])
    compiler = Compiler()
    scheduler = WorkflowScheduler(bindings=executor_bindings({"experiment": ModuleBinding(owner=AgentOwner.EXECUTOR, port=Port())}),
                                  store=store, artifact_root=registry.root, data_root=tmp_path / "data")
    controller = ResearchController(scientific_port=None, compiler=compiler, scheduler=scheduler,
        registry=default_execution_registry(),
        interpreter=DeterministicWorkInterpreter())
    with monkeypatch.context() as patch:
        if boundary == "allocated":
            original = store.save
            def save_then_crash(run):
                original(run)
                if any(pending.ref.kind == "acceptance_requirements" for pending in run.pending_registrations.values()):
                    raise ProcessCrash()
            patch.setattr(store, "save", save_then_crash)
        elif boundary == "frozen":
            original = scheduler.artifact_registry._freeze
            def freeze_then_crash(candidate, **kwargs):
                ref = original(candidate, **kwargs)
                if candidate.kind == "acceptance_requirements":
                    raise ProcessCrash()
                return ref
            patch.setattr(scheduler.artifact_registry, "_freeze", freeze_then_crash)
        else:
            original = scheduler.apply_patch
            def accept_then_crash(*args):
                original(*args)
                raise ProcessCrash()
            patch.setattr(scheduler, "apply_patch", accept_then_crash)
        with pytest.raises(ProcessCrash):
            controller._execute_work_request(run.run_id, "work_original")
    interrupted = store.load(run.run_id)
    assert interrupted.pending_compilation.add_tasks[0].id == "task_original" and compiler.calls == 1
    expected = (interrupted.task("task_original").acceptance_ref if boundary == "accepted" else
                next(pending.ref for pending in interrupted.pending_registrations.values()
                     if pending.ref.kind == "acceptance_requirements"))
    restored_scheduler = WorkflowScheduler(bindings=executor_bindings(scheduler.bindings), store=JsonRunStore(store.root),
        artifact_root=registry.root, data_root=tmp_path / "data")
    restored = ResearchController(scientific_port=None, compiler=compiler, scheduler=restored_scheduler,
        registry=controller.registry, interpreter=DeterministicWorkInterpreter())
    result = restored._execute_work_request(run.run_id, "work_original")
    result = restored_scheduler.run_until_stable(run.run_id)
    assert compiler.calls == 1 and result.pending_compilation is None
    assert result.task("task_original").acceptance_ref == expected
    assert result.task("task_original").id == "task_original" and result.task("task_original").status == "completed"
    assert result.pending_registrations == {}


def test_rejected_compilation_does_not_replace_the_next_work_request_graph(tmp_path):
    from e2e.native_fixtures import tool_turns
    from resagent2_orchestrator import CompilationResult
    from resagent2_contracts import WorkflowPatch
    from resagent2_runtime import JsonSessionStore, ScriptedLLMClient
    from resagent2_scientific import ScientificAgent
    class Compiler:
        work_ids = []
        def compile(self, request, **kwargs):
            self.work_ids.append(request.id)
            first = len(self.work_ids) == 1
            return CompilationResult(WorkflowPatch(work_request_id=request.id, based_on_revision=kwargs["current"].revision, add_tasks=[TaskProposal(execution_ref="experiment@1",
                id="task_rejected" if first else "task_accepted", work_request_id=request.id,
                workflow_agent_kind="executor", instruction="Execute the current work request",
                input_artifacts=["artifact_missing"] if first else [],
            )]))
    class Worker:
        def invoke(self, request):
            assert request.task_id == "task_accepted"
            return AgentResult(status="completed", report="The new work request completed")
    actions = [{"tool": "request_work", "arguments": {
        "assessment": {"statement": f"Request {index}"}, "work_request": {'objective': f'Run independent request {index}', 'tasks': [{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': f'Run independent request {index}'}]},
    }} for index in range(2)]
    actions.append({"tool": "finish", "arguments": {"report": "The replacement request completed", "artifacts": [{
        "kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json", "summary": "Conclusion",
        "content": {"verdict":"inconclusive","statement":"The second request completed","limitations":["The first work graph was rejected"]},
    }]}})
    compiler = Compiler()
    scheduler = WorkflowScheduler(bindings=executor_bindings({"experiment": ModuleBinding(owner=AgentOwner.EXECUTOR, port=Worker())}),
        store=JsonRunStore(tmp_path / "state"), artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    controller = ResearchController(scientific_port=ScientificAgent(ScriptedLLMClient(tool_turns(actions)),
        store=JsonSessionStore(tmp_path / "sessions")), compiler=compiler, scheduler=scheduler,
        registry=default_execution_registry(),
        interpreter=DeterministicWorkInterpreter())
    run = controller.create_run("run_replacement_compilation", ResearchRequest(goal="Recover from a rejected work graph",
        permissions=RunPermissions(), budget=RunBudget(max_llm_calls=10, timeout_seconds=60)))
    assert run.status == "completed", run.terminal_error
    assert compiler.work_ids == [work.id for work in run.work_requests if work.owner_task_id is not None] and len(set(compiler.work_ids)) == 2
    assert [work.status.value for work in run.work_requests if work.owner_task_id is not None] == ["failed", "consumed"]
    assert run.pending_compilation is None
    assert [task.id for task in execution_tasks(run)] == ["task_accepted"]
    assert run.task("task_accepted").work_request_id == compiler.work_ids[-1]


def test_final_candidates_have_distinct_slots_without_weakening_immutability(tmp_path):
    from resagent2_orchestrator.artifacts import ArtifactRegistrationError
    store, registry = setup_run(tmp_path)
    candidate = ArtifactCandidate(kind="final_report", path="final_report.md",
        media_type="text/markdown", summary="Final report", metadata={"source_type": "final_report"})
    def register(key, content="Identical candidate bytes"):
        return registry.register_final_report(candidate, content, run_id="run_recovery", operation_key=key)

    first = register("final_report:first")
    assert register("final_report:first") == first
    second = register("final_report:second")
    assert second.id != first.id and second.sha256 == first.sha256
    with pytest.raises(ArtifactRegistrationError, match="different content"):
        register("final_report:first", "Changed same candidate")
    run = store.load("run_recovery")
    run.final_report_artifact_id = second.id
    store.save(run)
    with pytest.raises(ArtifactRegistrationError, match="differs from committed"):
        register("final_report:third", "Changed final report")
    assert store.load("run_recovery").final_report_artifact_id == second.id


@pytest.mark.parametrize("damage", ["changed", "changed_unindexed", "missing"])
def test_retiring_final_report_preserves_integrity_checks(tmp_path, damage):
    from resagent2_orchestrator.artifacts import ArtifactRegistrationError
    store, registry = setup_run(tmp_path)
    candidate = ArtifactCandidate(kind="final_report", path="final_report.md",
        media_type="text/markdown", summary="Final report", metadata={"source_type": "final_report"})
    ref = registry.register_final_report(candidate, "Original report", run_id="run_recovery")
    if damage == "changed_unindexed":
        run = store.load("run_recovery")
        del run.artifacts[ref.id]
        store.save(run)
    frozen = Path(url2pathname(urlparse(ref.uri).path))
    if damage == "missing":
        frozen.unlink()
    else:
        frozen.write_text("Changed report")
    before = store.load("run_recovery").model_dump()
    with pytest.raises(ArtifactRegistrationError, match="hash changed or file is missing"):
        registry.settle_final_report_registrations(run_id="run_recovery")
    assert store.load("run_recovery").model_dump() == before


@pytest.mark.parametrize("entry", ["import", "task"])
@pytest.mark.parametrize("frozen", [False, True])
def test_final_report_settlement_does_not_take_over_other_registration_owners(tmp_path, monkeypatch, entry, frozen):
    store, registry = setup_run(tmp_path)
    source = tmp_path / "result.md"
    source.write_text("Original result")
    def register():
        if entry == "import":
            return registry.register_import(ArtifactImport(uri=str(source), kind="final_report",
                media_type="text/markdown", summary="User's report"), run_id="run_recovery",
                operation_key="initialization:import:0")
        return registry.register(ArtifactCandidate(kind="final_report", path=source.name,
            media_type="text/markdown", summary="Executor's report"),
            grant=WorkspaceGrant(root=str(tmp_path), access={"read_paths": ["."], "write_paths": ["."]}, source="local"),
            producer=AgentOwner.EXECUTOR, run_id="run_recovery", task_id="task_result",
            attempt_number=1, index=1)

    original = registry._freeze_prepared
    def crash(prepared, *, checkpoint=None):
        if frozen:
            original(prepared, checkpoint=checkpoint)
        raise ProcessCrash()
    with monkeypatch.context() as patch:
        patch.setattr(registry, "_freeze_prepared", crash)
        with pytest.raises(ProcessCrash):
            register()
    before = store.load("run_recovery")
    allocated = next(iter(before.pending_registrations.values())).ref
    registry.settle_final_report_registrations(run_id="run_recovery")
    assert store.load("run_recovery").model_dump() == before.model_dump()
    assert register() == allocated
