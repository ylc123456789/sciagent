import hashlib
import json
from types import SimpleNamespace

import pytest

from sciagent_contracts import AgentOwner, ArtifactRef, AttemptStatus, WorkflowAgentKind, RunStatus, TaskStatus

from e2e.real_e2e import _application, _literature_succeeded, _real_e2e_succeeded, _repair_succeeded


def _artifact(tmp_path, task, kind, content, *, number=1, metadata=None):
    artifact_id = f"artifact_{task.id}_{number}_{kind}"
    text = content if isinstance(content, str) else json.dumps(content)
    path = tmp_path / artifact_id
    path.write_text(text)
    return ArtifactRef(
        id=artifact_id, kind=kind, producer=AgentOwner(task.workflow_agent_kind.value),
        run_id="run_real", task_id=task.id, attempt_number=number,
        uri=path.as_uri(), sha256=hashlib.sha256(text.encode()).hexdigest(),
        media_type="text/x-diff" if kind == "code_patch" else "application/json",
        summary=kind, metadata=metadata or {},
    )


def _command(command="python train.py --epochs 1 --seed 0", exit_code=0):
    return dict(command=command, exit_code=exit_code, timed_out=False,
                stdout_path="stdout.log", stderr_path="stderr.log", duration_seconds=1)


def _run(tmp_path, *, metrics=None) -> SimpleNamespace:
    tasks = [
        SimpleNamespace(
            id=f"task_{preset}",
            workflow_agent_kind=WorkflowAgentKind.EXECUTOR,
            status=TaskStatus.COMPLETED,
            attempts=[SimpleNamespace(number=1, status=AttemptStatus.COMPLETED, artifact_ids=[])],
        )
        for preset in ("coding", "experiment")
    ]
    patch = _artifact(tmp_path, tasks[0], "code_patch", "--- a/train.py\n+++ b/train.py\n@@ -1 +1 @@\n-old\n+new\n")
    measured = _artifact(tmp_path, tasks[1], "experiment_result", metrics or {
        "baseline_accuracy": 0.6, "candidate_accuracy": 0.62,
        "epochs": 1, "seed": 0, "device": "cuda",
    }, metadata={"source_path": "metrics.json", "source_root": "workspace"})
    execution = _artifact(tmp_path, tasks[1], "execution_record", {"results": [_command()]})
    tasks[0].attempts[0].artifact_ids = [patch.id]
    tasks[1].attempts[0].artifact_ids = [measured.id, execution.id]
    tasks.append(SimpleNamespace(
        id="task_root", workflow_agent_kind=WorkflowAgentKind.SCIENTIFIC,
        status=TaskStatus.COMPLETED,
        attempts=[SimpleNamespace(number=1, status=AttemptStatus.COMPLETED, artifact_ids=[])],
    ))
    artifacts = {ref.id: ref for ref in (patch, measured, execution)}
    artifacts["artifact_final_report"] = ArtifactRef(
        id="artifact_final_report", kind="final_report", producer=AgentOwner.ORCHESTRATOR,
        run_id="run_real", uri=patch.uri, sha256=patch.sha256, media_type="text/markdown",
        summary="Final report", metadata={"source_type": "final_report"},
    )
    return SimpleNamespace(
        run_id="run_real",
        status=RunStatus.COMPLETED,
        final_opinion=SimpleNamespace(evidence_artifact_ids=[measured.id]),
        final_report_artifact_id="artifact_final_report",
        workflow=SimpleNamespace(tasks=tasks),
        artifacts=artifacts,
    )


def test_real_e2e_accepts_frozen_training_metrics_and_extra_task(tmp_path) -> None:
    run = _run(tmp_path)
    run.workflow.tasks.append(SimpleNamespace(
        id="task_smoke", workflow_agent_kind=WorkflowAgentKind.EXECUTOR,
        status=TaskStatus.COMPLETED,
        attempts=[SimpleNamespace(number=1, status=AttemptStatus.COMPLETED, artifact_ids=[])],
    ))

    assert _real_e2e_succeeded(run)


@pytest.mark.parametrize("kind", ["code_patch", "execution_record", "experiment_result", "final_report"])
def test_real_e2e_requires_registered_evidence(tmp_path, kind) -> None:
    run = _run(tmp_path)
    run.artifacts = {key: ref for key, ref in run.artifacts.items() if ref.kind != kind}

    assert not _real_e2e_succeeded(run)


@pytest.mark.parametrize("change", [
    {"baseline_accuracy": "0.6"}, {"candidate_accuracy": True},
    {"candidate_accuracy": float("nan")}, {"baseline_accuracy": 2},
    {"epochs": 0}, {"seed": 1}, {"device": "cpu"},
])
def test_real_e2e_rejects_invalid_or_incomplete_training(tmp_path, change) -> None:
    metrics = {"baseline_accuracy": 0.6, "candidate_accuracy": 0.62,
               "epochs": 1, "seed": 0, "device": "cuda"} | change
    assert not _real_e2e_succeeded(_run(tmp_path, metrics=metrics))


@pytest.mark.parametrize("rows", [
    [_command("python smoke.py")], [_command(exit_code=1), _command("python --version")],
    [dict(_command(), timed_out=True)], [_command("echo train.py")],
    [_command(), _command(exit_code=1)],
    [_command(exit_code=1), _command("python train.py --epochs 0")],
])
def test_real_e2e_requires_successful_training_not_just_a_command(tmp_path, rows) -> None:
    run = _run(tmp_path)
    ref = _artifact(tmp_path, run.workflow.tasks[1], "execution_record", {"results": rows})
    run.artifacts[ref.id] = ref
    assert not _real_e2e_succeeded(run)


def test_real_e2e_rejects_uncited_metrics(tmp_path) -> None:
    run = _run(tmp_path)
    run.final_opinion.evidence_artifact_ids = []
    assert not _real_e2e_succeeded(run)


def test_real_e2e_accepts_cited_metrics_without_a_read_log_gate(tmp_path) -> None:
    run = _run(tmp_path)
    assert not hasattr(run, "scientific_observed_artifact_ids")
    assert _real_e2e_succeeded(run)


def test_real_e2e_rejects_inline_model_metrics(tmp_path) -> None:
    run = _run(tmp_path)
    key = run.final_opinion.evidence_artifact_ids[0]
    run.artifacts[key] = run.artifacts[key].model_copy(update={"metadata": {}})
    assert not _real_e2e_succeeded(run)


def test_real_e2e_rejects_metrics_from_another_attempt(tmp_path) -> None:
    run = _run(tmp_path)
    key = run.final_opinion.evidence_artifact_ids[0]
    run.artifacts[key] = run.artifacts[key].model_copy(update={"attempt_number": 2})
    assert not _real_e2e_succeeded(run)


def test_real_e2e_rejects_tampered_metrics(tmp_path) -> None:
    run = _run(tmp_path)
    (tmp_path / run.final_opinion.evidence_artifact_ids[0]).write_text("{}")
    assert not _real_e2e_succeeded(run)


def test_real_e2e_requires_a_final_opinion(tmp_path) -> None:
    run = _run(tmp_path)
    run.final_opinion = None
    assert not _real_e2e_succeeded(run)


def _repair_run(tmp_path):
    run = _run(tmp_path, metrics={"accuracy": 0.8})
    failed_task = SimpleNamespace(
        id="task_failed", workflow_agent_kind=WorkflowAgentKind.EXECUTOR,
        status=TaskStatus.FAILED,
        attempts=[SimpleNamespace(number=1, status=AttemptStatus.FAILED, artifact_ids=[])],
    )
    ref = _artifact(tmp_path, failed_task, "execution_record", {"results": [_command(exit_code=1)]})
    failed_task.attempts[0].artifact_ids = [ref.id]
    run.artifacts[ref.id] = ref
    run.workflow.tasks.insert(0, failed_task)
    return run


def test_repair_requires_preserved_failure_patch_and_measured_recovery(tmp_path) -> None:
    assert _repair_succeeded(_repair_run(tmp_path))


@pytest.mark.parametrize("missing", ["failure", "patch", "metrics", "execution"])
def test_repair_rejects_incomplete_recovery(tmp_path, missing) -> None:
    run = _repair_run(tmp_path)
    if missing == "failure":
        run.workflow.tasks[0].attempts[0].status = AttemptStatus.COMPLETED
    else:
        kind = {"patch": "code_patch", "metrics": "experiment_result", "execution": "execution_record"}[missing]
        run.artifacts = {key: ref for key, ref in run.artifacts.items()
                         if ref.kind != kind or ref.task_id == "task_failed"}
    assert not _repair_succeeded(run)


def test_real_e2e_uses_the_configured_current_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SCIAGENT_WEB_SEARCH_PROVIDER", "off")
    monkeypatch.delenv("SCIAGENT_MODEL", raising=False)
    assert _application(tmp_path / "default").controller.scientific_port.llm_client.model == "deepseek-v4-flash"

    monkeypatch.setenv("SCIAGENT_MODEL", "deepseek-v4-pro")
    assert _application(tmp_path / "configured").controller.scientific_port.llm_client.model == "deepseek-v4-pro"


def test_direct_agent_uses_the_shared_declared_dataset_context(tmp_path):
    from e2e.real_e2e import _dataset_context
    from sciagent_components import ResourceLayout, request_dataset_refs
    from sciagent_contracts import AgentPermissions, AgentRequest, DatasetRef, TaskBudget
    layout = ResourceLayout(resource_root=tmp_path / "resources")
    layout.dataset_root.mkdir(parents=True)
    catalog = layout.dataset_root / "catalog.json"
    catalog.write_text('{"demo": "demo"}')
    context = _dataset_context(layout)
    request = AgentRequest(work_request_id="work_direct", task_id="task_root", attempt_number=1,
        run_id="run_direct", agent=AgentOwner.SCIENTIFIC, instruction="Inspect available data",
        permissions=AgentPermissions(), budget=TaskBudget(max_llm_calls=5, timeout_seconds=30),
        task_context=context,
    )
    assert request.input_artifacts == []
    assert request_dataset_refs(request) == [DatasetRef(dataset_id="demo", relative_path="demo")]
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("kind,accepted", [
    ("literature_paper", True), ("literature_fulltext", True),
    ("web_page", False), ("literature_pdf", False),
])
def test_literature_evaluation_requires_readable_literature_material(tmp_path, kind, accepted):
    run = _run(tmp_path)
    path = tmp_path / "literature.txt"
    path.write_text("Registered source content")
    ref = ArtifactRef(
        id="artifact_literature", kind=kind, producer=AgentOwner.SCIENTIFIC,
        run_id=run.run_id, session_id="session_literature", uri=path.as_uri(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        media_type="text/plain", summary="Registered literature",
    )
    run.artifacts[ref.id] = ref
    run.final_opinion.evidence_artifact_ids = [ref.id]
    assert _literature_succeeded(run) is accepted
    if accepted:
        path.write_text("Changed after registration")
        assert not _literature_succeeded(run)


def test_training_acceptance_allows_one_executor_to_implement_and_measure(tmp_path):
    run = _run(tmp_path)
    implementation, measurement, scientific = run.workflow.tasks
    ids = measurement.attempts[0].artifact_ids
    for artifact_id in ids:
        run.artifacts[artifact_id] = run.artifacts[artifact_id].model_copy(
            update={"task_id": implementation.id})
    implementation.attempts[0].artifact_ids.extend(ids)
    run.workflow.tasks = [implementation, scientific]
    assert _real_e2e_succeeded(run)
