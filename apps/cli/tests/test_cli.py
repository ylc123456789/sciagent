from __future__ import annotations

from sciagent_contracts import WorkspaceAccess, RunPermissions, ExecutionLimits

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from sciagent_cli import composition
from sciagent_cli.composition import CliApplication, build_application
from sciagent_cli.main import EXIT_COMPLETED, EXIT_PAUSED, cli
from sciagent_contracts import (
    WorkflowAgentKind,
    ActionSnapshot,
    DatasetRef,
    PendingQuestion,
    ResearchRequest,
    RunBudget,
    RunStatus,
    WorkspaceRecord,
    WorkspaceSourceKind,
    WorkspaceSpec,
)
from sciagent_orchestrator import InMemoryRunStore, JsonRunStore, ResearchRun
from sciagent_runtime import AgentAction
from sciagent_cli.main import _parser, _specs_for_existing_run


def _request(goal: str = "test goal") -> ResearchRequest:
    return ResearchRequest(goal=goal, budget=RunBudget(max_llm_calls=10, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=2, max_attempts_per_task=1))


def _run(status: RunStatus = RunStatus.COMPLETED) -> ResearchRun:
    now = datetime.now(UTC)
    return ResearchRun(
        run_id="run_test",
        request=_request(),
        status=status,
        created_at=now,
        updated_at=now,
    )


class _Controller:
    def __init__(self, result: ResearchRun) -> None:
        self.result = result
        self.created = None
        self.answered = None
        self.resumed = None

    def create_run(self, run_id, request):
        self.created = (run_id, request)
        return self.result

    def answer_question(self, run_id, answer):
        self.answered = (run_id, answer)
        return self.result

    def resume_run(self, run_id):
        self.resumed = run_id
        return self.result


class _Builder:
    def __init__(self, controller: _Controller) -> None:
        self.controller = controller
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return CliApplication(controller=self.controller, run_store=InMemoryRunStore())


def test_composition_reads_explicit_model_and_component_context_limits(monkeypatch):
    monkeypatch.setenv("SCIAGENT_CONTEXT_WINDOW", "32000")
    monkeypatch.setenv("SCIAGENT_RESERVED_OUTPUT_TOKENS", "3000")
    monkeypatch.setenv("SCIAGENT_CONTEXT_SAFETY_MARGIN_TOKENS", "1000")
    monkeypatch.setenv("SCIAGENT_CODING_CONTEXT_TOKENS", "12000")

    profile = composition._model_profile()
    assert profile.context_window == 32000
    assert profile.reserved_output_tokens == 3000
    assert profile.safety_margin_tokens == 1000
    assert composition._component_context_limit("coding") == 12000


def test_agents_and_execution_presets_share_default_with_independent_overrides(monkeypatch):
    for component, default in (
        ("coding", 256_000), ("experiment", 256_000),
        ("scientific", 256_000),
    ):
        name = f"SCIAGENT_{component.upper()}_CONTEXT_TOKENS"
        monkeypatch.delenv(name, raising=False)
        assert composition._component_context_limit(component) == default
        monkeypatch.setenv(name, "1024")
        assert composition._component_context_limit(component) == 1024


def test_fixed_compiler_has_no_model_client(tmp_path, monkeypatch):
    from sciagent_orchestrator import FixedWorkflowCompiler
    from sciagent_runtime import ScriptedLLMClient

    clients = []

    def client():
        result = ScriptedLLMClient([])
        clients.append(result)
        return result

    monkeypatch.setattr(composition, "_client", client)
    app = composition.build_application(data_root=tmp_path)
    assert isinstance(app.controller.compiler, FixedWorkflowCompiler)
    assert not hasattr(app.controller.compiler, "_client")
    assert len(clients) == 2  # Scientific and operation reviewer; Executor resolves lazily.
    assert all(item.contexts == [] for item in clients)


def test_run_passes_goal_and_workspace_to_existing_interfaces(tmp_path: Path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    controller = _Controller(_run())
    builder = _Builder(controller)

    result = cli(
        [
            "run",
            "--run-id",
            "run_test",
            "--goal",
            "inspect this repository",
            "--workspace",
            str(workspace),
            "--data-root",
            str(tmp_path / "data"),
        ],
        application_builder=builder,
    )

    assert result == EXIT_COMPLETED
    assert controller.created[0] == "run_test"
    assert controller.created[1].goal == "inspect this repository"
    assert builder.calls[0]["workspaces"]["ws_main"].location == str(workspace.resolve())


@pytest.mark.parametrize("names", [[], ["metrics.json", "Summary", "metrics.json"]])
def test_run_passes_only_explicit_required_artifacts_to_controller(tmp_path, names):
    controller = _Controller(_run())
    flags = [part for name in names for part in ("--required-artifact", name)]

    result = cli(
        ["run", "--goal", "Produce metrics.json", "--data-root", str(tmp_path), *flags],
        application_builder=_Builder(controller),
    )

    assert result == EXIT_COMPLETED
    assert controller.created[1].required_artifacts == names


def test_run_rejects_invalid_required_artifact_before_building_application(tmp_path):
    controller = _Controller(_run())
    builder = _Builder(controller)

    with pytest.raises(ValidationError, match="required_artifacts"):
        cli(
            ["run", "--goal", "Produce metrics", "--data-root", str(tmp_path),
             "--required-artifact", "results/metrics.json"],
            application_builder=builder,
        )

    assert controller.created is None
    assert builder.calls == []


def test_goal_file_is_read_explicitly(tmp_path: Path):
    goal_file = tmp_path / "goal.txt"
    goal_file.write_text("a long research goal", encoding="utf-8")
    controller = _Controller(_run())

    result = cli(
        [
            "run",
            "--run-id",
            "run_test",
            "--goal-file",
            str(goal_file),
            "--data-root",
            str(tmp_path / "data"),
        ],
        application_builder=_Builder(controller),
    )

    assert result == EXIT_COMPLETED
    assert controller.created[1].goal == "a long research goal"


@pytest.mark.parametrize("answer", ["accuracy", "第二个", "第二个，mul=2*3"])
def test_answer_uses_persisted_question(tmp_path: Path, answer: str):
    run = _run(RunStatus.PAUSED)
    run.pending_question = PendingQuestion(
        session_id="session_test",
        id="question_metric",
        run_id=run.run_id,
        text="Choose the primary metric",
        requested_fields=["primary_metric"],
        created_at=datetime.now(UTC),
    )
    store = InMemoryRunStore()
    store.save(run)
    controller = _Controller(_run())

    result = cli(
        [
            "answer",
            run.run_id,
            "--field",
            f"primary_metric={answer}",
            "--data-root",
            str(tmp_path / "data"),
        ],
        application_builder=_Builder(controller),
        store_factory=lambda root: store,
    )

    assert result == EXIT_COMPLETED
    assert controller.answered[0] == run.run_id
    assert controller.answered[1].question_id == "question_metric"
    assert controller.answered[1].values == {"primary_metric": answer}


def test_answer_rejects_invalid_key_before_calling_controller(tmp_path: Path):
    run = _run(RunStatus.PAUSED)
    run.pending_question = PendingQuestion(
        session_id="session_test",
        id="question_mode", run_id=run.run_id, text="Which mode?",
        requested_fields=["mode"], created_at=datetime.now(UTC),
    )
    store = InMemoryRunStore()
    store.save(run)
    controller = _Controller(_run())

    with pytest.raises(ValidationError, match="String should match pattern"):
        cli(
            ["answer", run.run_id, "--field", "mode choice=第二个",
             "--data-root", str(tmp_path / "data")],
            application_builder=_Builder(controller), store_factory=lambda root: store,
        )

    assert controller.answered is None
    assert store.load(run.run_id).pending_question == run.pending_question


@pytest.mark.parametrize("decision", ["yes", "no"])
def test_operation_answer_loads_disk_snapshot_and_passes_separate_comment(tmp_path: Path, decision):
    run = _run(RunStatus.PAUSED)
    action = ActionSnapshot(action_id="action_train", tool="run_shell",
        arguments={"command": "python train.py"}, context={"cwd": "/workspace", "environment": "/env"},
        run_id=run.run_id, session_id="session_coding", task_id="task_train", attempt_number=1)
    run.pending_question = PendingQuestion(id="question_operation", run_id=run.run_id,
        session_id=action.session_id, task_id=action.task_id, attempt_number=1,
        text="Confirm this command", requested_fields=["approve"], action=action, created_at=datetime.now(UTC))
    data_root = tmp_path / "data"
    store = JsonRunStore(data_root / "state")
    store.save(run)
    before = (data_root / "state" / f"{run.run_id}.json").read_bytes()
    controller = _Controller(_run())
    comment = "  使用原任务指定的顺序。\n不要提前评估。  "

    assert cli(["answer", run.run_id, decision, "--comment", comment, "--data-root", str(data_root)],
        application_builder=_Builder(controller)) == EXIT_COMPLETED

    assert controller.answered[1].values == {"approve": decision}
    assert controller.answered[1].comment == comment
    assert controller.answered[1].question_id == run.pending_question.id
    assert (data_root / "state" / f"{run.run_id}.json").read_bytes() == before


@pytest.mark.parametrize("is_approval, field, heading", [
    (False, "metric", "Pending user question:"),
    (False, "approve", "Pending user question:"),
    (True, "approve", "Pending operation approval:"),
])
def test_show_reads_store_without_building_application(
    tmp_path: Path, capsys, is_approval, field, heading,
):
    run = _run(RunStatus.PAUSED)
    action = ActionSnapshot(
        action_id="action_train", tool="run_shell",
        arguments={"command": "python train.py"},
        context={"cwd": "/workspace", "environment": "/env"},
        run_id=run.run_id, session_id="session_coding",
        task_id="task_train", attempt_number=1,
    ) if is_approval else None
    run.pending_question = PendingQuestion(
        session_id="session_test",
        id="question_next", run_id=run.run_id, text="Choose the next step",
        requested_fields=[field], created_at=datetime.now(UTC), action=action,
    )
    store = InMemoryRunStore()
    store.save(run)

    result = cli(
        ["show", "run_test", "--data-root", str(tmp_path / "data")],
        application_builder=lambda **kwargs: (_ for _ in ()).throw(AssertionError()),
        store_factory=lambda root: store,
    )

    assert result == EXIT_COMPLETED
    output = capsys.readouterr().out
    assert "Status: paused" in output
    assert heading in output
    assert "Choose the next step" in output
    assert f"Fields: {field}" in output


def test_resume_uses_controller_and_returns_paused_exit(tmp_path: Path):
    run = _run(RunStatus.RUNNING)
    store = InMemoryRunStore()
    store.save(run)
    controller = _Controller(_run(RunStatus.PAUSED))

    result = cli(
        ["resume", run.run_id, "--data-root", str(tmp_path / "data")],
        application_builder=_Builder(controller),
        store_factory=lambda root: store,
    )

    assert result == EXIT_PAUSED
    assert controller.resumed == run.run_id


def test_production_composition_builds_without_calling_external_services(tmp_path: Path):
    application = build_application(data_root=tmp_path)

    assert application.controller.scheduler.store is application.run_store


def test_production_composition_loads_the_shared_dataset_catalog(
    tmp_path: Path, monkeypatch
):
    dataset_root = tmp_path / "datasets"
    (dataset_root / "cifar-10").mkdir(parents=True)
    (dataset_root / "catalog.json").write_text(
        '{"cifar10": "cifar-10"}', encoding="utf-8"
    )
    monkeypatch.setenv("SCIAGENT_DATASET_ROOT", str(dataset_root))

    application = build_application(data_root=tmp_path / "data")

    assert application.controller.dataset_ref_source.references() == [
        DatasetRef(dataset_id="cifar10", relative_path="cifar-10")
    ]


def test_scientific_and_executor_share_resource_layout(tmp_path: Path):
    application = build_application(data_root=tmp_path)
    bindings = application.controller.scheduler.bindings

    executor = bindings[WorkflowAgentKind.EXECUTOR].port

    # Both execution presets use this one configured Executor and resource layout.
    assert application.controller.scientific_port.resource_layout is executor.resource_layout


def test_resume_reuses_persisted_workspace(
    tmp_path: Path,
):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    source = WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.LOCAL, location=str(workspace), access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))
    run = _run(RunStatus.PAUSED)
    run.workspaces["ws_main"] = WorkspaceRecord(
        workspace_id="ws_main",
        root=str(workspace),
        source=source,
        managed=False,
    )
    args = _parser().parse_args(["resume", run.run_id, "--workspace", str(workspace)])

    assert _specs_for_existing_run(args, run) == {"ws_main": source}


@pytest.mark.parametrize("command", ["answer", "resume"])
def test_recovery_rejects_adding_workspace(tmp_path, command):
    run = _run(RunStatus.PAUSED)
    store = InMemoryRunStore()
    store.save(run)
    controller = _Controller(run)
    builder = _Builder(controller)
    tokens = [command, run.run_id, "--workspace", str(tmp_path)]
    if command == "answer":
        tokens += ["--field", "ready=yes"]
    with pytest.raises(ValueError, match="cannot add a workspace"):
        cli(tokens, application_builder=builder, store_factory=lambda root: store)
    assert builder.calls == []
    assert store.load(run.run_id).model_dump_json() == run.model_dump_json()


@pytest.mark.parametrize("flags", [["--read-only"], ["--read-path", "src"],
                                   ["--write-path", "src"], ["--deny-path", "secrets"]])
def test_workspace_permissions_require_a_workspace_source(flags):
    args = _parser().parse_args(["resume", "run_test"] + flags)
    with pytest.raises(ValueError, match="require --workspace or --git"):
        _specs_for_existing_run(args, _run())


@pytest.mark.parametrize("flags,accepted", [
    (["--read-path", "src", "--read-only", "--deny-path", "src/private"], True),
    (["--read-path", "src", "--read-only"], False),
    (["--read-path", "src", "--write-path", "src", "--deny-path", "src/private"], False),
    (["--read-only"], False),
    ([], False),
])
def test_recovery_workspace_flags_can_only_repeat_persisted_grants(tmp_path, flags, accepted):
    run = _run(RunStatus.PAUSED)
    source = WorkspaceSpec(
        workspace_id="ws_main", source_kind="local", location=str(tmp_path),
        access=WorkspaceAccess(read_paths=["src"], write_paths=[], denied_paths=["src/private"]),
    )
    run.workspaces["ws_main"] = WorkspaceRecord(
        workspace_id="ws_main", root=str(tmp_path), source=source, managed=False,
    )
    args = _parser().parse_args(["resume", run.run_id, "--workspace", str(tmp_path)] + flags)
    if accepted:
        assert _specs_for_existing_run(args, run) == {"ws_main": source}
    else:
        with pytest.raises(ValueError, match="does not match"):
            _specs_for_existing_run(args, run)
    assert _specs_for_existing_run(_parser().parse_args(["resume", run.run_id]), run) == {
        "ws_main": source,
    }


def test_run_passes_normalized_literature_to_controller(tmp_path):
    import json

    class Controller(_Controller):
        def create_run(self, run_id, request, *, literature=()):
            self.literature = literature
            return super().create_run(run_id, request)

    manifest = tmp_path / "papers.json"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    manifest.write_text(json.dumps({"papers": [{"title": "External paper", "doi": "10.1/example",
                                             "pdf_path": "paper.pdf"}]}))
    controller = Controller(_run())
    result = cli(
        ["run", "--goal", "Use this paper", "--literature-file", str(manifest), "--data-root", str(tmp_path)],
        application_builder=_Builder(controller),
    )
    assert result == EXIT_COMPLETED
    assert controller.literature[0].paper.title == "External paper"
    assert controller.literature[0].pdf_path == pdf.resolve()


def test_literature_import_returns_success_without_advancing_paused_run(tmp_path):
    import json

    class Controller(_Controller):
        def import_literature(self, run_id, literature):
            self.imported = (run_id, literature)
            return self.result

    paused = _run(RunStatus.PAUSED)
    store = InMemoryRunStore()
    store.save(paused)
    controller = Controller(paused)
    builder = _Builder(controller)
    manifest = tmp_path / "papers.json"
    manifest.write_text(json.dumps({"papers": [{"title": "External paper"}]}))
    result = cli(
        ["literature", "import", paused.run_id, "--file", str(manifest), "--data-root", str(tmp_path)],
        application_builder=builder, store_factory=lambda root: store,
    )
    assert result == EXIT_COMPLETED
    assert controller.imported[0] == paused.run_id
    assert controller.imported[1][0].paper.title == "External paper"
    assert controller.created is controller.answered is controller.resumed is None
    assert store.load(paused.run_id).status == RunStatus.PAUSED


@pytest.mark.parametrize("status", [RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.FAILED])
def test_literature_import_rejects_non_paused_run_before_building(tmp_path, status):
    import json
    run = _run(status)
    store = InMemoryRunStore()
    store.save(run)
    controller = _Controller(run)
    builder = _Builder(controller)
    manifest = tmp_path / "papers.json"
    manifest.write_text(json.dumps({"papers": [{"title": "External paper"}]}))
    with pytest.raises(ValueError, match="paused Run"):
        cli(
            ["literature", "import", run.run_id, "--file", str(manifest), "--data-root", str(tmp_path)],
            application_builder=builder, store_factory=lambda root: store,
        )
    assert builder.calls == []
    assert store.load(run.run_id) == run


@pytest.mark.parametrize("flag", [["--python-version", "3.12"], ["--prepare-environment"], ["--no-prepare-environment"]])
def test_retired_environment_flags_are_explicitly_rejected(flag):
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["run", "--goal", "Inspect the workspace", *flag])
    assert error.value.code == 2


def test_python_requirement_uses_the_normal_constraint_channel():
    from sciagent_cli.main import _request_from_args

    requirement = "Use Python 3.11 for this task; reuse /shared/envs/project if suitable."
    args = _parser().parse_args(["run", "--goal", "Execute the task", "--constraint", requirement])
    request = _request_from_args(args)
    assert request.constraints == [requirement]
    assert "prepare_environment" not in request.permissions.model_dump()


@pytest.mark.parametrize("command", ["show", "resume", "answer"])
def test_schema27_cli_recovery_refuses_without_mutating_evidence(tmp_path, command):
    from sciagent_contracts import SCHEMA_VERSION

    store = JsonRunStore(tmp_path / "state")
    run = _run(RunStatus.PAUSED)
    store.save(run)
    path = store.root / f"{run.run_id}.json"
    old = path.read_bytes().replace(f'"{SCHEMA_VERSION}"'.encode(), b'"27.0"')
    path.write_bytes(old)
    builder = _Builder(_Controller(run))
    args = [command, run.run_id, "--data-root", str(tmp_path)]
    if command == "answer":
        args += ["ready=yes"]
    with pytest.raises(ValidationError):
        cli(args, application_builder=builder)
    assert path.read_bytes() == old
    assert builder.calls == []


@pytest.mark.parametrize("mode,execute_commands", [("manual", False), ("manual", True),
                                                       ("auto_review", False), ("auto_review", True)])
def test_run_persists_approval_mode_without_expanding_execute_grant(tmp_path, mode, execute_commands):
    controller = _Controller(_run())
    result = cli(
        ["run", "--goal", "Inspect the repository", "--data-root", str(tmp_path),
         "--approval-mode", mode,
         "--execute-commands" if execute_commands else "--no-execute-commands"],
        application_builder=_Builder(controller),
    )
    assert result == EXIT_COMPLETED
    permissions = controller.created[1].permissions
    assert permissions.approval_mode == mode
    assert permissions.execute_commands is execute_commands


def test_run_defaults_to_manual_approval_and_resume_cannot_override_it():
    from sciagent_cli.main import _request_from_args

    request = _request_from_args(_parser().parse_args(["run", "--goal", "Inspect"]))
    assert request.permissions.approval_mode == "manual"
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["resume", "run_test", "--approval-mode", "auto_review"])
    assert error.value.code == 2


def test_run_rejects_unknown_approval_mode_before_building_application(tmp_path):
    controller = _Controller(_run())
    builder = _Builder(controller)
    with pytest.raises(SystemExit) as error:
        cli(["run", "--goal", "Inspect", "--data-root", str(tmp_path),
             "--approval-mode", "always_allow"], application_builder=builder)
    assert error.value.code == 2
    assert builder.calls == []


def test_run_rejects_retired_confirmation_flag():
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["run", "--goal", "Inspect", "--confirm-commands"])
    assert error.value.code == 2
