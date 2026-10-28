"""Attempt-owned dependency storage stays outside source and approval boundaries."""
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import pytest

from e2e.native_fixtures import tool_turns
from sciagent_capabilities import RunShellInput, RunShellTool
from sciagent_components import ProcessRunner, WorkspaceBoundary
from sciagent_components.execution_environment import ExecutionEnvironment
from sciagent_contracts import AgentOwner, AgentPermissions, AgentRequest, ExecutionDefinition, RecordedAnswer, TaskBudget, WorkspaceAccess, WorkspaceGrant
from sciagent_executor import NativeExecutorAgent, build_coding_definition, build_experiment_definition
from sciagent_runtime import AgentState, JsonSessionStore, ScriptedLLMClient


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="fixed Linux Bash execution")


def _request(root, **updates):
    values = dict(run_id="run_environment", work_request_id="work_environment", task_id="task_environment",
        attempt_number=1, agent=AgentOwner.EXECUTOR, instruction="Complete the task with actual execution evidence",
        execution_definition=build_experiment_definition(), budget=TaskBudget(max_llm_calls=8, timeout_seconds=30),
        workspace=WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(read_paths=["."], write_paths=["."])),
        output_dir=str(root.parent / "results"), permissions=AgentPermissions(execute_commands=True))
    values.update(updates)
    return AgentRequest(**values)


def _finish():
    return {"tool": "finish", "arguments": {"report": "Actual execution completed"}}


def _approve(request, result):
    question = result.control.question
    answer = RecordedAnswer(question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, answered_at=datetime.now(UTC), run_id=request.run_id,
        task_id=request.task_id, attempt_number=request.attempt_number, session_id=result.session.id,
        action=question.action)
    return request.model_copy(update={"parent_session_id": result.session.id,
        "resume_answers": [answer], "control_ack": result.control.origin})


def _runtime_context(client):
    return json.loads(client.contexts[-1].runtime_context.split("\n", 1)[1])


def _state():
    now = datetime.now(UTC)
    return AgentState(session_id="session_environment", agent_name="executor", owner=AgentOwner.EXECUTOR,
        run_id="run_environment", task_id="task_environment", attempt_number=1, created_at=now, updated_at=now)


def _shell(root, environment):
    grant = WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(read_paths=["."], write_paths=["."]))
    return RunShellTool(ProcessRunner(WorkspaceBoundary(grant)), timeout_seconds=5,
        log_dir=str(root.parent / "commands"), extra_env=environment.env_overrides(), execution_environment=environment)


def test_real_venv_dependencies_do_not_enter_patch_or_modify_real_index(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "value.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(root), "add", "value.py"], check=True)
    original_index = (root / ".git/index").read_bytes()
    script = """import json, os, pathlib, venv
p = pathlib.Path(os.environ['SCIAGENT_ENVIRONMENT_DIR'])
venv.EnvBuilder(with_pip=False).create(p)
packages = next(p.glob('lib/python*/site-packages')) / 'local_dependency'
packages.mkdir()
for i in range(128):
    (packages / f'module_{i}.py').write_text(f'DEPENDENCY_LEAF = {i}\\n')
pathlib.Path('value.py').write_text('VALUE = 2\\n')
pathlib.Path('environment_source.py').write_text('SOURCE_MUST_SURVIVE = True\\n')
pathlib.Path('cache').mkdir()
pathlib.Path('cache/source_logic.py').write_text('CACHE_NAMED_SOURCE = True\\n')
print(json.dumps({k: os.environ[k] for k in ['SCIAGENT_ENVIRONMENT_DIR', 'UV_PROJECT_ENVIRONMENT', 'PIP_CACHE_DIR', 'XDG_CACHE_HOME', 'UV_CACHE_DIR']}))
"""
    command = shlex.quote(sys.executable) + " -c " + shlex.quote(script)
    first_client = ScriptedLLMClient(tool_turns([{"tool": "run_shell", "arguments": {"command": command}}]))
    store = JsonSessionStore(tmp_path / "sessions")
    req = _request(root, execution_definition=build_coding_definition())
    paused = NativeExecutorAgent(lambda _: first_client, store=store).invoke(req)
    assert paused.status == "needs_user_input", paused.report
    memory = store.load(paused.session.id).memory["execution_environment"]
    environment = ExecutionEnvironment.from_memory(memory, workspace_root=root)
    assert not environment.environment_dir.exists(), "approval must precede venv creation"
    authority = _runtime_context(first_client)
    assert authority["execution_environment"] == environment.context()
    assert str(environment.root) not in first_client.contexts[-1].material_index
    assert paused.control.question.action.context["execution_environment"] == environment.context()
    assert "execution_environment" not in json.loads(first_client.contexts[-1].task_context.split("\n", 1)[1])
    second_client = ScriptedLLMClient(tool_turns([_finish()]))
    completed = NativeExecutorAgent(lambda _: second_client, store=JsonSessionStore(store.root)).invoke(_approve(req, paused))
    assert completed.status == "completed", completed.report
    assert (environment.environment_dir / "pyvenv.cfg").is_file()
    assert len(list(environment.environment_dir.glob("lib/python*/site-packages/local_dependency/module_*.py"))) == 128
    patch = (Path(req.output_dir) / next(item.path for item in completed.artifacts if item.kind == "code_patch")).read_text()
    assert "-VALUE = 1" in patch and "+VALUE = 2" in patch
    assert "environment_source.py" in patch and "cache/source_logic.py" in patch
    assert "DEPENDENCY_LEAF" not in patch and "pyvenv.cfg" not in patch and "site-packages" not in patch
    assert (root / ".git/index").read_bytes() == original_index
    rows = json.loads(next(item.content for item in completed.artifacts if item.kind == "execution_record"))["results"]
    assert rows[0]["exit_code"] == 0
    assert rows[0]["execution_environment"] == environment.context()
    assert json.loads(Path(rows[0]["stdout_path"]).read_text()) == environment.env_overrides()
    shutil.rmtree(environment.root)
    terminal_client = ScriptedLLMClient(tool_turns([_finish()]))
    recovered = NativeExecutorAgent(lambda _: terminal_client, store=JsonSessionStore(store.root)).invoke(_approve(req, paused))
    assert recovered.status == completed.status and recovered.report == completed.report
    assert recovered.artifacts == completed.artifacts and recovered.llm_calls == 0
    assert not terminal_client.contexts and not environment.root.exists()


@pytest.mark.parametrize("output_kind", ["inside_workspace", "missing"])
def test_pause_resume_reuses_external_root_when_output_directory_changes(tmp_path, monkeypatch, output_kind):
    root = tmp_path / "workspace"
    root.mkdir()
    system_temp = tmp_path / "system-temp"
    system_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_temp))
    output = str(root / "outputs") if output_kind == "inside_workspace" else None
    req = _request(root, output_dir=output)
    command = 'mkdir -p "$PIP_CACHE_DIR"; printf retained > "$PIP_CACHE_DIR/probe"'
    first_client = ScriptedLLMClient(tool_turns([{"tool": "run_shell", "arguments": {"command": command}}]))
    store = JsonSessionStore(tmp_path / "sessions")
    paused = NativeExecutorAgent(lambda _: first_client, store=store).invoke(req)
    assert paused.status == "needs_user_input", paused.report
    memory = store.load(paused.session.id).memory["execution_environment"]
    environment = ExecutionEnvironment.from_memory(memory, workspace_root=root)
    assert environment.root.is_relative_to(system_temp)
    assert not environment.root.is_relative_to(root)
    assert not environment.environment_dir.exists()
    probe = Path(environment.env_overrides()["PIP_CACHE_DIR"]) / "probe"
    second_client = ScriptedLLMClient(tool_turns([
        {"tool": "read_file", "arguments": {"path": str(probe)}}, _finish()]))
    recreated_store = JsonSessionStore(store.root)
    resumed = _approve(req, paused).model_copy(update={"output_dir": str(tmp_path / "different-output")})
    completed = NativeExecutorAgent(lambda _: second_client, store=recreated_store).invoke(resumed)
    assert completed.status == "completed", completed.report
    assert probe.read_text() == "retained"
    assert recreated_store.load(completed.session.id).memory["execution_environment"] == memory
    assert _runtime_context(second_client)["execution_environment"] == environment.context()
    state = recreated_store.load(completed.session.id)
    reads = [json.loads(turn.tool_results[call.id]) for turn in state.tool_turns
             for call in turn.tool_calls if call.name == "read_file"]
    assert len(reads) == 1 and reads[0]["ok"] is True
    assert "retained" in json.dumps(reads[0]["value"])


@pytest.mark.parametrize("restriction", ["no_shell", "no_execute", "read_only", "scoped", "denied"])
def test_ineligible_configuration_creates_no_environment_or_read_grant(tmp_path, monkeypatch, restriction):
    root = tmp_path / "workspace"
    root.mkdir()
    req = _request(root)
    if restriction == "no_shell":
        definition = req.execution_definition
        req = req.model_copy(update={"execution_definition": ExecutionDefinition(ref="custom-read@1",
            config=definition.config.model_copy(update={"tools": ["read_file"]}))})
    elif restriction == "no_execute":
        req = req.model_copy(update={"permissions": AgentPermissions(execute_commands=False)})
    else:
        access = {"read_only": WorkspaceAccess(read_paths=["."]),
            "scoped": WorkspaceAccess(read_paths=["."], write_paths=["src"]),
            "denied": WorkspaceAccess(read_paths=["."], write_paths=["."], denied_paths=["secret"])}[restriction]
        req = req.model_copy(update={"workspace": req.workspace.model_copy(update={"access": access})})
    monkeypatch.setattr(ExecutionEnvironment, "allocate", lambda **_: pytest.fail("ineligible invocation must not allocate environment storage"))
    client = ScriptedLLMClient(tool_turns([_finish()]))
    agent = NativeExecutorAgent(lambda _: client)
    result = agent.invoke(req)
    assert result.status == "completed", result.report
    assert "execution_environment" not in agent.loop.store.load(result.session.id).memory
    assert not _runtime_context(client).get("execution_environment")
    assert not (Path(req.output_dir) / "runtime").exists()


@pytest.mark.parametrize("corruption", ["missing_memory", "missing_root", "symlink_root"])
def test_resume_refuses_lost_environment_without_recreating_or_reselecting(tmp_path, corruption):
    root = tmp_path / "workspace"
    root.mkdir()
    store = JsonSessionStore(tmp_path / "sessions")
    first_client = ScriptedLLMClient(tool_turns([{"tool": "run_shell", "arguments": {"command": "touch executed"}}]))
    req = _request(root)
    paused = NativeExecutorAgent(lambda _: first_client, store=store).invoke(req)
    assert paused.status == "needs_user_input"
    state = store.load(paused.session.id)
    environment = ExecutionEnvironment.from_memory(state.memory["execution_environment"], workspace_root=root)
    if corruption == "missing_memory":
        del state.memory["execution_environment"]
        store.save(state)
    else:
        shutil.rmtree(environment.root)
        if corruption == "symlink_root":
            environment.root.symlink_to(root, target_is_directory=True)
    client = ScriptedLLMClient(tool_turns([_finish()]))
    failed = NativeExecutorAgent(lambda _: client, store=JsonSessionStore(store.root)).invoke(_approve(req, paused))
    assert failed.status == "blocked", failed.report
    assert not client.contexts and not (root / "executed").exists()
    if corruption == "missing_root":
        assert not environment.root.exists()


@pytest.mark.parametrize("change", ["environment_value", "metadata", "cache_symlink"])
def test_prepared_shell_rejects_changed_environment_before_process_launch(tmp_path, change):
    root = tmp_path / "workspace"
    root.mkdir()
    environment = ExecutionEnvironment.allocate(workspace_root=root, output_dir=str(tmp_path / "output"))
    tool = _shell(root, environment)
    args = RunShellInput(command="touch should_not_execute")
    prepared = tool.prepare(args)
    assert prepared["execution_environment"] == environment.context()
    if change == "environment_value":
        tool.extra_env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / "different-environment")
    elif change == "metadata":
        prepared["execution_environment"] = {**prepared["execution_environment"], "root": str(tmp_path / "different-root")}
    else:
        if environment.cache_dir.exists():
            shutil.rmtree(environment.cache_dir)
        environment.cache_dir.symlink_to(root, target_is_directory=True)
    result = tool.execute_prepared(_state(), args, prepared)
    assert not result.ok and result.value["execution_status"] == "not_executed"
    assert not (root / "should_not_execute").exists()
    assert not (tmp_path / "commands").exists()


def test_temporary_directory_inside_workspace_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    with pytest.raises((ValueError, PermissionError)):
        ExecutionEnvironment.allocate(workspace_root=root, output_dir=None)
    assert list(root.iterdir()) == []


def test_memory_cannot_grant_workspace_ancestor_or_redirected_environment(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises((ValueError, PermissionError)):
        ExecutionEnvironment.from_memory({"version": 1, "root": str(tmp_path)}, workspace_root=root)
    environment = ExecutionEnvironment.allocate(workspace_root=root, output_dir=str(tmp_path / "output"))
    assert environment.to_memory() == {"version": 1, "root": str(environment.root)}
    assert environment.env_overrides() == {
        "SCIAGENT_ENVIRONMENT_DIR": str(environment.environment_dir),
        "UV_PROJECT_ENVIRONMENT": str(environment.environment_dir),
        "XDG_CACHE_HOME": str(environment.cache_dir),
        "PIP_CACHE_DIR": str(environment.cache_dir / "pip"),
        "UV_CACHE_DIR": str(environment.cache_dir / "uv"),
    }
    assert not ({"HOME", "PATH", "VIRTUAL_ENV"} & environment.env_overrides().keys())
    environment.environment_dir.symlink_to(root, target_is_directory=True)
    with pytest.raises((ValueError, PermissionError)):
        ExecutionEnvironment.from_memory(environment.to_memory(), workspace_root=root)
