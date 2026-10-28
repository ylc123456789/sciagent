"""Every CLI process reloads the deployment environment, including resumes."""

import json
import os
from pathlib import Path
import shlex
import shutil
import select
import signal
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "run-configured.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash deployment entry point")
def test_separate_cli_processes_share_config_and_preserve_arguments(tmp_path):
    config = tmp_path / "deployment config.sh"
    trace_dir = tmp_path / "trace directory"
    config.write_text(
        "SCIAGENT_LLM_TRACE_LEVEL=full\n"
        f"SCIAGENT_LLM_TRACE_DIR={shlex.quote(str(trace_dir))}\n",
        encoding="utf-8",
    )
    executable = tmp_path / "sciagent"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({"
        "'argv': sys.argv[1:], "
        "'level': os.environ['SCIAGENT_LLM_TRACE_LEVEL'], "
        "'directory': os.environ['SCIAGENT_LLM_TRACE_DIR']}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    env = {
        **os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
        "SCIAGENT_LLM_TRACE_LEVEL": "off", "SCIAGENT_LLM_TRACE_DIR": "/stale",
    }
    for command in ("run", "answer", "resume"):
        args = [command, "run_test", "--field", "decision=a value with spaces"]
        result = subprocess.run(
            ["bash", str(LAUNCHER), str(config), *args],
            env=env, capture_output=True, text=True, check=True,
        )
        assert json.loads(result.stdout) == {
            "argv": args, "level": "full", "directory": str(trace_dir),
        }


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash deployment entry point")
def test_missing_config_does_not_start_cli(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER), str(tmp_path / "missing.sh"), "run"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "missing.sh" in result.stderr


@pytest.fixture
def pinned_launch(tmp_path):
    """Drive the launcher with the real approval component, without model calls."""
    if sys.platform != "linux":
        pytest.skip("Linux shell approval startup")
    workspace = tmp_path / "actual workspace with spaces"
    workspace.mkdir()
    modules = tmp_path / "probe modules"
    modules.mkdir()
    (modules / "sciagent_cli.py").write_text(
        """
import json
import os
from pathlib import Path
import signal
import sys

from sciagent_components.process import ProcessRunner
from sciagent_components.workspace import WorkspaceBoundary
from sciagent_contracts import WorkspaceAccess, WorkspaceGrant

runner = ProcessRunner(WorkspaceBoundary(WorkspaceGrant(
    root=os.getcwd(), source="local",
    access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
)))
arguments = sys.argv[1:]
output = {
    "argv": arguments, "python": sys.executable, "prefix": sys.prefix, "cwd": os.getcwd(),
    "context": runner.shell_context(), "pid": os.getpid(),
    "level": os.environ["SCIAGENT_LLM_TRACE_LEVEL"],
    "directory": os.environ["SCIAGENT_LLM_TRACE_DIR"],
}
if "--approved-context" in arguments:
    approved = json.loads(Path(arguments[arguments.index("--approved-context") + 1]).read_text())
    try:
        result = runner.run_shell(
            "printf approved > accepted.txt", log_dir=str(Path.cwd() / ".probe-logs"), index=0,
            timeout_seconds=5, expected_context=approved,
        )
    except PermissionError as error:
        output["execution"] = "blocked"
        output["error"] = str(error)
        print(json.dumps(output))
        raise SystemExit(79)
    output["execution"] = "executed"
    output["execution_context"] = result.environment_fingerprint
    assert result.exit_code == 0
print(json.dumps(output), flush=True)
if "--probe-wait" in arguments:
    signal.pause()
if "--probe-exit" in arguments:
    raise SystemExit(int(arguments[arguments.index("--probe-exit") + 1]))
""",
        encoding="utf-8",
    )
    # Preserve pyvenv.cfg and site-packages when giving the interpreter a spaced path.
    # A lone executable symlink outside the environment can fall back to base Python.
    environment = tmp_path / "pinned environment"
    environment.symlink_to(sys.prefix, target_is_directory=True)
    python = environment / Path(sys.executable).relative_to(Path(sys.prefix))
    config = workspace / "deployment config.sh"
    trace_dir = workspace / "trace directory"
    config.write_text(
        "SCIAGENT_LLM_TRACE_LEVEL=full\n"
        f"SCIAGENT_LLM_TRACE_DIR={shlex.quote(str(trace_dir))}\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(modules) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "L3_RUNTIME_FACT": "original",
        "SCIAGENT_LLM_TRACE_LEVEL": "off", "SCIAGENT_LLM_TRACE_DIR": "/stale",
    }
    # This is the same explicit non-login argv documented for every L3 command.
    argv = ["/bin/bash", "--noprofile", "--norc", str(LAUNCHER),
            "--python", str(python), config.name]

    def invoke(*arguments, environment=None):
        return subprocess.run(
            [*argv, *arguments], cwd=workspace, env=environment or env,
            capture_output=True, text=True, timeout=15,
        )

    return invoke, argv, env, workspace, python, trace_dir


def test_pinned_processes_share_real_shell_approval_and_preserve_arguments(pinned_launch):
    invoke, _, _, workspace, python, trace_dir = pinned_launch
    payload = ["run_test", "--field", "decision=a value with spaces", "", "$() ; literal\nsecond line"]
    first = invoke("run", *payload)
    assert first.returncode == 0, first.stderr
    initial = json.loads(first.stdout)
    assert initial["argv"] == ["run", *payload]
    approved = workspace / "approved context.json"
    approved.write_text(json.dumps(initial["context"]), encoding="utf-8")
    for command in ("show", "answer", "resume"):
        arguments = [command, *payload]
        if command != "show":
            arguments += ["--approved-context", str(approved)]
        result = invoke(*arguments)
        assert result.returncode == 0, result.stderr
        observed = json.loads(result.stdout)
        assert observed["argv"] == arguments
        assert observed["context"] == initial["context"]
        assert observed["python"] == initial["python"] == str(python)
        assert Path(observed["prefix"]).resolve() == Path(sys.prefix).resolve()
        assert observed["cwd"] == initial["cwd"] == str(workspace)
        assert observed["level"] == initial["level"] == "full"
        assert observed["directory"] == initial["directory"] == str(trace_dir)
        if command != "show":
            assert observed["execution"] == "executed"
            assert observed["execution_context"] == initial["context"]["environment_fingerprint"]
            assert (workspace / "accepted.txt").read_text() == "approved"
            (workspace / "accepted.txt").unlink()


@pytest.mark.parametrize("command", ["answer", "resume"])
def test_changed_inherited_environment_still_blocks_old_approval(pinned_launch, command):
    invoke, _, env, workspace, _, _ = pinned_launch
    first = invoke("run")
    assert first.returncode == 0, first.stderr
    original_context = json.loads(first.stdout)["context"]
    approved = workspace / "approved context.json"
    approved.write_text(json.dumps(original_context), encoding="utf-8")

    result = invoke(command, "--approved-context", str(approved),
                    environment={**env, "L3_RUNTIME_FACT": "changed"})

    assert result.returncode == 79, result.stderr
    observed = json.loads(result.stdout)
    assert observed["execution"] == "blocked"
    assert observed["error"] == "Shell startup configuration changed after approval; script was not executed"
    assert observed["context"]["environment_fingerprint"] != original_context["environment_fingerprint"]
    assert not (workspace / "accepted.txt").exists()


@pytest.mark.parametrize("exit_code", [3, 4, 23])
def test_pinned_launcher_preserves_cli_exit_code(pinned_launch, exit_code):
    invoke, *_ = pinned_launch
    result = invoke("resume", "--probe-exit", str(exit_code))
    assert result.returncode == exit_code, result.stderr


def test_pinned_launcher_exec_preserves_pid_and_termination(pinned_launch):
    _, argv, env, workspace, _, _ = pinned_launch
    process = subprocess.Popen(
        [*argv, "resume", "--probe-wait"], cwd=workspace, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        ready, _, _ = select.select([process.stdout], [], [], 15)
        assert ready, "launcher did not reach the Python process"
        observed = json.loads(process.stdout.readline())
        assert observed["pid"] == process.pid
        process.terminate()
        process.communicate(timeout=5)
        assert process.returncode == -signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_pinned_launcher_requires_absolute_python(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux shell deployment entry point")
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", str(LAUNCHER),
         "--python", "python", str(tmp_path / "unused.sh"), "run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "absolute interpreter path" in result.stderr
