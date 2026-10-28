"""Real Bash execution shares the existing process budget and durable receipts."""

from pathlib import Path
import sys
import time

import pytest

from sciagent_components.process import ProcessRunner, output_tail
from sciagent_components.workspace import WorkspaceBoundary
from sciagent_contracts import WorkspaceAccess, WorkspaceGrant
from sciagent_runtime.budget import execution_budget


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="fixed Linux Bash execution")


@pytest.fixture
def runner(tmp_path):
    return ProcessRunner(WorkspaceBoundary(WorkspaceGrant(
        root=str(tmp_path), source="local",
        access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
    )))


def run_shell(runner, script, **kwargs):
    return runner.run_shell(
        script, log_dir=".sciagent/shell", index=1, timeout_seconds=5, **kwargs,
    )


@pytest.mark.parametrize("script,code,stdout", [
    ("printf 'abc\\n' | tr a-z A-Z", 0, "ABC\n"),
    ("false | cat", 1, ""),
    ("false\nprintf 'continued\\n'", 1, ""),
    ("false | cat\nprintf 'continued\\n'", 1, ""),
])
def test_shell_stops_on_unhandled_sequence_and_pipeline_failures(runner, script, code, stdout):
    result = run_shell(runner, script)
    assert result.exit_code == code and not result.timed_out
    assert Path(result.stdout_path).read_text() == stdout


@pytest.mark.parametrize("failure", ["(exit 7)", "(exit 7) | cat"])
def test_unhandled_failure_preserves_status_and_prevents_later_side_effects(runner, tmp_path, failure):
    result = run_shell(
        runner,
        "printf 'started\\n'\nprintf 'diagnostic\\n' >&2\n"
        + failure + "\nprintf 'continued\\n'\ntouch should-not-exist",
    )
    assert result.exit_code == 7 and not result.timed_out
    assert Path(result.stdout_path).read_text() == "started\n"
    assert Path(result.stderr_path).read_text() == "diagnostic\n"
    assert not (tmp_path / "should-not-exist").exists()


@pytest.mark.parametrize("script,stdout", [
    ("if false; then printf unexpected; else printf handled; fi", "handled"),
    ("false || printf recovered", "recovered"),
    ("! false\nprintf continued", "continued"),
])
def test_explicit_conditions_and_recovery_can_succeed(runner, script, stdout):
    result = run_shell(runner, script)
    assert result.exit_code == 0
    assert Path(result.stdout_path).read_text() == stdout


def test_explicit_status_capture_allows_cleanup_without_hiding_failure(runner, tmp_path):
    result = run_shell(
        runner,
        "rc=0\n(exit 7) || rc=$?\nprintf 'cleanup\\n'\ntouch cleaned\nexit \"$rc\"",
    )
    assert result.exit_code == 7
    assert Path(result.stdout_path).read_text() == "cleanup\n"
    assert (tmp_path / "cleaned").exists()


@pytest.mark.parametrize("script,stdout", [
    ("f() { false; printf continued; }; if f; then printf handled; fi", "continuedhandled"),
    ('printf "%s" "$(false)"', ""),
])
def test_bash_errexit_exceptions_are_not_reinterpreted_as_failures(runner, script, stdout):
    result = run_shell(runner, script)
    assert result.exit_code == 0
    assert Path(result.stdout_path).read_text() == stdout


def test_multiline_heredoc_and_receipt_preserve_script_bytes(runner, tmp_path):
    payload = "  unicode: \u4e2d\u6587\n$HOME $(printf changed) `printf changed`\n\ttrailing spaces  \n"
    script = "\ncat <<'PAYLOAD' > payload.txt\n" + payload + "PAYLOAD\n\n"
    result = run_shell(runner, script)
    assert result.exit_code == 0
    assert (tmp_path / "payload.txt").read_bytes() == payload.encode()
    assert result.command == script


def test_workspace_and_environment_filtering(runner, tmp_path):
    result = run_shell(
        runner,
        "printf '%s\\n' \"$PWD\" \"$SCIAGENT_TEST_VALUE\" \"$SCIAGENT_PREFIX\" "
        "\"${SCIAGENT_TEST_API_KEY-unset}\"",
        extra_env={"SCIAGENT_TEST_VALUE": "available", "SCIAGENT_PREFIX": "explicit", "SCIAGENT_TEST_API_KEY": "hidden"},
    )
    assert result.exit_code == 0
    assert Path(result.stdout_path).read_text().splitlines() == [
        str(tmp_path), "available", "explicit", "unset",
    ]


@pytest.mark.parametrize("name,first,second", [
    ("SHLVL", "1", "7"), ("_", "/first/launcher", "/second/launcher"),
    ("SSH_CLIENT", "192.0.2.1 10001 22", "192.0.2.1 10002 22"),
    ("SSH_CONNECTION", "192.0.2.1 10001 192.0.2.2 22", "192.0.2.1 10002 192.0.2.2 22"),
])
def test_approval_stays_stable_across_launcher_metadata(
    runner, monkeypatch, name, first, second,
):
    import sciagent_components.process as component
    monkeypatch.setenv(name, first)
    approved = runner.shell_context()
    monkeypatch.setenv(name, second)
    launched = []
    real_popen = component.subprocess.Popen

    def capture(*args, **kwargs):
        launched.append(kwargs["env"])
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(component.subprocess, "Popen", capture)
    result = run_shell(runner, 'printf "%s\\n" "$SHLVL"', expected_context=approved)
    assert result.exit_code == 0
    assert result.environment_fingerprint == approved["environment_fingerprint"]
    assert Path(result.stdout_path).read_text() == "1\n"
    assert "SHLVL" not in launched[0] and "_" not in launched[0]
    assert "SSH_CLIENT" not in launched[0] and "SSH_CONNECTION" not in launched[0]


def test_approval_still_rejects_changed_path_before_execution(runner, monkeypatch):
    import os
    approved = runner.shell_context()
    monkeypatch.setenv("PATH", os.environ["PATH"] + ":/different-search-path")
    with pytest.raises(PermissionError, match="configuration changed"):
        run_shell(runner, "touch should-not-exist", expected_context=approved)
    assert not (runner.boundary.root / "should-not-exist").exists()
    assert not (runner.boundary.root / ".sciagent").exists()


def test_legacy_shell_flags_are_rejected_before_launch(runner):
    approved = runner.shell_context()
    assert "-e" in approved["shell_args"]
    approved["shell_args"] = [arg for arg in approved["shell_args"] if arg != "-e"]
    with pytest.raises(PermissionError, match="configuration changed"):
        run_shell(runner, "touch should-not-exist", expected_context=approved)
    assert not (runner.boundary.root / "should-not-exist").exists()
    assert not (runner.boundary.root / ".sciagent").exists()


def test_inherited_shell_startup_and_exported_functions_are_not_executed(runner, tmp_path):
    startup = tmp_path / "startup.sh"
    startup.write_text("touch startup-executed\n")
    result = run_shell(
        runner,
        "printf 'continued\\n'\nif declare -F injected >/dev/null; then injected; fi",
        extra_env={
            "BASH_ENV": str(startup), "ENV": str(startup),
            "SHELLOPTS": "errexit:xtrace", "BASHOPTS": "nullglob",
            "CDPATH": str(tmp_path),
            "BASH_FUNC_injected%%": "() { touch function-executed; }",
        },
    )
    assert result.exit_code == 0
    assert Path(result.stdout_path).read_text() == "continued\n"
    assert Path(result.stderr_path).read_text() == ""
    assert not (tmp_path / "startup-executed").exists()
    assert not (tmp_path / "function-executed").exists()


def test_shell_obeys_shared_deadline_and_keeps_partial_logs(runner, tmp_path):
    started = time.monotonic()
    with execution_budget(max_llm_calls=1, timeout_seconds=0.4):
        result = run_shell(
            runner,
            "printf 'started\\n'; printf 'diagnostic\\n' >&2; "
            "sleep 30 & child=$!; printf '%s' \"$child\" > child.pid; wait \"$child\"",
        )
    assert result.timed_out and result.exit_code != 0
    assert time.monotonic() - started < 3
    assert output_tail(result.stdout_path) == "started\n"
    assert output_tail(result.stderr_path) == "diagnostic\n"
    child = int((tmp_path / "child.pid").read_text())
    status = Path(f"/proc/{child}/stat")
    until = time.monotonic() + 1
    while status.exists() and status.read_text().rsplit(")", 1)[1].split()[0] != "Z":
        assert time.monotonic() < until, "shell child survived the shared deadline"
        time.sleep(0.01)




def test_shell_rejects_non_linux_before_starting_process(runner, monkeypatch):
    monkeypatch.setattr("sciagent_components.process.sys.platform", "darwin")
    with pytest.raises(RuntimeError, match="only on Linux"):
        run_shell(runner, "printf never")
    assert not (runner.boundary.root / ".sciagent").exists()


@pytest.mark.parametrize("script", [" \n\t", "printf hi\x00"])
def test_shell_rejects_invalid_script_before_starting_process(runner, script):
    with pytest.raises(ValueError, match="nonempty.*NUL"):
        run_shell(runner, script)
    assert not (runner.boundary.root / ".sciagent").exists()


def test_output_tail_reads_only_bounded_suffix(tmp_path, monkeypatch):
    path = tmp_path / "large.log"
    path.write_bytes(b"x" * 100_000 + "\u4e2d\u6587: end\n".encode())
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __enter__(self):
            self.handle = original_open(path, "rb")
            return self

        def __exit__(self, *args):
            self.handle.close()

        def seek(self, *args):
            return self.handle.seek(*args)

        def tell(self):
            return self.handle.tell()

        def read(self, size=-1):
            reads.append(size)
            return self.handle.read(size)

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: BoundedReader())
    assert output_tail(path, limit=8) == "\u4e2d\u6587: end\n"
    assert reads == [32]


def test_output_tail_handles_absent_logs_and_empty_limits(tmp_path):
    assert output_tail(tmp_path / "absent.log") == ""
    assert output_tail(tmp_path, limit=0) == ""
