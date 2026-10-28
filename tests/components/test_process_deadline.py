"""Shared Run deadlines cancel both framework commands and their descendants."""
import os
import signal
import subprocess
from pathlib import Path
import shlex
import sys
import time

import pytest

from sciagent_components.process import ProcessRunner, run_process
from sciagent_components.workspace import WorkspaceBoundary
from sciagent_contracts import WorkspaceAccess, WorkspaceGrant
from sciagent_runtime.budget import DeadlineExceededError, execution_budget
from sciagent_runtime.cancellation import OperationCancelled


@pytest.mark.skipif(os.name != "posix", reason="process group cleanup is POSIX")
@pytest.mark.parametrize("framework", [False, True])
def test_run_deadline_terminates_child_tree(tmp_path, framework):
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)\n"
        "Path('child.pid').write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )
    boundary = WorkspaceBoundary(WorkspaceGrant(
        root=str(tmp_path), source="local",
        access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
    ))
    started = time.monotonic()
    with execution_budget(max_llm_calls=1, timeout_seconds=0.6):
        if framework:
            with pytest.raises(DeadlineExceededError):
                run_process([sys.executable, str(script)], cwd=tmp_path, timeout=30)
        else:
            result = ProcessRunner(boundary).run_shell(
                f"{shlex.quote(sys.executable)} parent.py", log_dir=".sciagent/log",
                index=1, timeout_seconds=30,
            )
            assert result.timed_out and result.exit_code != 0
    assert time.monotonic() - started < 3
    pid = int((tmp_path / "child.pid").read_text())
    stat = Path(f"/proc/{pid}/stat")
    until = time.monotonic() + 1
    while stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
        assert time.monotonic() < until, "child survived the Run deadline"
        time.sleep(0.01)


def test_second_operation_cannot_start_after_shared_deadline(tmp_path):
    clock = [0.0]
    with execution_budget(max_llm_calls=1, timeout_seconds=5, clock=lambda: clock[0]):
        run_process([sys.executable, "-c", "pass"])
        clock[0] = 5.0
        marker = tmp_path / "should-not-exist"
        with pytest.raises(DeadlineExceededError):
            run_process([sys.executable, "-c", f"from pathlib import Path;Path({str(marker)!r}).touch()"])
    assert not marker.exists()



@pytest.mark.skipif(os.name != "posix", reason="managed process trees are POSIX")
def test_shell_timeout_stops_remembered_child_after_its_parent_exits(tmp_path, monkeypatch):
    from sciagent_components import process as process_module

    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)\n"
        "Path('child.pid').write_text(str(child.pid))\n"
        "while not Path('observed').exists(): time.sleep(0.01)\n"
        "Path('parent-done').touch()\n"
    )
    boundary = WorkspaceBoundary(WorkspaceGrant(
        root=str(tmp_path), source="local",
        access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
    ))
    remember = process_module._remember_descendants

    def observe(process, identities):
        remember(process, identities)
        pidfile = tmp_path / "child.pid"
        if pidfile.exists() and int(pidfile.read_text()) in identities:
            (tmp_path / "observed").touch()

    monkeypatch.setattr(process_module, "_remember_descendants", observe)
    try:
        result = ProcessRunner(boundary).run_shell(
            f"{shlex.quote(sys.executable)} parent.py\nsleep 30", log_dir=".sciagent/log",
            index=1, timeout_seconds=1,
        )
        assert result.timed_out and result.exit_code != 0
        assert (tmp_path / "parent-done").exists()
        pid = int((tmp_path / "child.pid").read_text())
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        pidfile = tmp_path / "child.pid"
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("framework", [False, True])
def test_unconfirmed_timeout_stops_dispatch_through_cancellation_gate(tmp_path, monkeypatch, framework):
    from sciagent_components import process as process_module

    stop = process_module._stop_process

    def unconfirmed(process, identities):
        # Perform real cleanup so the fault injection never leaks a test child.
        assert stop(process, identities)
        return False

    monkeypatch.setattr(process_module, "_stop_process", unconfirmed)
    with pytest.raises(OperationCancelled) as caught:
        if framework:
            run_process([sys.executable, "-c", "import time;time.sleep(30)"], timeout=0.1)
        else:
            boundary = WorkspaceBoundary(WorkspaceGrant(
                root=str(tmp_path), source="local",
                access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
            ))
            ProcessRunner(boundary).run_shell(
                "sleep 30", log_dir=".sciagent/log", index=1, timeout_seconds=0.1,
            )
    assert not caught.value.stopped
    assert "timed out" in caught.value.reason
    assert caught.value.details["timed_out"] is True
    assert caught.value.details["files_retained"] is True
    assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
    if not framework:
        assert Path(caught.value.details["stdout_path"]).exists()
        assert Path(caught.value.details["stderr_path"]).exists()
