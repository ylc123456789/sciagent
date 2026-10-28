"""Explicit interruption stops foreground process trees and retains partial files."""
import os
import signal
from pathlib import Path
import shlex
import sys
from threading import Thread
import time
from types import SimpleNamespace

import pytest

from resagent2_components.process import ProcessRunner, run_process
from resagent2_components.workspace import WorkspaceBoundary
from resagent2_contracts import WorkspaceAccess, WorkspaceGrant
from resagent2_runtime.cancellation import CancellationToken, OperationCancelled, bind_cancellation, check_cancelled


@pytest.mark.skipif(os.name != "posix", reason="managed process trees are POSIX")
@pytest.mark.parametrize("framework", [False, True])
def test_explicit_interrupt_stops_detached_descendant_and_keeps_file(tmp_path, framework):
    token = CancellationToken()
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)\n"
        "Path('child.pid').write_text(str(child.pid))\n"
        "Path('partial-checkpoint').write_text('partial')\n"
        "print('training started',flush=True)\ntime.sleep(30)\n"
    )
    boundary = WorkspaceBoundary(WorkspaceGrant(
        root=str(tmp_path), source="local",
        access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
    ))

    def interrupt():
        until = time.monotonic() + 3
        while not (tmp_path / "partial-checkpoint").exists():
            assert time.monotonic() < until
            time.sleep(0.01)
        token.cancel("user pause")

    worker = Thread(target=interrupt)
    worker.start()
    try:
        with bind_cancellation(token), pytest.raises(OperationCancelled) as caught:
            if framework:
                run_process([sys.executable, str(script)], cwd=tmp_path, timeout=30)
            else:
                ProcessRunner(boundary).run_shell(
                    f"{shlex.quote(sys.executable)} parent.py", log_dir=".resagent2/log",
                    index=1, timeout_seconds=30,
                )
        assert caught.value.stopped
        assert caught.value.details["exit_code"] != 0
        assert caught.value.details["files_retained"]
        assert (tmp_path / "partial-checkpoint").read_text() == "partial"
        pid = int((tmp_path / "child.pid").read_text())
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        if not framework:
            assert "training started" in Path(caught.value.details["stdout_path"]).read_text()
    finally:
        worker.join(3)


def test_cancel_before_process_does_not_create_file(tmp_path):
    token = CancellationToken()
    token.cancel()
    marker = tmp_path / "never"
    with bind_cancellation(token), pytest.raises(OperationCancelled):
        run_process([sys.executable, "-c", f"from pathlib import Path;Path({str(marker)!r}).touch()"])
    assert not marker.exists()






@pytest.mark.skipif(os.name != "posix", reason="managed process trees are POSIX")
def test_interrupt_stops_pipe_writer_after_parent_exits_before_any_scan(tmp_path, monkeypatch):
    from resagent2_components import process as process_module

    # Force the hard boundary: the short-lived parent exits before polling can
    # discover its detached child. Its unique inherited output pipe still owns
    # that child even after reparenting.
    monkeypatch.setattr(process_module, "_remember_descendants", lambda *args: None)
    token = CancellationToken()
    pidfile = tmp_path / "child.pid"
    parentfile = tmp_path / "parent.pid"
    child_code = (
        "import os,time;from pathlib import Path;"
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()));"
        "print('detached child started',flush=True);time.sleep(30)"
    )
    parent_code = (
        "import os,subprocess,sys;from pathlib import Path;"
        f"Path({str(parentfile)!r}).write_text(str(os.getpid()));"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True)"
    )

    def interrupt_after_parent_reaped():
        until = time.monotonic() + 3
        while not pidfile.exists() or not parentfile.exists():
            assert time.monotonic() < until
            time.sleep(0.01)
        parent = int(parentfile.read_text())
        while Path(f"/proc/{parent}").exists():
            assert time.monotonic() < until
            time.sleep(0.01)
        token.cancel("interrupt after parent exit")

    worker = Thread(target=interrupt_after_parent_reaped)
    worker.start()
    try:
        with bind_cancellation(token), pytest.raises(OperationCancelled) as caught:
            run_process([sys.executable, "-c", parent_code], timeout=10)
        assert caught.value.stopped
        assert caught.value.details["exit_code"] == 0  # Parent really completed.
        assert "detached child started" in caught.value.details["stdout_tail"]
        assert caught.value.details["unconfirmed_processes"] == []
        child = int(pidfile.read_text())
        stat = Path(f"/proc/{child}/stat")
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        worker.join(3)
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cleanup_pipe_timeout_preserves_cancel_and_unconfirmed_stop(monkeypatch):
    import subprocess
    from io import BytesIO
    from resagent2_components import process as process_module

    token = CancellationToken()

    class UnclosedPipes:
        pid = 999999999
        returncode = 0
        stdout = BytesIO()
        stderr = BytesIO()
        args = ["framework-command"]
        calls = 0

        def poll(self):
            return self.returncode

        def communicate(self, timeout):
            self.calls += 1
            if self.calls == 1:
                token.cancel("stop command")
            raise subprocess.TimeoutExpired(self.args, timeout, output=b"partial output")

    process = UnclosedPipes()
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(process_module, "_stop_process", lambda *args: True)
    with bind_cancellation(token), pytest.raises(OperationCancelled) as caught:
        run_process(process.args, timeout=10)
    assert not caught.value.stopped
    assert caught.value.reason == "stop command"
    assert caught.value.details["cleanup_error"] == "command output pipes did not close after cleanup"
    assert caught.value.details["stdout_tail"] == "partial output"


@pytest.mark.skipif(os.name != "posix", reason="Linux pipe ownership")
def test_pipe_cleanup_only_owns_writers_and_not_unrelated_reader():
    import subprocess
    from resagent2_components import process as process_module

    read_fd, write_fd = os.pipe()
    reader = writer = None
    stream = os.fdopen(read_fd, "rb")
    try:
        reader = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            pass_fds=(read_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        writer = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        identities = {}
        process_module._remember_pipe_owners(SimpleNamespace(stdout=stream, stderr=None), identities)
        assert writer.pid in identities
        assert reader.pid not in identities
        assert os.getpid() not in identities
    finally:
        stream.close()
        os.close(write_fd)
        for child in (reader, writer):
            if child is not None:
                child.kill()
                child.wait(timeout=3)


def test_reused_descendant_pid_is_not_signalled(monkeypatch):
    from resagent2_components import process as process_module

    class CompletedParent:
        pid = 999999998
        stdout = stderr = None

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(process_module, "_remember_descendants", lambda *args: None)
    monkeypatch.setattr(process_module, "_process_identity", lambda pid: "new_start_time")
    monkeypatch.setattr(process_module.os, "kill", lambda *args: pytest.fail("signalled reused PID"))
    monkeypatch.setattr(process_module.os, "killpg", lambda *args: pytest.fail("signalled reused group"))
    assert process_module._stop_process(CompletedParent(), {999999999: "old_start_time"})



def test_reaped_parent_pid_does_not_discover_new_unrelated_descendants(monkeypatch):
    from resagent2_components import process as process_module

    monkeypatch.setattr(process_module, "_descendant_pids", lambda pid: pytest.fail("scanned reaped PID"))
    identities = {123: "previously_seen_identity"}
    process_module._remember_descendants(SimpleNamespace(pid=999999999, returncode=0), identities)
    assert identities == {123: "previously_seen_identity"}
