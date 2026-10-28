"""Exercise Esc through an actual Unix readline terminal, without a model."""

import os
import select
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX pty terminal behavior")
def test_escape_accepts_interrupt_without_enter(tmp_path):
    import pty

    script = """
from pathlib import Path
import sys
Path.home = lambda: Path(sys.argv[1])
from resagent2_cli.terminal import terminal_input
print('RESULT=' + terminal_input('ready> '), flush=True)
"""
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True,
    )
    os.close(slave)
    output = b""

    def until(expected):
        nonlocal output
        deadline = time.monotonic() + 5
        while expected not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    output += os.read(master, 4096)
                except OSError:
                    break
        assert expected in output, output.decode(errors="replace")

    try:
        until(b"ready> ")
        os.write(master, b"unsubmitted text")
        os.write(master, b"\x1b")
        until(b"RESULT=/interrupt")
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
