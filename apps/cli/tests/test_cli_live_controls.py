"""Live CLI controls submit records, never compete with the Run writer."""

import io
import queue
import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from sciagent_cli.main import cli
from sciagent_cli.shell import Shell
from sciagent_contracts import RunStatus
from sciagent_runtime.controls import RunControlInbox


def _run(status=RunStatus.RUNNING):
    return SimpleNamespace(run_id="run_live", status=status, pending_question=None)


@pytest.mark.parametrize("kind,status", [
    ("input", RunStatus.RUNNING), ("input", RunStatus.PAUSED),
    ("interrupt", RunStatus.RUNNING), ("interrupt", RunStatus.PAUSED),
])
def test_cli_controls_only_publish_records(tmp_path, kind, status):
    run = _run(status)
    text = "  完整原文\n约束 x=2；不要提前执行。  "
    def no_application(**kwargs):
        raise AssertionError("control submission must not construct model clients or a second owner")
    store = SimpleNamespace(load=lambda run_id: run)
    assert cli([kind, run.run_id, "--text", text, "--data-root", str(tmp_path)],
               application_builder=no_application, store_factory=lambda root: store) == 0
    records = RunControlInbox(tmp_path / "controls").pending(run.run_id)
    assert len(records) == 1
    assert records[0].text == text and records[0].kind == kind
    assert run.status == status and run.pending_question is None


def test_cli_input_file_keeps_text_and_pending_question(tmp_path):
    run = _run(RunStatus.PAUSED)
    question = SimpleNamespace(id="question_original")
    run.pending_question = question
    path = tmp_path / "supplement.txt"
    path.write_text("补充目标\n保留原始要求", encoding="utf-8")
    assert cli(["input", run.run_id, "--file", str(path), "--data-root", str(tmp_path)],
               store_factory=lambda root: SimpleNamespace(load=lambda run_id: run)) == 0
    assert RunControlInbox(tmp_path / "controls").pending(run.run_id)[0].text == path.read_text()
    assert run.pending_question is question


@pytest.mark.parametrize("status", [RunStatus.COMPLETED, RunStatus.FAILED])
def test_terminal_run_rejects_live_input(tmp_path, status):
    with pytest.raises(ValueError, match="running or paused"):
        cli(["input", "run_live", "--text", "supplement", "--data-root", str(tmp_path)],
            store_factory=lambda root: SimpleNamespace(load=lambda run_id: _run(status)))
    assert not (tmp_path / "controls").exists()


def test_watch_accepts_input_and_waits_for_owner_after_interrupt(tmp_path, monkeypatch):
    run = _run()
    inbox = RunControlInbox(tmp_path / "controls")
    stopped = threading.Event()
    shell = Shell(data_root=tmp_path, application_builder=lambda **kwargs: None,
                  store=SimpleNamespace(load=lambda run_id: run), stream=io.StringIO())
    shell.current_run_id = run.run_id
    shell._input_queue = queue.Queue()
    shell._input_queue.put("  preserve these user words  ")
    shell._input_queue.put("/interrupt")
    monkeypatch.setattr("sciagent_cli.render.render_live", lambda *a: [])
    monkeypatch.setattr("sciagent_cli.render.render_final", lambda *a: [])

    def owner():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if any(record.kind == "interrupt" for record in inbox.pending(run.run_id)):
                time.sleep(0.02)
                stopped.set()
                return _run(RunStatus.PAUSED)
            time.sleep(0.002)
        raise AssertionError("shell did not publish interruption")

    shell.runner.start(owner)
    result = shell._watch(run.run_id, poll_interval=0.002)
    assert stopped.is_set() and result.status == RunStatus.PAUSED
    records = inbox.pending(run.run_id)
    assert [(record.kind, record.text) for record in records] == [
        ("input", "  preserve these user words  "), ("interrupt", ""),
    ]
    assert not shell.runner.active
    assert "current work is not interrupted" in shell.stream.getvalue()


def test_quit_cannot_abandon_running_local_owner(tmp_path):
    shell = Shell(data_root=tmp_path, application_builder=lambda **kwargs: None,
                  store=SimpleNamespace(), stream=io.StringIO())
    shell.runner = SimpleNamespace(active=True)
    with pytest.raises(ValueError, match="confirmed stop"):
        shell._dispatch("/quit")


def test_one_shot_ctrl_c_waits_for_owner_confirmation(tmp_path, monkeypatch):
    from sciagent_cli.main import _owner_call
    from sciagent_cli.shell import Runner

    inbox = RunControlInbox(tmp_path / "controls")
    stopped = threading.Event()
    actual_wait = Runner.wait
    first_wait = True

    def interrupt_once(self, timeout=0.1):
        nonlocal first_wait
        if first_wait:
            first_wait = False
            raise KeyboardInterrupt
        return actual_wait(self, timeout)

    monkeypatch.setattr(Runner, "wait", interrupt_once)

    def owner():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if inbox.pending("run_live"):
                time.sleep(0.02)
                stopped.set()
                return _run(RunStatus.PAUSED)
            time.sleep(0.002)
        raise AssertionError("Ctrl-C did not deliver interruption to owner")

    result = _owner_call(tmp_path, "run_live", owner)
    assert stopped.is_set() and result.status == RunStatus.PAUSED
    assert [record.kind for record in inbox.pending("run_live")] == ["interrupt"]


def test_interrupt_supplement_resumes_only_after_confirmed_stop(tmp_path, monkeypatch):
    run = _run()
    stopped = threading.Event()
    resumed = []
    paused = _run(RunStatus.PAUSED)
    completed = _run(RunStatus.COMPLETED)
    completed.workspaces = {}
    paused.workspaces = {}
    def resume(run_id):
        assert stopped.is_set()
        resumed.append(run_id)
        return completed
    shell = Shell(data_root=tmp_path,
        application_builder=lambda **kwargs: SimpleNamespace(controller=SimpleNamespace(resume_run=resume)),
        store=SimpleNamespace(load=lambda run_id: run), stream=io.StringIO())
    shell.current_run_id = run.run_id
    monkeypatch.setattr("sciagent_cli.render.render_live", lambda *a: [])
    monkeypatch.setattr("sciagent_cli.render.render_final", lambda *a: [])
    def owner():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if RunControlInbox(tmp_path / "controls").pending(run.run_id):
                stopped.set()
                return paused
            time.sleep(0.002)
        raise AssertionError("no interruption")
    shell.runner.start(owner)
    shell._dispatch("/interrupt Revised goal")
    assert shell._watch(run.run_id, poll_interval=0.002) is completed
    assert resumed == [run.run_id]
    assert RunControlInbox(tmp_path / "controls").pending(run.run_id)[0].text == "Revised goal"


def test_cli_revalidates_stale_running_snapshot_before_publication(tmp_path):
    statuses = iter([_run(RunStatus.RUNNING), _run(RunStatus.COMPLETED)])
    store = SimpleNamespace(load=lambda run_id: next(statuses))
    with pytest.raises(ValueError, match="running or paused"):
        cli(["input", "run_live", "--text", "late supplement", "--data-root", str(tmp_path)],
            store_factory=lambda root: store)
    assert not RunControlInbox(tmp_path / "controls").pending("run_live")


def test_ctrl_c_after_terminal_commit_does_not_publish_stale_interrupt(tmp_path, monkeypatch):
    from sciagent_cli.main import _owner_call
    from sciagent_cli.shell import Runner

    committed = _run(RunStatus.COMPLETED)
    finish = threading.Event()
    actual_wait = Runner.wait
    first_wait = True
    import importlib
    monkeypatch.setattr(importlib.import_module("sciagent_cli.main"), "_run_store", lambda root: SimpleNamespace(
        exists=lambda run_id: True, load=lambda run_id: committed))
    def interrupt_after_commit(self, timeout=0.1):
        nonlocal first_wait
        if first_wait:
            first_wait = False
            finish.set()
            raise KeyboardInterrupt
        return actual_wait(self, timeout)
    monkeypatch.setattr(Runner, "wait", interrupt_after_commit)
    def owner():
        assert finish.wait(2)
        return committed
    result = _owner_call(tmp_path, committed.run_id, owner)
    assert result.status == RunStatus.COMPLETED
    assert not RunControlInbox(tmp_path / "controls").pending(committed.run_id)


@pytest.mark.parametrize("line", ["/quit", None])
def test_attach_exit_does_not_interrupt_external_owner(tmp_path, monkeypatch, line):
    run = _run()
    shell = Shell(data_root=tmp_path, application_builder=lambda **kwargs: None,
                  store=SimpleNamespace(load=lambda run_id: run), stream=io.StringIO())
    shell._input_queue = queue.Queue()
    shell._input_queue.put(line)
    monkeypatch.setattr("sciagent_cli.render.render_live", lambda *a: [])
    monkeypatch.setattr("sciagent_cli.shell.time.sleep", lambda _: pytest.fail(
        "an observer must exit without waiting for the external Run to finish"))

    assert shell._cmd_attach([run.run_id]) is True

    assert shell._exit_requested
    assert run.status == RunStatus.RUNNING
    assert not RunControlInbox(tmp_path / "controls").pending(run.run_id)
    assert not shell.runner.active
