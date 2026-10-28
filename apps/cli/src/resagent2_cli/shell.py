"""Interactive shell over the production Controller and durable control inbox.

The Run owner remains the only writer of Run/Session snapshots. Live user text
and explicit interruption publish small immutable records while the shell
watches persisted progress. Ctrl-C requests interruption; /attach observes an
existing owner, and /resume explicitly starts one for a paused Run.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from resagent2_components.literature import load_literature_manifest
from resagent2_contracts import RunStatus

from . import render
from .composition import build_application
from .terminal import terminal_input
from .main import (
    EXIT_COMPLETED,
    _answer_from_args,
    _default_data_root,
    _new_run_id,
    _parser,
    _request_from_args,
    _run_store,
    _specs_for_existing_run,
    _submit_control,
    _workspace_specs,
)

TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.PAUSED}
POLL_INTERVAL = 0.25


class _NoExitParser(argparse.ArgumentParser):
    """Argparse parser that raises instead of killing the shell process.

    The one-shot CLI keeps the stock ``sys.exit`` behavior; the shell uses this
    so a malformed ``/run`` returns to the prompt instead of terminating it.
    """

    def error(self, message: str) -> None:
        raise ValueError(message)


class Runner:
    """Run one blocking controller method on a daemon thread."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._result = None
        self._error = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def start(self, fn: Callable[[], object]) -> None:
        if self.active:
            raise RuntimeError("a run is already active; wait for it to finish")
        self._done.clear()
        with self._lock:
            self._result = None
            self._error = None

        def _target() -> None:
            try:
                result = fn()
            except Exception as error:  # noqa: BLE001
                with self._lock:
                    self._error = error
            else:
                with self._lock:
                    self._result = result
            finally:
                self._done.set()

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()

    def wait(self, timeout: float = 0.1) -> bool:
        return self._done.wait(timeout)

    def outcome(self) -> tuple[object, Exception | None]:
        with self._lock:
            return self._result, self._error


class Display:
    """Compact progress blocks; terminal input remains on its own editable line."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self._live_lines = 0
        self._last_non_tty_lines: list[str] | None = None

    def _ansi(self) -> bool:
        return self.stream.isatty()

    def render(self, lines: list[str]) -> None:
        ansi = self._ansi()
        if lines == self._last_non_tty_lines:
            return
        if ansi and self._live_lines:
            self.stream.write(f"\x1b[{self._live_lines}A")
            self.stream.write("\x1b[J")
        for line in lines:
            self.stream.write(line + "\n")
        self.stream.flush()
        self._live_lines = len(lines) if ansi else 0
        self._last_non_tty_lines = list(lines)

    def clear(self) -> None:
        if self._ansi() and self._live_lines:
            self.stream.write(f"\x1b[{self._live_lines}A")
            self.stream.write("\x1b[J")
            self.stream.flush()
        self._live_lines = 0
        self._last_non_tty_lines = None


class TraceTail:
    """Incrementally read the append-only JSONL LLM trace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0

    def reset(self) -> None:
        self.offset = self.path.stat().st_size if self.path.exists() else 0

    def new_records(self, run_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
        if size <= self.offset:
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            raw = handle.read()
        # The writer may still be appending JSON or a UTF-8 character. Consume
        # complete JSONL records only, keeping the byte offset before that tail.
        complete_end = raw.rfind(b"\n") + 1
        if not complete_end:
            return []
        complete = raw[:complete_end]
        self.offset += complete_end
        records: list[dict] = []
        for line in complete.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if run_id is not None and record.get("run_id") != run_id:
                continue
            records.append(record)
        return records


def _reject_shell_data_root(tokens: list[str]) -> None:
    """Keep one interactive session bound to its startup data root."""
    if any(token == "--data-root" or token.startswith("--data-root=") for token in tokens):
        raise ValueError(
            "the shell data root is fixed at startup; exit and restart with "
            "resagent2 shell --data-root PATH"
        )


class Shell:
    """REPL loop with slash commands over the production controller."""

    def __init__(
        self,
        *,
        data_root: Path,
        application_builder: Callable[..., object],
        store,
        stream=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.application_builder = application_builder
        self.store = store
        self.stream = stream or sys.stdout
        self.runner = Runner()
        self.current_run_id: str | None = None
        self.display = Display(self.stream)
        self._input_queue: queue.Queue[str | None] | None = None
        self._exit_requested = False
        self._resume_after_interrupt = False
        self._reader_stop = threading.Event()

    # -- entry ---------------------------------------------------------------

    def run(self) -> int:
        _setup_readline()
        self.stream.write("ResAgent2 shell. Type /help for commands, /quit to exit.\n")
        self._start_input_reader()
        while True:
            try:
                raw = self._read_input()
            except EOFError:
                self._reader_stop.set()
                self.stream.write("\n")
                return EXIT_COMPLETED
            except KeyboardInterrupt:
                self.stream.write("\n")
                continue
            if not raw.strip():
                continue
            try:
                if self._dispatch(raw):
                    return EXIT_COMPLETED
            except KeyboardInterrupt:
                self.display.clear()
                if self.current_run_id is not None:
                    self._cmd_interrupt([])
                    if getattr(self.runner, "active", False):
                        self._watch(self.current_run_id)
                else:
                    self.stream.write("No active Run to interrupt.\n")
            except SystemExit:
                # argparse --help on a slash command; help was already printed.
                continue
            except Exception as error:  # noqa: BLE001
                self.display.clear()
                self.stream.write(f"error: {error}\n")
        return EXIT_COMPLETED

    def _dispatch(self, line: str) -> bool:
        if not line.startswith("/"):
            run = self._require_current_run("user input")
            record = _submit_control(self.data_root, run, kind="input", text=line, store=self.store)
            self.stream.write(f"Input saved: {record.event_id}; current work is not interrupted.\n")
            if run.status == RunStatus.PAUSED:
                self.stream.write("Use /resume to advance; this is not an /answer.\n")
            return False
        if line == "/interrupt" or line.startswith("/interrupt "):
            self._request_interrupt(line[len("/interrupt"):].lstrip())
            return False
        try:
            tokens = shlex.split(line)
        except ValueError as error:
            raise ValueError(f"could not parse command: {error}") from error
        command = tokens[0][1:]
        rest = tokens[1:]
        if command == "help":
            self.stream.write(
                "/run --goal … [--workspace …]   start a new research Run\n"
                "/show <run_id>                  show a persisted Run\n"
                "/attach <run_id>                watch a persisted Run (read-only)\n"
                "/resume <run_id>                continue a paused/running Run\n"
                "/answer [value|name=value]      answer the pending question; no value prompts\n"
                "        [--comment TEXT]        optional operation approval explanation\n"
                "ordinary text                  queue a business supplement; does not interrupt\n"
                "/interrupt [text]               stop current work, optionally add a supplement\n"
                "/artifacts                      list the current Run's artifacts\n"
                "/trace [run_id]                 show raw LLM trace (full level)\n"
                "/quit                           exit the shell\n"
            )
            return False
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            raise ValueError(f"unknown command /{command}")
        _reject_shell_data_root(rest)
        return handler(rest)

    # -- command handlers ----------------------------------------------------

    def _cmd_quit(self, rest: list[str]) -> bool:
        if getattr(self.runner, "active", False):
            raise ValueError("a Run is active; use /interrupt and wait for the confirmed stop before /quit")
        self._reader_stop.set()
        return True

    def _cmd_run(self, rest: list[str]) -> bool:
        self._require_idle_runner()
        args = _parser(_NoExitParser).parse_args(["run"] + rest)
        run_id = args.run_id or _new_run_id()
        request = _request_from_args(args)
        workspaces = _workspace_specs(args)
        prepared = load_literature_manifest(args.literature_file) if args.literature_file else None
        literature_kwargs = {"literature": prepared} if prepared is not None else {}
        application = self.application_builder(
            data_root=self.data_root,
            workspaces=workspaces,
        )
        self.current_run_id = run_id
        self.runner.start(
            lambda: application.controller.create_run(run_id, request, **literature_kwargs)
        )
        self._watch(run_id)
        return self._exit_requested

    def _cmd_resume(self, rest: list[str]) -> bool:
        self._require_idle_runner()
        args = _parser(_NoExitParser).parse_args(["resume"] + rest)
        run_id = args.run_id
        existing = self.store.load(run_id)
        application = self.application_builder(
            data_root=self.data_root,
            workspaces=_specs_for_existing_run(args, existing),
        )
        self.current_run_id = run_id
        self.runner.start(
            lambda: application.controller.resume_run(run_id)
        )
        self._watch(run_id)
        return self._exit_requested

    def _cmd_show(self, rest: list[str]) -> bool:
        run_id = self._require_run_id(rest, "/show")
        run = self.store.load(run_id)
        if not getattr(self.runner, "active", False):
            self.current_run_id = run_id
        for line in render.render_final(run):
            self.stream.write(line + "\n")
        return False

    def _cmd_attach(self, rest: list[str]) -> bool:
        self._require_idle_runner()
        run_id = self._require_run_id(rest, "/attach")
        self.store.load(run_id)
        self.current_run_id = run_id
        self._watch(run_id, wait_for_runner=False)
        return self._exit_requested

    def _cmd_answer(self, rest: list[str]) -> bool:
        self._require_idle_runner()
        run = self._require_current_run("/answer")
        args = _parser(_NoExitParser).parse_args(["answer", run.run_id] + rest)
        if not args.values and not args.field:
            self._prompt_answer(args, run)
        answer = _answer_from_args(args, run)
        application = self.application_builder(
            data_root=self.data_root,
            workspaces=_specs_for_existing_run(args, run),
        )
        self.runner.start(
            lambda: application.controller.answer_question(run.run_id, answer)
        )
        self._watch(run.run_id)
        return self._exit_requested

    def _cmd_artifacts(self, rest: list[str]) -> bool:
        run = self._require_current_run("/artifacts")
        for line in render.render_artifacts(run):
            self.stream.write(line + "\n")
        return False

    def _cmd_trace(self, rest: list[str]) -> bool:
        if len(rest) > 1:
            raise ValueError("/trace accepts at most one run_id")
        run_id = rest[0] if rest else self.current_run_id
        for line in render.render_trace(self._read_trace(run_id)):
            self.stream.write(line + "\n")
        return False

    def _cmd_interrupt(self, rest: list[str]) -> bool:
        self._request_interrupt(" ".join(rest))
        return False

    def _request_interrupt(self, text: str = "") -> None:
        run = self._require_current_run("/interrupt")
        record = _submit_control(self.data_root, run, kind="interrupt", text=text, store=self.store)
        self._resume_after_interrupt = bool(text.strip())
        self.stream.write(f"Interruption requested: {record.event_id}; waiting for the Run owner to confirm the stop.\n")
        if run.status == RunStatus.PAUSED and not getattr(self.runner, "active", False):
            # A paused question has no running owner. Start one only to settle
            # this queued interruption; pure interruption still returns paused.
            self._start_existing_owner(run)
            self._watch(run.run_id)

    def _start_existing_owner(self, run) -> None:
        args = _parser(_NoExitParser).parse_args(["resume", run.run_id])
        application = self.application_builder(
            data_root=self.data_root, workspaces=_specs_for_existing_run(args, run),
        )
        self.runner.start(lambda: application.controller.resume_run(run.run_id))

    def _continue_after_confirmed_stop(self, run):
        resume = self._resume_after_interrupt
        self._resume_after_interrupt = False
        if resume and run.status == RunStatus.PAUSED:
            if run.pending_question is not None:
                self.stream.write("Stop confirmed; answer the retained user question before continuing.\n")
            else:
                self._start_existing_owner(run)
                return self._watch(run.run_id)
        return run

    def _require_idle_runner(self) -> None:
        if getattr(self.runner, "active", False):
            raise ValueError("the Run owner is active; use ordinary input or /interrupt, then wait before resuming or answering")

    def _start_input_reader(self) -> None:
        self._input_queue = queue.Queue(maxsize=1)
        self._reader_stop.clear()
        read_input = terminal_input if sys.stdin.isatty() else input

        def publish(line: str | None) -> bool:
            while not self._reader_stop.is_set():
                try:
                    self._input_queue.put(line, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def read_lines() -> None:
            while not self._reader_stop.is_set():
                try:
                    line = read_input("resagent2> ")
                except (EOFError, StopIteration, OSError):
                    publish(None)
                    return
                if not publish(line):
                    return

        threading.Thread(target=read_lines, daemon=True).start()

    def _read_input(self, prompt: str = "") -> str:
        if self._input_queue is None:
            return input(prompt)
        if prompt:
            self.stream.write(prompt)
            self.stream.flush()
        line = self._input_queue.get()
        if line is None:
            raise EOFError
        return line

    # -- helpers -------------------------------------------------------------

    def _prompt_answer(self, args: argparse.Namespace, run) -> None:
        question = run.pending_question
        if question is None:
            raise ValueError(f"Run {run.run_id} has no pending question")
        for line in render.render_question(question):
            self.stream.write(line + "\n")
        if question.action is None and args.comment is not None:
            raise ValueError("--comment is for operation approvals; answer user questions through their requested fields")
        for field in question.requested_fields:
            while True:
                value = self._read_input(f"{field}> ")
                if value == "/interrupt":
                    raise KeyboardInterrupt
                if question.action is not None and value.strip().lower() not in {"yes", "no"}:
                    self.stream.write("Enter yes or no; add any explanation in the next prompt.\n")
                elif not value.strip():
                    self.stream.write("Enter an option or your own text.\n")
                else:
                    args.field.append(f"{field}={value}")
                    break
        if question.action is not None and args.comment is None:
            comment = self._read_input("Explanation (optional; Enter to skip)> ")
            if comment == "/interrupt":
                raise KeyboardInterrupt
            args.comment = comment or None

    def _require_run_id(self, rest: list[str], command: str) -> str:
        if len(rest) != 1:
            raise ValueError(f"{command} needs exactly one run_id")
        return rest[0]

    def _require_current_run(self, command: str):
        if self.current_run_id is None:
            raise ValueError(
                f"{command} needs a current Run; "
                "use /run, /show, /attach, or /resume first"
            )
        return self.store.load(self.current_run_id)

    def _trace_path(self) -> Path | None:
        value = os.environ.get("RESAGENT2_LLM_TRACE_DIR")
        if not value:
            return None
        return Path(value).expanduser() / "llm_traces.jsonl"

    def _read_trace(self, run_id: str | None) -> list[dict]:
        path = self._trace_path()
        if path is None:
            raise ValueError(
                "trace not enabled: set RESAGENT2_LLM_TRACE_DIR "
                "(and RESAGENT2_LLM_TRACE_LEVEL=full)"
            )
        if not path.exists():
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id is not None and record.get("run_id") != run_id:
                continue
            records.append(record)
        return records

    def _watch(
        self,
        run_id: str,
        *,
        poll_interval: float = POLL_INTERVAL,
        wait_for_runner: bool = True,
    ):
        path = self._trace_path()
        tail = TraceTail(path) if path is not None else None
        if tail is not None:
            tail.reset()
        while True:
            try:
                run = self.store.load(run_id)
            except Exception:  # noqa: BLE001
                run = None
            records = tail.new_records(run_id) if tail is not None else []
            self.display.render(render.render_live(run, records))
            if self._input_queue is not None:
                try:
                    line = self._input_queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self.display.clear()
                    if line is None:
                        self._exit_requested = True
                        if wait_for_runner and run is not None and run.status == RunStatus.RUNNING:
                            self._request_interrupt()
                    elif line.strip():
                        try:
                            if self._dispatch(line):
                                self._exit_requested = True
                        except Exception as error:  # noqa: BLE001
                            self.stream.write(f"error: {error}\n")
            if self._exit_requested and not wait_for_runner:
                self.display.clear()
                return run
            if wait_for_runner and self.runner.done:
                result, error = self.runner.outcome()
                self.display.clear()
                if error is not None:
                    raise RuntimeError(f"run failed: {error}") from error
                if result is not None:
                    for line in render.render_final(result):
                        self.stream.write(line + "\n")
                return self._continue_after_confirmed_stop(result) if result is not None else None
            if not wait_for_runner and run is not None and run.status in TERMINAL_STATUSES:
                self.display.clear()
                for line in render.render_final(run):
                    self.stream.write(line + "\n")
                return self._continue_after_confirmed_stop(run)
            time.sleep(poll_interval)


def _setup_readline() -> None:
    """Enable line editing and a best-effort history file (Linux only)."""
    if not sys.stdin.isatty():
        return
    try:
        import readline  # noqa: F401
    except ImportError:
        return
    history = Path.home() / ".resagent2_history"
    try:
        readline.read_history_file(str(history))
    except (OSError, FileNotFoundError):
        pass

    def _save() -> None:
        try:
            readline.write_history_file(str(history))
        except OSError:
            pass

    import atexit

    atexit.register(_save)


def run_shell(
    *,
    data_root: str | Path | None = None,
    application_builder: Callable[..., object] = build_application,
    store_factory: Callable[..., object] = _run_store,
    stream=None,
) -> int:
    root = Path(data_root or _default_data_root()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = store_factory(root)
    shell = Shell(
        data_root=root,
        application_builder=application_builder,
        store=store,
        stream=stream,
    )
    return shell.run()
