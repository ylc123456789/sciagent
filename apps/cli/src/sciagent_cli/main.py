"""Minimal command-line adapter over the existing sciagent interfaces."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from sciagent_contracts import (
    ExecutionLimits,
    ResearchRequest,
    RunBudget,
    RunPermissions,
    RunStatus,
    UserAnswer,
    WorkspaceSourceKind,
    WorkspaceSpec,
    WorkspaceAccess,
)
from sciagent_components.literature import load_literature_manifest
from sciagent_orchestrator import JsonRunStore, ResearchRun

from sciagent_runtime.controls import RunControlInbox

from . import render
from .composition import CliApplication, build_application


EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_PAUSED = 3
EXIT_RUNNING = 4

ApplicationBuilder = Callable[..., CliApplication]
StoreFactory = Callable[[Path], JsonRunStore]


def _default_data_root() -> str:
    return os.environ.get("SCIAGENT_DATA_ROOT", ".sciagent/data")


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid4().hex[:8]}"


def _assignment(value: str, *, label: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{label} must use NAME=VALUE syntax")
    name, assigned = value.split("=", 1)
    if not name.strip() or not assigned.strip():
        raise ValueError(f"{label} must contain a non-empty name and value")
    return name.strip(), assigned


def _goal(args: argparse.Namespace) -> str:
    if args.goal is not None:
        return args.goal
    return Path(args.goal_file).expanduser().read_text(encoding="utf-8")


def _workspace_args(parser: argparse.ArgumentParser) -> None:
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--workspace", help="existing local workspace directory")
    sources.add_argument("--git", help="Git repository URL to clone into the Run")
    parser.add_argument("--read-path", action="append", help="readable workspace prefix; default: .")
    writes = parser.add_mutually_exclusive_group()
    writes.add_argument("--write-path", action="append", help="writable workspace prefix; default: .")
    writes.add_argument("--read-only", action="store_true", help="grant no source file writes")
    parser.add_argument("--deny-path", action="append", default=[], help="workspace prefix excluded from reads and writes")


def _workspace_specs(args: argparse.Namespace) -> dict[str, WorkspaceSpec]:
    if not args.workspace and not args.git:
        if (args.read_path is not None or args.write_path is not None
                or args.read_only or args.deny_path):
            raise ValueError("workspace options require --workspace or --git")
        return {}
    access = WorkspaceAccess(
        read_paths=args.read_path if args.read_path is not None else ["."],
        write_paths=[] if args.read_only else (args.write_path if args.write_path is not None else ["."]),
        denied_paths=args.deny_path,
    )
    if args.workspace:
        root = Path(args.workspace).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"workspace is not a directory: {root}")
        return {
            "ws_main": WorkspaceSpec(
                workspace_id="ws_main",
                source_kind=WorkspaceSourceKind.LOCAL,
                location=str(root),
                access=access,
            )
        }
    if args.git:
        return {
            "ws_main": WorkspaceSpec(
                workspace_id="ws_main",
                source_kind=WorkspaceSourceKind.GIT,
                location=args.git,
                access=access,
            )
        }


def _specs_for_existing_run(
    args: argparse.Namespace,
    run: ResearchRun,
) -> dict[str, WorkspaceSpec]:
    supplied = _workspace_specs(args)
    persisted = {
        workspace_id: record.source
        for workspace_id, record in run.workspaces.items()
    }
    if supplied:
        if not persisted:
            raise ValueError("cannot add a workspace to an existing Run; create a new Run")
        if set(supplied) != set(persisted):
            raise ValueError("supplied workspace does not match the persisted Run workspace")
        for workspace_id, proposed in supplied.items():
            existing = persisted[workspace_id]
            if (
                proposed.source_kind != existing.source_kind
                or proposed.location != existing.location
                or proposed.access != existing.access
            ):
                raise ValueError(
                    "supplied workspace does not match the persisted Run workspace"
                )
    return persisted


def _answer_from_args(args: argparse.Namespace, run: ResearchRun) -> UserAnswer:
    question = run.pending_question
    if question is None:
        raise ValueError(f"Run {run.run_id} has no pending question")
    fields = args.field + args.values
    if not fields:
        raise ValueError("answer requires a value or NAME=VALUE fields")
    if len(fields) == 1 and "=" not in fields[0]:
        if len(question.requested_fields) != 1:
            raise ValueError(
                "single-value shorthand needs exactly one requested field; use NAME=VALUE"
            )
        values = {question.requested_fields[0]: fields[0]}
    else:
        values = dict(_assignment(value, label="answer field") for value in fields)
    if question.action is None:
        if args.comment is not None:
            raise ValueError("--comment is for operation approvals; answer user questions through their requested fields")
    elif values.get("approve", "").strip().lower() not in {"yes", "no"}:
        raise ValueError("operation approval requires approve=yes or approve=no; put any explanation in --comment")
    return UserAnswer(
        question_id=question.id,
        values=values,
        answered_at=datetime.now(UTC),
        comment=args.comment,
    )


def _run_store(data_root: Path) -> JsonRunStore:
    return JsonRunStore(data_root / "state")


def _request_from_args(args: argparse.Namespace) -> ResearchRequest:
    """Assemble a ResearchRequest from parsed ``run`` flags.

    Shared by the one-shot ``cli`` and the interactive shell so both build the
    identical request for the same flags.
    """
    return ResearchRequest(
        goal=_goal(args),
        hypothesis=args.hypothesis,
        context=args.context,
        constraints=args.constraint,
        required_artifacts=args.required_artifact,
        budget=RunBudget(
            max_llm_calls=args.max_llm_calls,
            timeout_seconds=args.timeout_seconds,
        ),
        execution_limits=ExecutionLimits(max_tasks=args.max_tasks, max_attempts_per_task=args.max_attempts, max_work_depth=args.max_work_depth),
        permissions=RunPermissions(
            execute_commands=args.execute_commands, approval_mode=args.approval_mode,
        ),
    )


def _exit_code(run: ResearchRun) -> int:
    if run.status == RunStatus.COMPLETED:
        return EXIT_COMPLETED
    if run.status == RunStatus.PAUSED:
        return EXIT_PAUSED
    if run.status == RunStatus.RUNNING:
        return EXIT_RUNNING
    return EXIT_FAILED



def _submit_control(data_root: Path, run: ResearchRun, *, kind: str, text: str = "", store=None):
    """Submit an immutable record, rechecking live status at publication."""
    if run.status not in {RunStatus.RUNNING, RunStatus.PAUSED}:
        raise ValueError("input and interruption require a running or paused Run")
    store = store or _run_store(data_root)
    def validate_live():
        if store.load(run.run_id).status not in {RunStatus.RUNNING, RunStatus.PAUSED}:
            raise ValueError("input and interruption require a running or paused Run")
    return RunControlInbox(data_root / "controls").submit(run.run_id, kind, text, validate_run=validate_live)


def _render_run(run: ResearchRun) -> None:
    for line in render.render_final(run):
        print(line)


def _parser(
    parser_cls: type[argparse.ArgumentParser] = argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = parser_cls(
        prog="sciagent",
        description="Thin command-line entrypoint for the existing sciagent system.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="create and execute a ResearchRun")
    goals = run.add_mutually_exclusive_group(required=True)
    goals.add_argument("--goal", help="research goal text, passed through unchanged")
    goals.add_argument("--goal-file", help="UTF-8 file containing the research goal")
    run.add_argument("--run-id", default=None)
    run.add_argument("--data-root", default=_default_data_root())
    run.add_argument("--hypothesis")
    run.add_argument("--context", default="")
    run.add_argument("--literature-file", help="JSON paper manifest with optional local PDFs")
    run.add_argument("--constraint", action="append", default=[], help="task requirement, such as a Python version")
    run.add_argument(
        "--required-artifact", action="append", default=[], metavar="NAME",
        help="require an exact, case-sensitive registered output_name; repeat for multiple names",
    )
    run.add_argument("--max-tasks", type=int, default=8)
    run.add_argument("--max-work-depth", type=int, default=3, help="maximum WorkRequest depth; root is depth 1")
    run.add_argument("--max-attempts", type=int, default=2)
    run.add_argument("--max-llm-calls", type=int, default=200)
    run.add_argument("--timeout-seconds", type=int, default=7200)
    run.add_argument("--execute-commands", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument(
        "--approval-mode", choices=("manual", "auto_review"), default="manual",
        help="operation approval strategy; auto_review escalates uncertain decisions to the user",
    )
    _workspace_args(run)

    literature = subparsers.add_parser("literature", help="manage caller-supplied paper materials")
    literature_commands = literature.add_subparsers(dest="literature_command", required=True)
    literature_import = literature_commands.add_parser("import", help="add papers to a paused Run")
    literature_import.add_argument("run_id")
    literature_import.add_argument("--file", required=True, help="JSON paper manifest")
    literature_import.add_argument("--data-root", default=_default_data_root())

    show = subparsers.add_parser("show", help="show one persisted ResearchRun")
    show.add_argument("run_id")
    show.add_argument("--data-root", default=_default_data_root())

    answer = subparsers.add_parser("answer", help="answer the current pending question")
    answer.add_argument("run_id")
    answer.add_argument("values", nargs="*", metavar="VALUE|NAME=VALUE")
    answer.add_argument("--field", action="append", default=[], metavar="NAME=VALUE")
    answer.add_argument("--comment", help="optional explanation for an operation approval; does not grant permission")
    answer.add_argument("--data-root", default=_default_data_root())
    _workspace_args(answer)

    resume = subparsers.add_parser("resume", help="continue a persisted running Run")
    resume.add_argument("run_id")
    resume.add_argument("--data-root", default=_default_data_root())
    _workspace_args(resume)

    user_input = subparsers.add_parser("input", help="queue a business supplement without interrupting")
    user_input.add_argument("run_id")
    input_text = user_input.add_mutually_exclusive_group(required=True)
    input_text.add_argument("--text", help="business supplement, saved verbatim")
    input_text.add_argument("--file", help="UTF-8 file containing the supplement")
    user_input.add_argument("--data-root", default=_default_data_root())

    interrupt = subparsers.add_parser("interrupt", help="request interruption; owner confirms the stop")
    interrupt.add_argument("run_id")
    interrupt.add_argument("--text", default="", help="optional business supplement after interruption")
    interrupt.add_argument("--data-root", default=_default_data_root())

    shell = subparsers.add_parser(
        "shell", help="enter the interactive monitoring shell"
    )
    shell.add_argument("--data-root", default=_default_data_root())
    return parser


def _owner_call(data_root: Path, run_id: str, operation: Callable[[], ResearchRun]) -> ResearchRun:
    """Keep the owner alive while Ctrl-C is delivered as a durable cancel request."""
    from .shell import Runner

    runner = Runner()
    runner.start(operation)
    while not runner.done:
        try:
            runner.wait(0.1)
        except KeyboardInterrupt:
            def validate_live():
                store = _run_store(data_root)
                # A very early Ctrl-C may precede the owner's initial snapshot.
                if store.exists(run_id) and store.load(run_id).status not in {RunStatus.RUNNING, RunStatus.PAUSED}:
                    raise ValueError("input and interruption require a running or paused Run")
            try:
                RunControlInbox(data_root / "controls").submit(run_id, "interrupt", validate_run=validate_live)
            except ValueError:
                # The owner committed its terminal result before publication.
                # Keep waiting for that result rather than acknowledge a stale stop.
                continue
            print("Interruption requested; waiting for the Run owner to confirm the stop.", file=sys.stderr)
    result, error = runner.outcome()
    if error is not None:
        raise error
    return result


def cli(
    argv: Sequence[str] | None = None,
    *,
    application_builder: ApplicationBuilder = build_application,
    store_factory: StoreFactory = _run_store,
) -> int:
    args = _parser().parse_args(argv)
    if args.command is None or args.command == "shell":
        from .shell import run_shell

        return run_shell(
            data_root=getattr(args, "data_root", None),
            application_builder=application_builder,
            store_factory=store_factory,
        )
    data_root = Path(args.data_root).expanduser().resolve()

    if args.command == "show":
        _render_run(store_factory(data_root).load(args.run_id))
        return EXIT_COMPLETED

    if args.command in {"input", "interrupt"}:
        store = store_factory(data_root)
        existing = store.load(args.run_id)
        text = (Path(args.file).expanduser().read_text(encoding="utf-8")
                if args.command == "input" and args.file else args.text)
        record = _submit_control(data_root, existing, kind=args.command, text=text, store=store)
        print(f"Saved {record.kind}: {record.event_id}")
        if args.command == "interrupt":
            print("Stop requested; use show to verify the owner's confirmed paused state.")
        elif existing.status == RunStatus.PAUSED:
            print("Input saved; the paused Run advances only after explicit resume.")
        return EXIT_COMPLETED

    if args.command == "run":
        request = _request_from_args(args)
        prepared = load_literature_manifest(args.literature_file) if args.literature_file else None
        application = application_builder(
            data_root=data_root,
            workspaces=_workspace_specs(args),
        )
        literature_kwargs = {"literature": prepared} if prepared is not None else {}
        run_id = args.run_id or _new_run_id()
        run = _owner_call(data_root, run_id, lambda: application.controller.create_run(
            run_id, request, **literature_kwargs,
        ))
    elif args.command == "literature":
        prepared = load_literature_manifest(args.file)
        existing = store_factory(data_root).load(args.run_id)
        if existing.status != RunStatus.PAUSED:
            raise ValueError("literature may only be appended to a paused Run")
        application = application_builder(
            data_root=data_root,
            workspaces={key: record.source for key, record in existing.workspaces.items()},
        )
        run = application.controller.import_literature(args.run_id, prepared)
        _render_run(run)
        return EXIT_COMPLETED
    else:
        existing = store_factory(data_root).load(args.run_id)
        application = application_builder(
            data_root=data_root,
            workspaces=_specs_for_existing_run(args, existing),
        )
        if args.command == "resume":
            run = _owner_call(data_root, args.run_id, lambda: application.controller.resume_run(args.run_id))
        else:
            answer = _answer_from_args(args, existing)
            run = _owner_call(data_root, args.run_id, lambda: application.controller.answer_question(
                args.run_id, answer,
            ))

    _render_run(run)
    return _exit_code(run)


def main() -> int:
    try:
        return cli()
    except KeyboardInterrupt:
        print("Interrupted before a Run owner started.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"sciagent: {error}", file=sys.stderr)
        return EXIT_FAILED
