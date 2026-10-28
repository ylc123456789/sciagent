"""The shell and the one-shot CLI must share the same flag -> request building."""

from __future__ import annotations

from e2e.native_fixtures import tool_turns

import io
import json
import shlex

import pytest

from resagent2_cli.main import _parser, _request_from_args, _workspace_specs, cli
from resagent2_cli.shell import _NoExitParser
from resagent2_orchestrator import InMemoryRunStore


def test_request_from_args_builds_request(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    args = _parser().parse_args(
        ["run", "--goal", "g", "--workspace", str(workspace), "--max-tasks", "3"]
    )
    request = _request_from_args(args)
    assert request.goal == "g"
    assert request.execution_limits.max_tasks == 3
    specs = _workspace_specs(args)
    assert specs["ws_main"].location == str(workspace.resolve())


def test_shell_reuses_run_subparser():
    args = _parser(_NoExitParser).parse_args(
        ["run", "--goal", "g", "--hypothesis", "h", "--max-llm-calls", "5"]
    )
    request = _request_from_args(args)
    assert request.goal == "g"
    assert request.hypothesis == "h"
    assert request.budget.max_llm_calls == 5


def test_cli_shell_subcommand_routes_to_shell(monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda *a: "/quit")
    code = cli(
        ["shell", "--data-root", str(tmp_path)],
        application_builder=lambda **kwargs: None,
        store_factory=lambda root: InMemoryRunStore(),
    )
    assert code == 0


def test_cli_no_arguments_routes_to_shell(monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda *a: "/quit")
    code = cli(
        [],
        application_builder=lambda **kwargs: None,
        store_factory=lambda root: InMemoryRunStore(),
    )
    assert code == 0


@pytest.mark.parametrize("answer_tokens", [
    [], ["accuracy"], ["primary_metric=accuracy"], ["--field", "primary_metric=accuracy"],
])
@pytest.mark.parametrize("repeat_workspace", [False, True])
def test_shell_answer_reaches_controller_and_resumes_same_session(
    tmp_path, monkeypatch, answer_tokens, repeat_workspace,
):
    from resagent2_cli import composition
    from resagent2_cli.shell import Shell
    from resagent2_contracts import RunStatus, SessionStatus
    from resagent2_runtime import ScriptedLLMClient

    actions = [
        {"tool": "ask_user", "arguments": {
            "assessment": {"statement": "Need the primary metric"},
            "text": "Which metric should be primary?", "requested_fields": ["primary_metric"],
        }},
        {"tool": "finish", "arguments": {
            "report": "Metric recorded; no empirical conclusion yet.",
            "artifacts": [{"kind": "scientific_opinion", "path": "opinion.json",
                           "media_type": "application/json", "summary": "Conclusion",
                           "content": {"verdict": "inconclusive", "statement": "Need empirical evidence"}}],
        }},
    ]
    monkeypatch.setattr(composition, "_client", lambda: ScriptedLLMClient(tool_turns(actions)))
    args = _parser().parse_args([
        "run", "--goal", "Choose a metric", "--workspace", str(tmp_path), "--read-only",
    ])
    specs = _workspace_specs(args)
    app = composition.build_application(data_root=tmp_path / "data", workspaces=specs)
    paused = app.controller.create_run("run_shell_answer", _request_from_args(args))
    assert paused.status == RunStatus.PAUSED
    builds = []

    def builder(**kwargs):
        builds.append(kwargs)
        return app

    shell = Shell(data_root=tmp_path / "data", application_builder=builder,
                  store=app.run_store, stream=io.StringIO())
    shell.current_run_id = paused.run_id
    if not answer_tokens:
        monkeypatch.setattr("builtins.input", lambda prompt: "accuracy")
    workspace_flags = ["--workspace", str(tmp_path), "--read-only"] if repeat_workspace else []
    shell._dispatch(shlex.join(["/answer", *answer_tokens, *workspace_flags]))
    assert shell.runner.done
    result, error = shell.runner.outcome()
    assert error is None
    assert result.status == RunStatus.COMPLETED, result.terminal_error
    saved = app.run_store.load(paused.run_id)
    assert saved.scientific_session.id == paused.scientific_session.id
    assert saved.scientific_session.status == SessionStatus.COMPLETED
    assert saved.answers[0].question_text == paused.pending_question.text
    assert saved.answers[0].values == {"primary_metric": "accuracy"}
    assert saved.pending_question is None
    assert saved.workspaces == paused.workspaces
    assert builds[0]["workspaces"] == specs
    assert saved.answers[0].session_id == saved.scientific_session.id
    assert not any(ref.kind in {"question", "answer"} for ref in saved.artifacts.values())


def test_shell_prompt_accepts_free_text_for_each_business_field(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from resagent2_cli.shell import Shell
    from resagent2_contracts import PendingQuestion

    question = PendingQuestion(
        id="question_details", run_id="run_prompt", session_id="session_prompt",
        text="Choose the metric and execution strategy",
        requested_fields=["metric", "strategy"],
        options={"metric": ["accuracy", "NLL"], "strategy": ["reuse", "new"]},
        created_at=datetime.now(UTC),
    )
    run = SimpleNamespace(run_id=question.run_id, pending_question=question, workspaces={})
    answers = []
    shell = Shell(
        data_root=tmp_path, store=SimpleNamespace(load=lambda run_id: run), stream=io.StringIO(),
        application_builder=lambda **kwargs: SimpleNamespace(controller=SimpleNamespace(
            answer_question=lambda run_id, answer: answers.append(answer))),
    )
    shell.current_run_id = run.run_id
    shell.runner = SimpleNamespace(start=lambda fn: fn())
    shell._watch = lambda run_id: None
    replies = iter(["", " 自定义指标 ", "复用已有环境；阈值 x=0.7"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))

    shell._dispatch("/answer")

    assert answers[0].values == {"metric": " 自定义指标 ", "strategy": "复用已有环境；阈值 x=0.7"}
    assert answers[0].comment is None
    output = shell.stream.getvalue()
    assert "Options for metric" in output and "accuracy" in output
    assert "Options for strategy" in output and "reuse" in output


@pytest.mark.parametrize("decision, comment", [("yes", ""), ("no", " 不要提前执行正式评估。 ")])
def test_shell_operation_prompt_keeps_decision_separate(tmp_path, monkeypatch, decision, comment):
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from resagent2_cli.shell import Shell
    from resagent2_contracts import ActionSnapshot, PendingQuestion

    action = ActionSnapshot(action_id="action_prompt", tool="run_shell",
        arguments={"command": "python evaluate.py"}, context={"cwd": "/workspace", "environment": "/bound/env"},
        run_id="run_prompt", session_id="session_prompt")
    question = PendingQuestion(id="question_prompt", run_id=action.run_id, session_id=action.session_id,
        text="Confirm execution", requested_fields=["approve"], action=action, created_at=datetime.now(UTC))
    run = SimpleNamespace(run_id=question.run_id, pending_question=question, workspaces={})
    answers = []
    shell = Shell(data_root=tmp_path, store=SimpleNamespace(load=lambda run_id: run), stream=io.StringIO(),
        application_builder=lambda **kwargs: SimpleNamespace(controller=SimpleNamespace(
            answer_question=lambda run_id, answer: answers.append(answer))))
    shell.current_run_id = run.run_id
    shell.runner = SimpleNamespace(start=lambda fn: fn())
    shell._watch = lambda run_id: None
    replies = iter(["yes because this is safe", decision, comment])
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))

    shell._dispatch("/answer")

    assert answers[0].values == {"approve": decision}
    assert answers[0].comment == (comment or None)
    output = shell.stream.getvalue()
    assert "python evaluate.py" in output and "/bound/env" in output
    assert "Enter yes or no" in output


def test_shell_run_passes_same_literature_option_as_one_shot_cli(tmp_path):
    from types import SimpleNamespace
    from resagent2_cli.shell import Shell

    manifest = tmp_path / "papers.json"
    manifest.write_text(json.dumps({"papers": [{"title": "External shell paper"}]}))
    calls = []

    class Controller:
        def create_run(self, run_id, request, *, literature=()):
            calls.append((run_id, request, literature))
            return None

    class Runner:
        def start(self, fn):
            fn()

    shell = Shell(data_root=tmp_path, application_builder=lambda **kwargs: SimpleNamespace(controller=Controller()),
                  store=InMemoryRunStore(), stream=io.StringIO())
    shell.runner = Runner()
    shell._watch = lambda *args, **kwargs: None
    shell._dispatch(shlex.join(["/run", "--run-id", "run_shell_import", "--goal", "Use paper",
                               "--literature-file", str(manifest)]))
    assert calls[0][0] == "run_shell_import"
    assert calls[0][1].goal == "Use paper"
    assert calls[0][2][0].paper.title == "External shell paper"
