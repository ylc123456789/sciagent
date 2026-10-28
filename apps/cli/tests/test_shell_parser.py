"""Both CLI entrypoints parse answers and workspace flags through argparse."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from resagent2_cli.main import _answer_from_args, _parser
from resagent2_cli.shell import _NoExitParser, _reject_shell_data_root
from resagent2_contracts import ActionSnapshot, PendingQuestion


def _answer(tokens, requested_fields=None, *, operation=False):
    action = ActionSnapshot(
        action_id="action_1", tool="run_shell", arguments={"command": "python train.py"},
        run_id="run_x", session_id="session_test",
    ) if operation else None
    question = None if requested_fields is None else PendingQuestion(
        session_id="session_test",
        id="question_1", run_id="run_x", text="Choose a value",
        requested_fields=requested_fields, created_at=datetime.now(UTC), action=action,
    )
    run = SimpleNamespace(run_id="run_x", pending_question=question)
    args = _parser(_NoExitParser).parse_args(["answer", "run_x"] + tokens)
    return _answer_from_args(args, run)


@pytest.mark.parametrize("tokens", [["accuracy"], ["primary_metric=accuracy"],
                                   ["--field", "primary_metric=accuracy"]])
def test_answer_single_field_spellings(tokens):
    assert _answer(tokens, ["primary_metric"]).values == {"primary_metric": "accuracy"}


def test_answer_shorthand_rejects_multiple_requested_fields():
    with pytest.raises(ValueError, match="exactly one requested field"):
        _answer(["accuracy"], ["a", "b"])


def test_answer_explicit_fields_and_value_with_equals():
    assert _answer(["a=1", "b=second, mul=2*3"], ["a", "b"]).values == {
        "a": "1", "b": "second, mul=2*3",
    }


def test_business_field_preserves_whitespace_and_line_breaks():
    value = "  自由回答\n    preserve indentation  "
    assert _answer(["--field", "metric=" + value], ["metric"]).values == {"metric": value}


@pytest.mark.parametrize("decision", ["yes", "no"])
def test_operation_comment_is_separate_and_preserves_user_text(decision):
    comment = "  请按原任务执行。\n不得提前评估。  "
    answer = _answer([decision, "--comment", comment], ["approve"], operation=True)
    assert answer.values == {"approve": decision}
    assert answer.comment == comment


@pytest.mark.parametrize("decision", ["yes, because it is safe", "no: use another command", "同意"])
def test_operation_decision_cannot_mix_explanation(decision):
    with pytest.raises(ValueError, match="approve=yes or approve=no"):
        _answer([decision], ["approve"], operation=True)


def test_business_answer_named_approve_still_accepts_free_text():
    answer = _answer(["--field", "approve=Use the existing implementation; x=2"], ["approve"])
    assert answer.values == {"approve": "Use the existing implementation; x=2"}


@pytest.mark.parametrize("comment", ["more requirements", "", "  "])
def test_business_comment_must_use_requested_fields(comment):
    with pytest.raises(ValueError, match="answer user questions through their requested fields"):
        _answer(["custom", "--comment", comment], ["metric"])


@pytest.mark.parametrize("comment", ["", "  "])
def test_explicit_blank_operation_comment_is_preserved(comment):
    answer = _answer(["no", "--comment", comment], ["approve"], operation=True)
    assert answer.comment == comment


def test_answer_rejects_missing_pending_question():
    with pytest.raises(ValueError, match="no pending question"):
        _answer(["accuracy"])


def test_answer_rejects_empty_fields():
    with pytest.raises(ValueError, match="requires a value"):
        _answer([], ["metric"])


@pytest.mark.parametrize("tokens", [["accuracy", "--unknown"], ["accuracy", "--workspace"]])
def test_answer_rejects_invalid_flags(tokens):
    with pytest.raises(ValueError):
        _answer(tokens, ["metric"])


@pytest.mark.parametrize("tokens", [["--data-root", "/other"], ["--data-root=/other"]])
def test_shell_data_root_is_fixed_at_startup(tokens):
    with pytest.raises(ValueError, match="fixed at startup"):
        _reject_shell_data_root(tokens)
