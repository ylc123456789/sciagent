"""Answer keys are machine identifiers; questions and values remain prose."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from resagent2_contracts import PendingQuestion, QuestionDraft, RecordedAnswer, UserAnswer


def _messages(name: str):
    now = datetime.now(UTC)
    text = "请选择模式：1=add，2=mul。Which mode?"
    value = "第二个，mul；a=2，b=3\nPlease keep this explanation."
    return [
        (QuestionDraft, dict(text=text, requested_fields=[name])),
        (PendingQuestion, dict(
            id="question_mode", run_id="run_test", session_id="session_test", text=text,
            requested_fields=[name], created_at=now,
        )),
        (UserAnswer, dict(
            question_id="question_mode", values={name: value}, answered_at=now,
        )),
        (RecordedAnswer, dict(
            question_id="question_mode", values={name: value},
            answered_at=now, question_text=text,
            requested_fields=[name], run_id="run_test", session_id="session_test",
        )),
    ]


@pytest.mark.parametrize("name", ["mode", "file_choice", "Metric2", "a" * 64])
def test_answer_key_round_trips_without_restricting_question_or_value(name):
    for model, data in _messages(name):
        result = model.model_validate(data)
        assert model.model_validate_json(result.model_dump_json()) == result
        for key, value in data.items():
            assert getattr(result, key) == value


@pytest.mark.parametrize("name", [
    "", "1mode", "_mode", "mode-choice", "mode.choice", "模式",
    "mode choice", "mode=mul", 'option: 1="add" or 2="mul"',
    " mode", "mode ", "mode\n", "a" * 65,
])
def test_question_and_answer_reject_the_same_invalid_key_without_renaming(name):
    for model, data in _messages(name):
        with pytest.raises(ValidationError, match="String should match pattern"):
            model.model_validate(data)


def test_public_question_and_answer_json_schemas_share_the_key_pattern():
    question_items = QuestionDraft.model_json_schema()["properties"]["requested_fields"]["items"]
    pending_items = PendingQuestion.model_json_schema()["properties"]["requested_fields"]["items"]
    answer_values = UserAnswer.model_json_schema()["properties"]["values"]
    recorded_values = RecordedAnswer.model_json_schema()["properties"]["values"]
    assert question_items["pattern"] == r"^[A-Za-z][A-Za-z0-9_]{0,63}$"
    assert pending_items["pattern"] == question_items["pattern"]
    for values in (answer_values, recorded_values):
        assert set(values["patternProperties"]) == {question_items["pattern"]}
        assert values["propertyNames"]["description"] == question_items["description"]
    assert "machine-readable" in question_items["description"]


@pytest.mark.parametrize("comment", ["", "  ", "不同于选项的业务补充\n原文"])
def test_business_feedback_must_use_answer_values_instead_of_operation_comment(comment):
    data = dict(_messages("choice")[-1][1], comment=comment)
    with pytest.raises(ValidationError, match="business feedback belongs in values"):
        RecordedAnswer.model_validate(data)
