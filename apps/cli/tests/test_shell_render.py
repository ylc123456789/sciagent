"""Pure renderer tests: state -> lines, no terminal, no LLM."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from resagent2_contracts import ActionSnapshot

from resagent2_cli.render import (
    render_artifacts,
    render_final,
    render_live,
    render_trace,
)


def _run(goal="g", status="running", llm_used=5, llm_max=200):
    return SimpleNamespace(
        run_id="run_x",
        status=SimpleNamespace(value=status),
        request=SimpleNamespace(
            goal=goal, budget=SimpleNamespace(max_llm_calls=llm_max)
        ),
        llm_calls_used=llm_used,
        usage=SimpleNamespace(outcomes={"succeeded": llm_used, "failed": 0, "unknown": 0}),
        workflow=None,
        latest_scientific_assessment=None,
        pending_question=None,
        artifacts={},
        final_opinion=None,
        completion_violations=[],
        terminal_error=None,
    )


def test_render_live_none():
    assert render_live(None) == ["… starting …"]


def test_render_final_preserves_terminal_cause():
    from resagent2_contracts import ErrorCode, ModuleError

    run = _run(status="failed")
    run.terminal_error = ModuleError(code=ErrorCode.TOOL_FAILED, message="provider unavailable", retryable=False)
    assert "Error: tool_failed: provider unavailable" in render_final(run)


def test_render_live_header_and_pending_question():
    run = _run()
    run.pending_question = SimpleNamespace(
        text="pick the primary metric", requested_fields=["primary_metric"], action=None
    )
    joined = "\n".join(render_live(run, None))
    assert "run_x" in joined
    assert "running" in joined
    assert "pick the primary metric" in joined
    assert "primary_metric" in joined


def test_render_live_shows_last_tool_activity():
    run = _run()
    lines = render_live(run, [{"agent": "coding", "tool": "read_file", "step": 3}])
    assert "→ coding/read_file (step 3)" in lines


def test_render_live_shows_serial_tools_and_compaction():
    run = _run()
    serial = {"agent": "coding", "tool": None, "tools": ["read_file", "write_file"], "step": 3}
    compacted = {
        "agent": "coding",
        "tool": None,
        "action_valid": None,
        "included_sections": ["goal", "compaction"],
    }

    assert "→ coding/read_file → write_file (step 3)" in render_live(run, [serial])
    assert "→ coding/compaction" in render_live(run, [compacted])


def test_render_live_task_attempts_and_error():
    task = SimpleNamespace(
        id="task_1",
        workflow_agent_kind=SimpleNamespace(value="executor"),
        execution_ref="coding@1",
        status=SimpleNamespace(value="failed"),
        instruction="fix the SE bug",
        attempts=[
            SimpleNamespace(
                number=1,
                status=SimpleNamespace(value="failed"),
                report="editing finished",
                error=SimpleNamespace(
                    code=SimpleNamespace(value="TOOL_FAILED"),
                    message="boom",
                ),
            )
        ],
    )
    run = _run()
    run.workflow = SimpleNamespace(tasks=[task])
    joined = "\n".join(render_live(run, None))
    assert "task_1" in joined
    assert "[coding@1]" in joined
    assert "failed" in joined
    assert "attempt 1" in joined
    assert "editing finished" in joined
    assert "TOOL_FAILED: boom" in joined


def test_render_final_basic_summary():
    run = _run(goal="my goal", status="completed")
    joined = "\n".join(render_final(run))
    assert "Run: run_x" in joined
    assert "Status: completed" in joined
    assert "Goal: my goal" in joined
    assert "LLM calls: 5/200" in joined
    assert "Request reservations: succeeded=5, failed=0, unknown=0" in joined


def test_render_final_prefers_final_opinion_without_changing_interim_history():
    run = _run(status="completed")
    run.latest_scientific_assessment = SimpleNamespace(statement="Dataset is missing")
    run.final_opinion = SimpleNamespace(
        verdict=SimpleNamespace(value="inconclusive"),
        statement="Dataset directory is now available; contents not validated",
    )
    before = deepcopy(run)

    joined = "\n".join(render_final(run))

    assert "Final opinion:" in joined
    assert "Dataset directory is now available" in joined
    assert "Scientific assessment" not in joined
    assert "Dataset is missing" not in joined
    assert run == before


@pytest.mark.parametrize("status", ["running", "paused", "failed"])
def test_render_final_keeps_interim_assessment_without_final_opinion(status):
    run = _run(status=status)
    run.latest_scientific_assessment = SimpleNamespace(statement="Need user-provided data")

    joined = "\n".join(render_final(run))

    assert "Scientific assessment (interim):" in joined
    assert "Need user-provided data" in joined
    assert "Final opinion:" not in joined


def test_render_final_pending_question_fields():
    run = _run(status="paused")
    run.pending_question = SimpleNamespace(
        text="choose", requested_fields=["metric", "seed"], action=None
    )
    joined = "\n".join(render_final(run))
    assert "Pending user question:" in joined
    assert "Fields: metric, seed" in joined


@pytest.mark.parametrize("renderer", [render_live, render_final])
def test_question_options_and_free_text_entry_are_visible(renderer):
    run = _run(status="paused")
    run.pending_question = SimpleNamespace(text="Choose the measurement", requested_fields=["metric"],
        action=None, options={"metric": ["accuracy", "negative log likelihood"]})

    joined = "\n".join(renderer(run))

    assert "Options for metric:" in joined
    assert "negative log likelihood" in joined
    assert "your own text" in joined
    assert "/answer prompts" in joined


@pytest.mark.parametrize("renderer", [render_live, render_final])
def test_operation_renders_full_action_and_explanation_entry(renderer):
    run = _run(status="paused")
    command = "python train.py " + "--long-argument " * 40
    action = ActionSnapshot(action_id="action_train", tool="run_shell", arguments={"command": command},
        context={"cwd": "/workspace", "environment": "/bound/env"}, run_id=run.run_id, session_id="session_coding")
    run.pending_question = SimpleNamespace(text="Confirm command", requested_fields=["approve"], action=action)

    joined = "\n".join(renderer(run))

    assert command in joined
    assert "/workspace" in joined and "/bound/env" in joined
    assert "yes or no" in joined and "--comment TEXT" in joined


@pytest.mark.parametrize("renderer", [render_live, render_final])
@pytest.mark.parametrize("is_approval, field, label", [
    (False, "metric", "User question"),
    (False, "approve", "User question"),
    (True, "approve", "Operation approval"),
])
def test_pending_question_kind_comes_from_action(renderer, is_approval, field, label):
    run = _run(status="paused")
    action = ActionSnapshot(
        action_id="action_train", tool="run_shell",
        arguments={"command": "python train.py"},
        context={"cwd": "/workspace", "environment": "/env"},
        run_id=run.run_id, session_id="session_coding",
        task_id="task_train", attempt_number=1,
    ) if is_approval else None
    run.pending_question = SimpleNamespace(
        text="Choose the next step", requested_fields=[field], action=action,
    )
    before = deepcopy(run)

    lines = renderer(run)

    if renderer is render_live:
        assert f"? {label}: Choose the next step" in lines
        assert f"  answer: {field}" in lines
    else:
        assert f"Pending {label.lower()}:" in lines
        assert "  Choose the next step" in lines
        assert f"  Fields: {field}" in lines
    other_label = "User question" if is_approval else "Operation approval"
    assert other_label.lower() not in "\n".join(lines).lower()
    assert run == before


def test_render_artifacts_empty():
    assert render_artifacts(_run()) == ["No artifacts."]


def test_render_artifacts_listing():
    run = _run()
    run.artifacts = {
        "artifact_1": SimpleNamespace(
            id="artifact_1",
            kind="experiment_result",
            producer=SimpleNamespace(value="experiment"),
            uri="file:///tmp/x.json",
        )
    }
    joined = "\n".join(render_artifacts(run))
    assert "Artifacts (1):" in joined
    assert "artifact_1: experiment_result [experiment] file:///tmp/x.json" in joined


def test_render_trace_empty():
    assert render_trace([]) == ["No trace records."]


def test_render_trace_full_level():
    records = [
        {
            "sequence": 1,
            "agent": "scientific",
            "tool": "finish",
            "step": 2,
            "request_text": "REQ",
            "raw_response_text": "RESP",
            "raw_reasoning_text": "THINK",
        }
    ]
    joined = "\n".join(render_trace(records))
    assert "agent=scientific" in joined
    assert "tool=finish" in joined
    assert "[request]" in joined and "REQ" in joined
    assert "[reasoning]" in joined and "THINK" in joined
    assert "[response]" in joined and "RESP" in joined


def test_render_trace_native_calls_with_null_content():
    calls = [{"id": "call_read", "type": "function", "function": {
        "name": "read_file", "arguments": '{"path":"train.py"}',
    }}]
    record = {"tool": "read_file", "raw_response_text": None, "raw_tool_calls": calls}
    before = deepcopy(record)
    joined = "\n".join(render_trace([record]))
    assert "[tool_calls]" in joined and "call_read" in joined and "train.py" in joined
    assert "[response]" not in joined
    assert record == before


def test_render_trace_labels_serial_tools_and_compaction():
    records = [
        {
            "sequence": 1,
            "agent": "coding",
            "tool": None,
            "tools": ["read_file", "write_file"],
            "step": 4,
        },
        {
            "sequence": 2,
            "agent": "coding",
            "tool": None,
            "action_valid": None,
            "included_sections": ["compaction"],
            "step": None,
        },
    ]

    joined = "\n".join(render_trace(records))
    assert "tool=read_file → write_file step=4" in joined
    assert "tool=compaction step=None" in joined


def test_render_trace_metadata_does_not_invent_raw_calls():
    joined = "\n".join(render_trace([{"tool": "finish", "tool_calls_sha256": "opaque"}]))
    assert "[tool_calls]" not in joined and "opaque" not in joined


@pytest.mark.parametrize("subject", [None, "metrics.json"])
def test_render_final_identifies_completion_violation_subject(subject):
    from resagent2_orchestrator import CompletionViolation

    run = _run(status="failed")
    run.completion_violations = [CompletionViolation(
        code="required_artifact_missing", message="required artifact was not produced",
        subject=subject,
    )]
    suffix = f" [{subject}]" if subject is not None else ""
    assert (
        f"  required_artifact_missing{suffix}: required artifact was not produced"
        in render_final(run)
    )
