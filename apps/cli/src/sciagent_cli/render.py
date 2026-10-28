"""Pure rendering of persisted Run state and LLM trace records.

Every function here is a side-effect-free mapping from structured state to a
list of text lines. The interactive shell calls these; the existing one-shot
``_render_run`` also delegates here so the terminal summary stays identical in
both modes.
"""

from __future__ import annotations

import json
from typing import Any

_TERMINAL_MARKS = {
    "pending": "○",
    "running": "●",
    "completed": "✓",
    "failed": "✗",
    "blocked": "⊘",
    "cancelled": "■",
    "interrupted": "■",
    "needs_user_input": "?",
}


def _truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _trace_activity(record: dict) -> str:
    tool = record.get("tool")
    if isinstance(tool, str) and tool:
        return tool
    tools = record.get("tools")
    if isinstance(tools, list):
        names = [name for name in tools if isinstance(name, str) and name]
        if names:
            return " → ".join(names)
    sections = record.get("included_sections")
    if isinstance(sections, list) and "compaction" in sections:
        return "compaction"
    return "None"


def _question_label(question: Any) -> str:
    return "User question" if question.action is None else "Operation approval"


def _question_details(question: Any) -> list[str]:
    lines = []
    for field, options in (getattr(question, "options", None) or {}).items():
        lines.append(f"  Options for {field}:")
        lines.extend(f"    - {option}" for option in options)
    if question.action is None:
        lines.append("  Reply with an option or your own text in each requested field.")
    else:
        action = question.action
        lines.append(f"  Tool: {action.tool}")
        lines.append("  Arguments:")
        lines.extend(f"    {line}" for line in json.dumps(action.arguments, ensure_ascii=False, indent=2).splitlines())
        lines.append("  Execution context:")
        lines.extend(f"    {line}" for line in json.dumps(action.context, ensure_ascii=False, indent=2).splitlines())
        lines.append("  Reply yes or no; use --comment TEXT for an optional explanation.")
    lines.append("  In the shell, /answer prompts for an answer; explicit values and --field NAME=VALUE also work.")
    return lines


def render_question(question: Any) -> list[str]:
    """Show the full question and answer choices without restricting free text."""
    return [
        f"Pending {_question_label(question).lower()}:",
        f"  {question.text}",
        f"  Fields: {', '.join(question.requested_fields)}",
        *_question_details(question),
    ]


def render_live(run: Any, trace_records: list[dict] | None = None) -> list[str]:
    """One compact, redrawable block for the in-progress view.

    ``trace_records`` is the most recent batch of append-only LLM trace lines
    (metadata level only: ``agent`` and ``tool`` names, never raw content).
    """
    if run is None:
        return ["… starting …"]
    lines = [
        f"Run {run.run_id} · {run.status.value} "
        f"· LLM {run.llm_calls_used}/{run.request.budget.max_llm_calls}"
    ]
    if run.workflow is not None:
        for task in run.workflow.tasks:
            mark = _TERMINAL_MARKS.get(task.status.value, "·")
            attempts = f" · attempt {len(task.attempts)}" if task.attempts else ""
            configuration = task.execution_ref or task.workflow_agent_kind.value
            lines.append(
                f"{mark} {task.id} [{configuration}] "
                f"{task.status.value}{attempts}"
            )
            if task.instruction:
                lines.append(f"     {_truncate(task.instruction, 64)}")
            if task.attempts:
                latest = task.attempts[-1]
                if latest.report:
                    lines.append(f"     {_truncate(latest.report, 72)}")
                if latest.error is not None:
                    lines.append(
                        f"     {latest.error.code.value}: {latest.error.message}"
                    )
    if trace_records:
        record = trace_records[-1]
        activity = f"{record.get('agent')}/{_trace_activity(record)}"
        if record.get("step") is not None:
            activity += f" (step {record.get('step')})"
        lines.append(f"→ {activity}")
    if run.pending_question is not None:
        question = run.pending_question
        lines.append(f"? {_question_label(question)}: {question.text}")
        if question.requested_fields:
            lines.append(f"  answer: {', '.join(question.requested_fields)}")
        lines.extend(_question_details(question))
    return lines


def render_final(run: Any) -> list[str]:
    """The full terminal summary (identical to the one-shot CLI output)."""
    lines = [
        f"Run: {run.run_id}",
        f"Status: {run.status.value}",
        f"Goal: {run.request.goal}",
        f"LLM calls: {run.llm_calls_used}/{run.request.budget.max_llm_calls}",
        "Request reservations: " + ", ".join(f"{key}={value}" for key, value in run.usage.outcomes.items()),
    ]
    if run.terminal_error is not None:
        lines.append(f"Error: {run.terminal_error.code.value}: {run.terminal_error.message}")
    # Keep the interim history in Run state, but prefer its final conclusion.
    if run.final_opinion is None and run.latest_scientific_assessment is not None:
        lines.append("Scientific assessment (interim):")
        lines.append(f"  {run.latest_scientific_assessment.statement}")
    if run.workflow is not None:
        lines.append("Tasks:")
        for task in run.workflow.tasks:
            suffix = f" ({len(task.attempts)} attempt(s))" if task.attempts else ""
            lines.append(f"  {task.id}: {task.status.value}{suffix}")
            if task.attempts:
                latest = task.attempts[-1]
                if latest.report:
                    lines.append(f"    {latest.report}")
                if latest.error is not None:
                    lines.append(
                        f"    {latest.error.code.value}: {latest.error.message}"
                    )
    if run.pending_question is not None:
        lines.extend(render_question(run.pending_question))
    if run.artifacts:
        lines.append("Artifacts:")
        for artifact in run.artifacts.values():
            lines.append(
                f"  {artifact.id}: {artifact.kind} "
                f"[{artifact.producer.value}] {artifact.uri}"
            )
    if run.final_opinion is not None:
        lines.append("Final opinion:")
        lines.append(f"  Verdict: {run.final_opinion.verdict.value}")
        lines.append(f"  {run.final_opinion.statement}")
    if run.completion_violations:
        lines.append("Completion violations:")
        for violation in run.completion_violations:
            subject = f" [{violation.subject}]" if violation.subject is not None else ""
            lines.append(f"  {violation.code.value}{subject}: {violation.message}")
    return lines


def render_artifacts(run: Any) -> list[str]:
    """Artifact-only view for the ``/artifacts`` command."""
    if not run.artifacts:
        return ["No artifacts."]
    lines = [f"Artifacts ({len(run.artifacts)}):"]
    for artifact in run.artifacts.values():
        lines.append(
            f"  {artifact.id}: {artifact.kind} "
            f"[{artifact.producer.value}] {artifact.uri}"
        )
    return lines


def render_trace(records: list[dict]) -> list[str]:
    """Raw LLM trace view for the ``/trace`` command (full level)."""
    if not records:
        return ["No trace records."]
    lines: list[str] = []
    for record in records:
        lines.append("-" * 40)
        meta = f"seq={record.get('sequence')} agent={record.get('agent')}"
        meta += f" tool={_trace_activity(record)} step={record.get('step')}"
        if record.get("latency_ms") is not None:
            meta += f" latency_ms={record.get('latency_ms')}"
        lines.append(meta)
        request = record.get("request_text")
        response = record.get("raw_response_text")
        reasoning = record.get("raw_reasoning_text")
        if request is not None:
            lines.append("[request]")
            lines.append(request)
        if reasoning:
            lines.append("[reasoning]")
            lines.append(reasoning)
        if response is not None:
            lines.append("[response]")
            lines.append(response)
        if record.get("raw_tool_calls") is not None:
            lines.append("[tool_calls]")
            lines.append(json.dumps(record["raw_tool_calls"], ensure_ascii=False, indent=2))
    return lines
