"""Minimal, loss-aware compaction planning for native tool continuation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .context import CONTEXT_TARGET_SHARE, ContextBudgetExceeded, ContextComposer
from .models import ComposedContext, ToolCallTurn
from .tool_calling import native_input_text, tool_messages


COMPACTION_RETAIN_SHARE = 0.20

COMPACTION_SYSTEM_INSTRUCTION = (
    "### Task\n"
    "Compress completed native-tool history into a concise working handoff for "
    "the same task. Return plain text only; do not invent facts.\n\n"
    "### Preserve for continuation\n"
    "Use short sections where useful for the goal and constraints, observed results, "
    "decisions and unexecuted proposals, unfinished work, and needed paths or identifiers. "
    "Omit empty sections. Keep observed successes and failures distinct from decisions "
    "or proposed actions that were not executed. Preserve unresolved questions.\n\n"
    "### Evidence boundary\n"
    "This handoff is navigation context, not evidence that an action succeeded or "
    "that current workspace state was verified; later checked context and "
    "authoritative runtime state take precedence."
)


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    """One atomic checkpoint proposal; callers commit it only after LLM success."""

    history_start: int
    prompt: str
    estimated_tokens: int


def compaction_input(prompt: str, *, runtime_status: str = "") -> dict:
    """Return the exact provider message payload used for one summary request."""

    messages = [
        {"role": "system", "content": COMPACTION_SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]
    if runtime_status:
        messages.append({"role": "user", "content": "Runtime status (current shared limits):\n" + runtime_status})
    return {"messages": messages}


def compaction_input_text(prompt: str, *, runtime_status: str = "") -> str:
    """Serialize the complete summary input for budgeting and full tracing."""

    return json.dumps(compaction_input(prompt, runtime_status=runtime_status), ensure_ascii=False)


def _history_tokens(turns: Sequence[ToolCallTurn]) -> int:
    """Measure only protocol history while validating every call/receipt group."""

    messages = tool_messages(turns)
    return ContextComposer.estimate_tokens(json.dumps(messages, ensure_ascii=False))


def _summary_prompt(
    previous_summary: str | None,
    turns: Sequence[ToolCallTurn],
    *,
    target_summary_chars: int,
) -> str:
    """Summarize the same bounded model-visible history; keep raw turns unchanged."""

    payload = {
        "output_requirement": (
            "Return a non-empty, concise handoff. "
            f"Aim for {target_summary_chars} characters or fewer."
        ),
        "previous_handoff": previous_summary,
        "completed_tool_messages": tool_messages(turns),
    }
    return json.dumps(payload, ensure_ascii=False)


def plan_compaction(
    *,
    current_context: str | ComposedContext,
    schemas: Sequence[dict],
    turns: Sequence[ToolCallTurn],
    history_start: int = 0,
    previous_summary: str | None = None,
    max_input_tokens: int,
    summary_input_limit: int | None = None,
    force: bool = False,
    fixed_input_tokens: int | None = None,
) -> CompactionPlan | None:
    """Plan one bounded summary without mutating or slicing a tool-call group.

    The full native request triggers compaction above 80% of its effective input
    budget. The newest complete turn is always retained, even when that one turn
    alone exceeds the 20% retention target. Older complete turns are retained
    while the whole suffix remains within that target. Already checkpointed turns
    before ``history_start`` are never summarized again.

    ``force`` only bypasses the 80% trigger so callers can recover when required
    current context could not be composed after reserving history. It does not
    relax the retention or summary-input limits.

    The returned boundary is only a proposal. Callers must leave their checkpoint
    unchanged if the summary call fails or returns an empty result. Source turns
    remain in durable state after a successful checkpoint as well.
    """

    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    if summary_input_limit is None:
        summary_input_limit = max_input_tokens
    if summary_input_limit < 1:
        raise ValueError("summary_input_limit must be positive")
    if history_start < 0 or history_start > len(turns):
        raise ValueError("history_start is outside the tool history")

    active_turns = turns[history_start:]
    # This exact wire projection also rejects an unfinished call/receipt group.
    request_tokens = ContextComposer.estimate_tokens(
        native_input_text(current_context, list(schemas), active_turns)
    )
    if not force and request_tokens <= max_input_tokens * CONTEXT_TARGET_SHARE:
        return None
    if len(active_turns) < 2:
        # There is no older prefix to summarize without discarding the newest
        # complete turn. The regular native budget check remains authoritative.
        return None

    if fixed_input_tokens is None:
        fixed_input_tokens = ContextComposer.estimate_tokens(
            native_input_text(current_context, list(schemas), [])
        )
    history_allowance = max_input_tokens - fixed_input_tokens
    if history_allowance < 1:
        raise ContextBudgetExceeded("fixed contract, complete task and directory exhaust input capacity")
    retain_target = max(1, int(history_allowance * COMPACTION_RETAIN_SHARE))
    first_kept = len(active_turns) - 1
    # Always keep the newest turn whole. Add older turns only when the resulting
    # suffix still fits the token target; an oversized turn is never split.
    while first_kept > 0:
        candidate = active_turns[first_kept - 1 :]
        if _history_tokens(candidate) > retain_target:
            break
        first_kept -= 1

    if first_kept == 0:
        return None

    source = active_turns[:first_kept]
    # Writing target only; the composer checks the complete continuation request.
    summary_allowance = max(1, history_allowance - _history_tokens(active_turns[first_kept:]))
    target_summary_chars = min(4096, max(1, min(summary_allowance, history_allowance // 4))) * 4
    prompt = _summary_prompt(
        previous_summary,
        source,
        target_summary_chars=target_summary_chars,
    )
    status = current_context.runtime_status if isinstance(current_context, ComposedContext) else ""
    summary_tokens = ContextComposer.estimate_tokens(compaction_input_text(prompt, runtime_status=status))
    if summary_tokens > summary_input_limit:
        raise ContextBudgetExceeded(
            "completed tool-history prefix exceeds the compaction input budget; "
            "no history boundary was advanced"
        )
    return CompactionPlan(
        history_start=history_start + first_kept,
        prompt=prompt,
        estimated_tokens=summary_tokens,
    )
