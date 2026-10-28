"""Lossless fixed-context composition."""
import pytest
from sciagent_runtime import ContextBudgetExceeded, ContextComposer, ContextSection

def test_context_rejects_zero_budget():
    with pytest.raises(ContextBudgetExceeded):
        ContextComposer().compose("rules", [], max_tokens=0)

def test_complete_sections_keep_order_and_content():
    sections = [ContextSection(name="task", content="original task"),
                ContextSection(name="material_index", content="all IDs")]
    result = ContextComposer().compose("rules", sections, max_tokens=1000)
    assert result.included_sections == ["system", "task", "material_index"]
    assert result.omitted_sections == []
    assert result.task_context == "## task\noriginal task"
    assert result.material_index == "## material_index\nall IDs"

def test_exact_budget_counts_headings_and_separators():
    composer = ContextComposer()
    sections = [ContextSection(name="task", content="complete")]
    exact = composer.compose("rules", sections, max_tokens=1000).estimated_tokens
    assert composer.compose("rules", sections, max_tokens=exact).estimated_tokens == exact
    with pytest.raises(ContextBudgetExceeded):
        composer.compose("rules", sections, max_tokens=exact-1)

def test_duplicate_section_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        ContextComposer().compose("rules", [ContextSection(name="task",content="a"),
                                          ContextSection(name="task",content="b")], max_tokens=1000)


def test_runtime_authority_is_required_in_history_part_not_task_requirements():
    from sciagent_runtime.models import ToolCallTurn
    from sciagent_runtime.tool_calling import native_input

    composer = ContextComposer()
    sections = [ContextSection(name="task", content="unchanged business requirements"),
                ContextSection(name="runtime_authority", content="complete current authority facts"),
                ContextSection(name="material_index", content="complete artifact directory")]
    context = composer.compose("rules", sections, max_tokens=1000)
    assert context.task_context == "## task\nunchanged business requirements"
    assert context.runtime_context == "## runtime_authority\ncomplete current authority facts"
    assert context.omitted_sections == []
    with pytest.raises(ContextBudgetExceeded):
        composer.compose("rules", sections, max_tokens=context.estimated_tokens - 1)
    context = context.model_copy(update={"history_summary": "earlier lossy recap"})
    wire = native_input(context, [], [ToolCallTurn(runtime_notice="actual past operation")])
    messages = [message["content"] for message in wire["messages"]]
    history_position = next(i for i, text in enumerate(messages) if "actual past operation" in text)
    authority_position = next(i for i, text in enumerate(messages) if "complete current authority facts" in text)
    assert history_position < authority_position < len(messages) - 1
    assert messages[-1] == context.material_index
    assert "not task requirements" in messages[authority_position]
