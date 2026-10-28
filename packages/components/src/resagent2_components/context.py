"""Persistent read-source facts, independent of model context projection."""

from resagent2_runtime import AgentState


def remember_source(state: AgentState, key: str, value: str) -> list[str]:
    """Return a deduplicated source list for a tool's existing memory update."""
    current = state.memory.get(key, [])
    values = list(current) if isinstance(current, list) else []
    if value not in values:
        values.append(value)
    return values
