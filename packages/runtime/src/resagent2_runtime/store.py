"""Session persistence protocol, in-memory and atomic JSON implementations."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from resagent2_contracts import SCHEMA_VERSION, SessionId

from .models import AgentState


class SessionLoadError(ValueError):
    """A persisted session cannot be read as a valid current snapshot."""


class SessionStore(Protocol):
    """Persistence boundary used by AgentLoop after every state change."""

    def save(self, state: AgentState) -> None:
        """Persist a complete session snapshot."""

    def load(self, session_id: SessionId) -> AgentState:
        """Load the latest session snapshot."""

    def exists(self, session_id: SessionId) -> bool:
        """Return whether a session already exists."""


class InMemorySessionStore:
    """Deep-copying session store for tests and local runtime development."""

    def __init__(self) -> None:
        self._current: dict[str, AgentState] = {}
        self._history: dict[str, list[AgentState]] = {}

    def save(self, state: AgentState) -> None:
        snapshot = state.model_copy(deep=True)
        self._current[state.session_id] = snapshot
        self._history.setdefault(state.session_id, []).append(snapshot)

    def load(self, session_id: SessionId) -> AgentState:
        return self._current[session_id].model_copy(deep=True)

    def exists(self, session_id: SessionId) -> bool:
        return session_id in self._current

    def history(self, session_id: SessionId) -> list[AgentState]:
        """Return independent snapshots for persistence assertions and debugging."""

        return [state.model_copy(deep=True) for state in self._history[session_id]]


class JsonSessionStore:
    """Atomic one-file-per-session JSON persistence for local recovery."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _path(self, session_id: SessionId) -> Path:
        return self.root / f"{session_id}.json"

    def save(self, state: AgentState) -> None:
        destination = self._path(state.session_id)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=f".{state.session_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                os.chmod(temporary, 0o600)
                handle.write(state.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def load(self, session_id: SessionId) -> AgentState:
        try:
            value = json.loads(self._path(session_id).read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
                raise SessionLoadError(f"Session schema_version must be {SCHEMA_VERSION}; old sessions cannot be resumed")
            return AgentState.model_validate(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
            # Do not include corrupt record contents or validation input values
            # in a model-visible diagnostic; keep the cause for local inspection.
            raise SessionLoadError(
                f"Session record could not be loaded ({type(error).__name__})"
            ) from error

    def exists(self, session_id: SessionId) -> bool:
        return self._path(session_id).is_file()
