"""ResearchRun persistence with in-memory and atomic JSON stores."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol

from resagent2_contracts import RunId

from .models import ResearchRun


class RunStore(Protocol):
    """Persistence boundary for complete ResearchRun snapshots."""

    def save(self, run: ResearchRun) -> None:
        """Atomically persist the latest run snapshot."""

    def load(self, run_id: RunId) -> ResearchRun:
        """Load and validate the latest run snapshot."""

    def exists(self, run_id: RunId) -> bool:
        """Return whether a run already exists."""


class InMemoryRunStore:
    """Deep-copying RunStore used by deterministic scheduler tests."""

    def __init__(self) -> None:
        self._runs: dict[str, ResearchRun] = {}

    def save(self, run: ResearchRun) -> None:
        self._runs[run.run_id] = run.model_copy(deep=True)

    def load(self, run_id: RunId) -> ResearchRun:
        return self._runs[run_id].model_copy(deep=True)

    def exists(self, run_id: RunId) -> bool:
        return run_id in self._runs


class JsonRunStore:
    """Atomic one-file-per-run JSON persistence suitable for local recovery."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: RunId) -> Path:
        return self.root / f"{run_id}.json"

    def save(self, run: ResearchRun) -> None:
        destination = self._path(run.run_id)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=f".{run.run_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(run.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def load(self, run_id: RunId) -> ResearchRun:
        return ResearchRun.model_validate_json(
            self._path(run_id).read_text(encoding="utf-8")
        )

    def exists(self, run_id: RunId) -> bool:
        return self._path(run_id).is_file()
