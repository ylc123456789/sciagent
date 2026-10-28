"""Small immutable input/control records, separate from owner-written Run state.

The UI may publish records while the Controller owns the Run. Only the owner
decides when they are consumed and persists consumed IDs with Run state.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, model_validator


class ControlRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    run_id: str
    kind: Literal["input", "interrupt"]
    text: str = ""
    created_at: datetime

    @model_validator(mode="after")
    def valid_text(self) -> ControlRecord:
        if self.kind == "input" and not self.text.strip():
            raise ValueError("user input must contain text")
        return self


class RunControlInbox:
    """Publish durable messages without loading or overwriting mutable Run state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _directory(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            raise ValueError("invalid run ID for input/control record")
        return self.root / run_id

    def submit(
        self, run_id: str, kind: Literal["input", "interrupt"], text: str = "",
        *, validate_run: Callable[[], None] | None = None,
    ) -> ControlRecord:
        directory = self._directory(run_id)
        # Validate before publishing anything, including the submission lock.
        ControlRecord(event_id="validation", run_id=run_id, kind=kind,
                      text=text, created_at=datetime.now(UTC))
        directory.mkdir(parents=True, exist_ok=True)
        with self._publication_lock(directory):
            # Recheck the producer's live Run status while terminal receiving
            # and publication are mutually excluded; a prior UI read can be stale.
            if validate_run is not None:
                validate_run()
            stamp = time.time_ns()
            record = ControlRecord(
                event_id=f"control_{stamp:020d}_{uuid4().hex}",
                run_id=run_id, kind=kind, text=text,
                created_at=datetime.fromtimestamp(stamp / 1_000_000_000, UTC),
            )
            destination = directory / f"{record.event_id}.json"
            temporary = directory / f".{record.event_id}.tmp"
            try:
                with temporary.open("x", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json())
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                if os.name == "posix":
                    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            finally:
                temporary.unlink(missing_ok=True)
        return record

    @contextmanager
    def _publication_lock(self, directory: Path, *, shared: bool = False):
        """Order concurrent publications, separate from the Run execution lock."""
        if os.name != "posix":
            yield
            return
        import fcntl
        with (directory / "submit.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def pending(self, run_id: str, consumed_ids: Iterable[str] = ()) -> list[ControlRecord]:
        directory = self._directory(run_id)
        consumed = set(consumed_ids)
        if not directory.exists():
            return []
        with self._publication_lock(directory, shared=True):
            return self._pending_locked(directory, run_id, consumed)

    def _pending_locked(self, directory: Path, run_id: str, consumed: set[str]):
        records = []
        for path in sorted(directory.glob("control_*.json")):
            if path.stem in consumed:
                continue
            record = ControlRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if record.run_id != run_id or record.event_id != path.stem:
                raise ValueError("input/control record identity does not match its path")
            records.append(record)
        return records

    @contextmanager
    def settlement(self, run_id: str, consumed_ids: Iterable[str] = ()):
        """Fence final receiving with publications and yield its pending snapshot.

        The owner commits terminal Run state before releasing this boundary.
        Producers must revalidate that state inside submit's validation callback.
        """
        directory = self._directory(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        with self._publication_lock(directory):
            yield self._pending_locked(directory, run_id, set(consumed_ids))
