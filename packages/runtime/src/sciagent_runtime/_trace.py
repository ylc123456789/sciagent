"""Best-effort storage for optional private model traces."""

import json
import logging
import os
from pathlib import Path


def write_trace_record(directory: Path, record: dict) -> None:
    """Keep diagnostic storage failures outside the model execution result."""
    _write_trace_record(directory, record, filename="llm_traces.jsonl", label="LLM")


def write_service_trace_record(directory: Path, record: dict) -> None:
    """Keep external service diagnostics separate from authoritative model usage."""
    _write_trace_record(directory, record, filename="service_requests.jsonl", label="service")


def _write_trace_record(directory: Path, record: dict, *, filename: str, label: str) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        line = json.dumps(record, ensure_ascii=False, default=str)
        path = directory / filename
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            os.chmod(path, 0o600)
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError) as error:
        # Paths, exception messages and trace contents may contain secrets.
        logging.getLogger(__name__).warning(
            "Could not write optional %s trace (%s).", label, type(error).__name__,
        )
