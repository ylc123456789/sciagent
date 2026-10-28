"""Stable Session names and content-independent artifact identities.

The SessionId body allows 128 characters. Hash complete identities rather than
truncating identifying suffixes; readable prefixes are only for diagnostics.
"""

import hashlib
from uuid import uuid4

from .models import ArtifactId, RunId, SessionId, TaskId


def task_session_id(run_id: RunId, task_id: TaskId, attempt_number: int) -> SessionId:
    """Name one task attempt in a shared store, independently of ID length."""
    identity = f"{run_id}\0{task_id}\0{attempt_number}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    readable = f"{run_id}_{task_id}_{attempt_number}"
    return f"session_{readable[:95]}_{digest}"


def new_artifact_id() -> ArtifactId:
    """Allocate an identity before content exists; SHA256 verifies frozen bytes."""
    return f"artifact_{uuid4().hex}"
