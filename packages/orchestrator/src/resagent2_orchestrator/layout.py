"""Standard per-run data directory layout.

``RunLayout`` maps ``data_root`` + ``run_id`` to the per-run directories
(state, workspaces, attempts, scientific sessions, artifacts). It carries no
scheduling logic. The precedence for ``data_root`` is:

    explicit constructor argument > RESAGENT2_DATA_ROOT > .resagent2/data
"""

from __future__ import annotations

import os
from pathlib import Path


class RunLayout:
    """Resolve the per-run data directories under one ``data_root``."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()

    @classmethod
    def from_env(cls) -> "RunLayout":
        return cls(os.environ.get("RESAGENT2_DATA_ROOT", ".resagent2/data"))

    def run_dir(self, run_id: str) -> Path:
        return self.data_root / "runs" / run_id

    def state_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "state"

    def workspace_dir(self, run_id: str, workspace_id: str) -> Path:
        base = self.run_dir(run_id) / "workspaces"
        target = (base / workspace_id).resolve()
        if not target.is_relative_to(base.resolve()):
            raise ValueError(
                f"workspace_id escapes the workspaces directory: {workspace_id!r}"
            )
        return target

    def workspace_meta_path(self, run_id: str, workspace_id: str) -> Path:
        """Path of the materialization metadata file for one managed workspace."""
        return self.workspace_dir(run_id, workspace_id) / "workspace.json"

    def workspace_repo_dir(self, run_id: str, workspace_id: str) -> Path:
        return self.workspace_dir(run_id, workspace_id) / "repo"

    def attempt_dir(self, run_id: str, task_id: str, attempt_number: int) -> Path:
        return (
            self.run_dir(run_id)
            / "attempts"
            / task_id
            / f"attempt_{attempt_number}"
        )

    def scientific_sessions_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "scientific" / "sessions"

    def artifacts_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "artifacts"
