"""Resolve a workspace's declared source into a usable repository worktree."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sciagent_contracts import WorkspaceSourceKind, WorkspaceSpec
from .process import run_process
from .git import (
    GitWorkspaceError, configured_git_filters, disabled_git_filters,
    fixed_git_command, fixed_git_environment, require_supported_git_filters,
)
from sciagent_runtime.budget import DeadlineExceededError


class RepoMaterializerError(ValueError):
    """Raised when a repository source is missing, conflicting, or unusable."""


@dataclass(frozen=True, slots=True)
class MaterializedRepo:
    """A resolved repository worktree plus its content identity."""

    repo_path: Path
    commit: str
    source: str


_METADATA_FILENAME = "workspace.json"


def _git_commit(repo_path: Path) -> str:
    """Return the HEAD commit of a repository, or ``""`` when unavailable."""
    try:
        result = run_process(
            fixed_git_command(["-C", str(repo_path), "rev-parse", "HEAD"]),
            env=fixed_git_environment(),
            text=True,
            capture_output=True,
            timeout=20,
        )
    except DeadlineExceededError:
        raise
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _is_git_worktree_root(path: Path) -> bool:
    """Require the repository root, including repositories without a HEAD."""
    if not path.is_dir():
        return False
    try:
        result = run_process(
            fixed_git_command(["-C", str(path), "rev-parse", "--show-toplevel"]),
            env=fixed_git_environment(),
            text=True,
            capture_output=True,
            timeout=20,
        )
    except DeadlineExceededError:
        raise
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and Path(result.stdout.strip()).resolve() == path.resolve()


def _has_content(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return True


def _normalize_source(source: str) -> str:
    """Normalize a repo URL or local path for source-identity comparison."""
    value = (source or "").strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value and "://" not in value:
        value = str(Path(value).expanduser().resolve())
    return value


def _default_metadata_path(workspace: Path) -> Path:
    """Metadata lives one level above the managed repo (workspace.json)."""
    return workspace.parent / _METADATA_FILENAME


def _read_metadata(metadata_path: Path) -> dict | None:
    if not metadata_path.is_file():
        return None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_metadata(
    metadata_path: Path, source_kind: str, location: str, commit: str
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {"source_kind": source_kind, "location": location, "commit": commit}
        ),
        encoding="utf-8",
    )


class RepoMaterializer:
    """Resolve one ``WorkspaceSpec`` into a repository worktree.

    ``workspace`` is the directory that becomes (or already is) the repository
    root. For managed kinds (GIT/COPY/GENERATED) a small ``workspace.json``
    records which source materialized the workspace, so a retry verifies it is
    reusing the same repository rather than silently adopting a leftover from
    another one. LOCAL simply binds the external directory in place and writes
    no metadata (the external path is already the source).
    """

    def __init__(self, *, clone_timeout_seconds: int = 300) -> None:
        self.clone_timeout_seconds = clone_timeout_seconds

    def materialize(
        self,
        *,
        workspace: Path,
        source: WorkspaceSpec,
        metadata_path: Path | None = None,
    ) -> MaterializedRepo:
        """Return the workspace repository for the declared source.

        ``WorkspaceSpec`` already guarantees ``location`` for GIT/LOCAL/COPY and
        forbids it for GENERATED, so no redundant check is done here.
        """
        workspace = Path(workspace).expanduser().resolve()
        kind = source.source_kind
        if kind == WorkspaceSourceKind.GIT:
            return self._clone(
                workspace, source.location, metadata_path or _default_metadata_path(workspace)
            )
        if kind == WorkspaceSourceKind.COPY:
            return self._copy(
                workspace, source.location, metadata_path or _default_metadata_path(workspace)
            )
        if kind == WorkspaceSourceKind.GENERATED:
            return self._generate(workspace, metadata_path or _default_metadata_path(workspace))
        if kind == WorkspaceSourceKind.LOCAL:
            return self._bind(source.location)
        raise RepoMaterializerError(f"unsupported workspace source kind: {kind!r}")

    def _verify_source(
        self,
        workspace: Path,
        metadata_path: Path,
        expected_kind: str,
        expected_location: str,
    ) -> None:
        metadata = _read_metadata(metadata_path)
        if metadata is None:
            raise RepoMaterializerError(
                f"workspace at {workspace} has no materialization metadata; "
                f"cannot verify it came from {expected_location!r}"
            )
        actual_kind = metadata.get("source_kind")
        if actual_kind != expected_kind:
            raise RepoMaterializerError(
                f"workspace was materialized as {actual_kind!r}, not {expected_kind!r}"
            )
        if _normalize_source(metadata.get("location", "")) != _normalize_source(
            expected_location
        ):
            raise RepoMaterializerError(
                f"workspace source {metadata.get('location')!r} does not match "
                f"requested {expected_location!r}"
            )

    def _stage(self, workspace: Path, builder) -> None:
        """Materialize into a sibling staging dir, then atomically promote it.

        A failed clone/copy/generate only ever leaves the staging directory,
        which is removed on error, so a retry never sees a half-built workspace
        (ADR-0011 §7.2).
        """
        workspace.parent.mkdir(parents=True, exist_ok=True)
        staging = workspace.parent / f".{workspace.name}.sciagent-staging"
        if staging.exists():
            shutil.rmtree(staging)
        try:
            builder(staging)
            if workspace.exists() and not _has_content(workspace):
                workspace.rmdir()
            os.replace(staging, workspace)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _clone(
        self, workspace: Path, repo_url: str, metadata_path: Path
    ) -> MaterializedRepo:
        if _is_git_worktree_root(workspace):
            self._verify_source(
                workspace, metadata_path, WorkspaceSourceKind.GIT.value, repo_url
            )
            return MaterializedRepo(workspace, _git_commit(workspace), "git")
        if _has_content(workspace):
            raise RepoMaterializerError(
                f"cannot clone into non-empty workspace: {workspace}"
            )

        def build(staging: Path) -> None:
            result = run_process(
                fixed_git_command([
                    "clone", "--no-checkout", "--template=", "--depth", "1",
                    "--single-branch", repo_url, str(staging),
                ]),
                env=fixed_git_environment(),
                text=True,
                capture_output=True,
                timeout=self.clone_timeout_seconds,
            )
            if result.returncode != 0:
                raise RepoMaterializerError(
                    f"git clone failed: {(result.stderr or '').strip() or 'unknown error'}"
                )
            commit = _git_commit(staging)
            if commit:
                def run(arguments, *, accepted=(0,), env=None):
                    outcome = run_process(
                        fixed_git_command(arguments), cwd=staging,
                        env=fixed_git_environment(env), text=True, capture_output=True,
                        timeout=self.clone_timeout_seconds,
                    )
                    if outcome.returncode not in accepted:
                        raise RepoMaterializerError(
                            (outcome.stderr or "").strip() or "git checkout preparation failed"
                        )
                    return outcome

                # Only this new staging index is initialized; source/user state is untouched.
                run(["read-tree", "HEAD"])
                paths = run(["ls-files", "--cached", "-z"]).stdout.split("\0")
                filters = configured_git_filters(run)
                try:
                    require_supported_git_filters(
                        run, [path for path in paths if path], filters,
                        cached=True, direction="smudge",
                    )
                except GitWorkspaceError as error:
                    raise RepoMaterializerError(str(error)) from error
                run(["reset", "--hard", "HEAD"], env=disabled_git_filters(filters))
            # Persist metadata BEFORE the atomic rename, so a crash can only
            # leave "metadata without repo" (recoverable), never "repo without
            # metadata" (unrecoverable).
            _write_metadata(
                metadata_path,
                WorkspaceSourceKind.GIT.value,
                repo_url,
                _git_commit(staging),
            )

        self._stage(workspace, build)
        commit = _git_commit(workspace)
        return MaterializedRepo(workspace, commit, "git")

    def _copy(
        self, workspace: Path, source: str, metadata_path: Path
    ) -> MaterializedRepo:
        src = Path(source).expanduser().resolve()
        if not _is_git_worktree_root(src):
            raise RepoMaterializerError(f"COPY source must be a Git repository root: {src}")
        git_metadata = src / ".git"
        if git_metadata.is_file() or git_metadata.is_symlink():
            raise RepoMaterializerError(
                "COPY requires an independent .git directory; linked worktrees "
                "and external Git metadata are not supported"
            )
        if _is_git_worktree_root(workspace):
            self._verify_source(
                workspace, metadata_path, WorkspaceSourceKind.COPY.value, source
            )
            return MaterializedRepo(workspace, _git_commit(workspace), "copy")
        if _has_content(workspace):
            raise RepoMaterializerError(
                f"cannot copy into non-empty workspace: {workspace}"
            )
        def build(staging: Path) -> None:
            shutil.copytree(str(src), str(staging), symlinks=True)
            _write_metadata(
                metadata_path,
                WorkspaceSourceKind.COPY.value,
                source,
                _git_commit(staging),
            )

        self._stage(workspace, build)
        commit = _git_commit(workspace)
        return MaterializedRepo(workspace, commit, "copy")

    def _bind(self, source: str) -> MaterializedRepo:
        repo = Path(source).expanduser().resolve()
        if not _is_git_worktree_root(repo):
            raise RepoMaterializerError(
                f"LOCAL source must be a Git repository root: {repo}"
            )
        return MaterializedRepo(repo, _git_commit(repo), "local")

    def _generate(self, workspace: Path, metadata_path: Path) -> MaterializedRepo:
        if _is_git_worktree_root(workspace):
            self._verify_source(
                workspace, metadata_path, WorkspaceSourceKind.GENERATED.value, ""
            )
            return MaterializedRepo(workspace, _git_commit(workspace), "generated")
        if _has_content(workspace):
            raise RepoMaterializerError(
                f"cannot generate into non-empty workspace: {workspace}"
            )

        def build(staging: Path) -> None:
            staging.mkdir(parents=True, exist_ok=True)
            result = run_process(
                fixed_git_command(["init", "--template=", "-q", str(staging)]),
                env=fixed_git_environment(),
                text=True,
                capture_output=True,
                timeout=self.clone_timeout_seconds,
            )
            if result.returncode != 0:
                raise RepoMaterializerError(
                    f"git init failed: {(result.stderr or '').strip() or 'unknown error'}"
                )
            _write_metadata(metadata_path, WorkspaceSourceKind.GENERATED.value, "", "")

        self._stage(workspace, build)
        return MaterializedRepo(workspace, "", "generated")
