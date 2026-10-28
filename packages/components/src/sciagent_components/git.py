"""Read-only Git observations used by deterministic module finalizers."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .workspace import WorkspaceBoundary, WorkspacePermissionError
from .process import run_process


class GitWorkspaceError(ValueError):
    """Raised when a workspace cannot provide an unambiguous Git baseline."""


_GIT_ENVIRONMENT = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES", "GIT_DISCOVERY_ACROSS_FILESYSTEM", "GIT_NAMESPACE",
    "GIT_CONFIG", "GIT_EXEC_PATH", "GIT_ATTR_SOURCE", "GIT_GRAFT_FILE",
    "GIT_REPLACE_REF_BASE", "GIT_NO_REPLACE_OBJECTS", "GIT_IMPLICIT_WORK_TREE",
    "GIT_PREFIX", "GIT_SHALLOW_FILE", "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS",
    "GIT_PAGER",
})


def fixed_git_environment(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Select Git state locally; accept only the component's explicit overrides."""
    environment = {
        name: value for name, value in os.environ.items()
        if name not in _GIT_ENVIRONMENT
        and not name.startswith(("GIT_CONFIG_", "GIT_TRACE"))
    }
    environment.update({
        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
        # Observing missing objects must not silently fetch through remote helpers.
        "GIT_NO_LAZY_FETCH": "1",
    })
    environment.update(overrides or {})
    return environment


def fixed_git_command(arguments: list[str]) -> list[str]:
    """Disable executable helpers for deterministic internal Git operations."""
    return [
        "git", "--no-pager", "-c", f"core.hooksPath={os.devnull}",
        "-c", "core.fsmonitor=false", "-c", "gc.auto=0",
        "-c", "maintenance.auto=false", *arguments,
    ]


def configured_git_filters(run: Callable) -> dict[str, dict[str, str]]:
    """Read effective driver commands without parsing command text as Git syntax."""
    names = run(
        ["config", "--null", "--name-only", "--get-regexp", r"^filter\."],
        accepted=(0, 1),
    ).stdout.split("\0")
    filters: dict[str, dict[str, str]] = {}
    for name in dict.fromkeys(names):
        if not name or not name.startswith("filter.") or "." not in name[7:]:
            continue
        driver, attribute = name[7:].rsplit(".", 1)
        if attribute not in {"clean", "smudge", "process", "required"}:
            continue
        value = run(["config", "--null", "--get", name], accepted=(0, 1)).stdout
        filters.setdefault(driver, {})[attribute] = value.removesuffix("\0")
    return filters


def disabled_git_filters(filters: dict[str, dict[str, str]]) -> dict[str, str]:
    """Use structured config so even driver names containing '=' remain literal."""
    # Selected external commands have already been rejected by preflight.
    # A driver used only in the opposite direction must not require an empty command.
    entries = [
        (f"filter.{driver}.{attribute}", "false" if attribute == "required" else "")
        for driver in filters
        for attribute in ("clean", "smudge", "process", "required")
    ]
    environment = {"GIT_CONFIG_COUNT": str(len(entries))}
    for index, (name, value) in enumerate(entries):
        environment[f"GIT_CONFIG_KEY_{index}"] = name
        environment[f"GIT_CONFIG_VALUE_{index}"] = value
    return environment


def require_supported_git_filters(
    run: Callable, paths: list[str], filters: dict[str, dict[str, str]], *,
    env: dict[str, str] | None = None, cached: bool = False, direction: str = "clean",
) -> None:
    """Reject required external normalization instead of storing different bytes."""
    if not filters:
        return
    for offset in range(0, len(paths), 256):
        arguments = ["check-attr", "--all", "-z"]
        if cached:
            arguments.append("--cached")
        attributes = run(
            [*arguments, "--", *paths[offset:offset + 256]], env=env,
        ).stdout.split("\0")
        for index in range(0, len(attributes) - 1, 3):
            path, attribute, driver = attributes[index:index + 3]
            if attribute != "filter":
                continue
            commands = filters.get(driver, {})
            # --all omits unspecified attributes. set/unset can still denote
            # either a boolean attribute or a literal configured driver name.
            if commands.get("process") or commands.get(direction):
                raise GitWorkspaceError(
                    f"Fixed Git observations do not support external {direction}/process "
                    f"filter {driver!r} selected or ambiguously named for {path!r}; "
                    "repository programs were not executed"
                )


@dataclass(frozen=True, slots=True)
class GitBaseline:
    """A content snapshot of the working directory at the start of one Attempt.

    ``tree_hash`` captures the complete visible working-directory state,
    including tracked and untracked files. Comparing it with a second tree
    snapshot yields only the increment produced by the current Attempt, without
    committing, touching the real index, or rolling anything back.
    """

    tree_hash: str

    def to_memory(self) -> dict:
        """Persist the Attempt baseline without copying workspace contents."""
        return {"kind": "git", "tree_hash": self.tree_hash}

    @classmethod
    def from_memory(cls, value: object) -> "GitBaseline":
        """Require a persisted Git baseline when resuming a Coding Attempt."""
        if not isinstance(value, dict) or value.get("kind") != "git":
            raise ValueError("missing Git baseline")
        tree_hash = value.get("tree_hash")
        if not isinstance(tree_hash, str) or not tree_hash:
            raise ValueError("Git baseline is missing tree_hash")
        return cls(tree_hash=tree_hash)


class GitWorkspace:
    """Observe one existing repository without mutating its Git state."""

    def __init__(self, boundary: WorkspaceBoundary) -> None:
        self.boundary = boundary
        top = self._run(["rev-parse", "--show-toplevel"]).stdout.strip()
        if Path(top).resolve() != boundary.root:
            raise GitWorkspaceError("workspace root must be the Git repository root")

    def _run(
        self,
        arguments: list[str],
        *,
        accepted: tuple[int, ...] = (0,),
        env: dict[str, str] | None = None,
        text: bool = True,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        try:
            result = run_process(
                fixed_git_command(arguments),
                cwd=self.boundary.root,
                text=text,
                capture_output=True,
                check=False,
                env=fixed_git_environment(env),
            )
        except (subprocess.SubprocessError, UnicodeError) as error:
            raise GitWorkspaceError(str(error)) from error
        if result.returncode not in accepted:
            message = result.stderr
            if isinstance(message, bytes):
                message = message.decode("utf-8", errors="replace")
            raise GitWorkspaceError(message.strip() or "git command failed")
        return result

    @staticmethod
    def _visible(path: str) -> bool:
        ignored_parts = {
            ".sciagent",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }
        return not ignored_parts.intersection(Path(path).parts)

    def _write_tree(self) -> str:
        """Return a tree hash of the complete visible working-directory state.

        Uses an empty temporary index and adds only authorized files. The real index,
        branch and working tree are untouched.
        """
        # A gitlink lists only a submodule directory, not its mutable contents.
        # Never certify a complete scoped snapshot while silently omitting it.
        for entry in self._run(["ls-files", "--stage", "-z"]).stdout.split("\0"):
            if not entry:
                continue
            metadata, path = entry.split("\t", 1)
            if metadata.split(" ", 1)[0] != "160000" or not self._visible(path):
                continue
            try:
                self.boundary.resolve_read_directory(path, traverse=True)
            except WorkspacePermissionError:
                continue
            except OSError:
                # An authorized uninitialized/missing submodule is unsupported too.
                pass
            raise GitWorkspaceError(
                f"Git snapshots do not support submodule contents in the readable scope: {path}"
            )
        fd, tmp_index = tempfile.mkstemp(prefix="sciagent-index-")
        os.close(fd)
        try:
            index_env = {"GIT_INDEX_FILE": tmp_index}
            snapshot_paths = self._run(
                ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            ).stdout.split("\0")
            self._run(["read-tree", "--empty"], env=index_env)
            visible_files = []
            for path in snapshot_paths:
                if not path or not self._visible(path) or not self.boundary.allows_read(path):
                    continue
                try:
                    self.boundary.resolve_read_file(path)
                except (ValueError, OSError):
                    continue
                visible_files.append(path)
            filters = configured_git_filters(self._run)
            # Match add's empty index, including its fallback for absent attributes.
            require_supported_git_filters(self._run, visible_files, filters, env=index_env)
            index_env.update(disabled_git_filters(filters))
            for offset in range(0, len(visible_files), 256):
                literal_paths = [
                    f":(literal){path}"
                    for path in visible_files[offset : offset + 256]
                ]
                self._run(["add", "--", *literal_paths], env=index_env)
            return self._run(["write-tree"], env=index_env).stdout.strip()
        finally:
            os.unlink(tmp_index)

    def snapshot(self) -> GitBaseline:
        """Capture the current working-directory content as an Attempt baseline."""
        return GitBaseline(tree_hash=self._write_tree())

    # ── Attempt-baseline-relative observations ────────────────────────────────

    def changed_paths_since(self, baseline: GitBaseline) -> list[str]:
        """Return paths whose content differs from ``baseline`` (this Attempt)."""
        current_tree = self._write_tree()
        changed = self._run(
            ["diff", "--no-ext-diff", "--no-textconv", "--name-only", "-z", baseline.tree_hash, current_tree]
        ).stdout.split("\0")
        return sorted(
            path
            for path in changed
            if path and self._visible(path) and self.boundary.allows_read(path)
        )

    def deleted_paths_since(self, baseline: GitBaseline) -> list[str]:
        return [
            path
            for path in self.changed_paths_since(baseline)
            if not (self.boundary.root / path).exists()
        ]

    def diff_bytes_since(self, baseline: GitBaseline) -> bytes:
        """Return exact Git patch bytes, including non-UTF-8 textual hunks."""
        current_tree = self._write_tree()
        changed = self._run(
            ["diff", "--no-ext-diff", "--no-textconv", "--name-only", "-z", baseline.tree_hash, current_tree]
        ).stdout.split("\0")
        visible = [
            path
            for path in changed
            if path and self._visible(path) and self.boundary.allows_read(path)
        ]
        patches: list[bytes] = []
        for offset in range(0, len(visible), 256):
            literal_paths = [
                f":(literal){path}" for path in visible[offset : offset + 256]
            ]
            patches.append(
                self._run(
                    ["diff", "--no-ext-diff", "--no-textconv", "--binary", baseline.tree_hash, current_tree, "--", *literal_paths],
                    text=False,
                ).stdout
            )
        return b"".join(patches)

    def diff_since(self, baseline: GitBaseline) -> str:
        """Return a strict UTF-8 text view; authoritative patches use bytes."""
        try:
            return self.diff_bytes_since(baseline).decode("utf-8")
        except UnicodeError as error:
            raise GitWorkspaceError(str(error)) from error

    def write_patch_since(self, baseline: GitBaseline, path: Path) -> str:
        """Write the Attempt-increment diff to an absolute ``path``."""
        patch = self.diff_bytes_since(baseline)
        if not patch:
            raise GitWorkspaceError("workspace has no code diff since the Attempt baseline")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(patch)
        return str(destination)
