"""Physical filesystem boundaries derived from WorkspaceGrant."""

from __future__ import annotations

import base64
import hashlib
import heapq
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from resagent2_contracts import WorkspaceGrant


class WorkspacePermissionError(PermissionError):
    """Raised before a path can escape or exceed a workspace grant."""


class DeleteChangedError(WorkspacePermissionError):
    """A prepared deletion no longer describes the filesystem."""


class DeleteInterruptedError(OSError):
    """A deletion stopped after making partial progress."""

    def __init__(self, reason: str, deleted: list[str], remaining: list[str]) -> None:
        super().__init__(reason)
        self.deleted = deleted
        self.remaining = remaining


def _normalize_relative(path: str, *, allow_root: bool = False) -> str:
    value = path.strip().replace("\\", "/")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or "\x00" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
    ):
        raise WorkspacePermissionError("path must be relative and cannot contain '..'")
    normalized = posix.as_posix()
    if normalized == "." and not allow_root:
        raise WorkspacePermissionError("path must identify a workspace entry")
    return normalized


FILESYSTEM_PATH_DESCRIPTION = (
    "Authorized absolute filesystem path, or a path relative to the fixed default workspace. "
    "Relative paths do not follow shell cd; agents without a default require absolute paths. "
    "A path grants no access."
)
FILESYSTEM_PATH_GUIDANCE = FILESYSTEM_PATH_DESCRIPTION + " "
MAIN_WORKSPACE_PATH_DESCRIPTION = (
    FILESYSTEM_PATH_DESCRIPTION + " This tool is limited to the main workspace grant."
)
MAIN_WORKSPACE_PATH_GUIDANCE = MAIN_WORKSPACE_PATH_DESCRIPTION + " "


def _tool_absolute_path(value: str, *, relative_root: Path | None, operation: str) -> Path:
    """Interpret public tool addresses without resolving away lexical aliases."""
    text = value.strip().replace("\\", "/")
    path = Path(text)
    if (not text or "\x00" in text or ".." in path.parts
            or PureWindowsPath(text).drive):
        requirement = "an absolute path" if relative_root is None else "an authorized path"
        raise WorkspacePermissionError(f"{operation} requires {requirement} without '..' or a Windows drive")
    if path.is_absolute():
        return path
    if relative_root is None:
        raise WorkspacePermissionError(f"{operation} requires an absolute path without '..'")
    return relative_root / _normalize_relative(text, allow_root=True)


@dataclass(frozen=True)
class _WorkspaceAddress:
    """One trusted root's declared spelling and stable physical address.

    Only the grant root is resolved. User suffixes retain their lexical spelling
    so denied paths and final-link deletion still use the original checks.
    """

    declared: Path
    root: Path

    @classmethod
    def from_root(cls, value: str | Path) -> _WorkspaceAddress:
        declared = Path(value).expanduser()
        if not declared.is_absolute():
            declared = Path.cwd() / declared
        # Resolve before collapsing '..': a symlink followed by its parent has
        # filesystem semantics that abspath/normpath would silently change.
        return cls(declared, declared.resolve(strict=False))

    def match(self, path: Path) -> tuple[int, str] | None:
        matches = [root for root in (self.declared, self.root) if path.is_relative_to(root)]
        if not matches:
            return None
        root = max(matches, key=lambda item: len(item.parts))
        # A root that was absent or an ordinary directory may later become a
        # link too. Recheck the matched root before rebuilding a read boundary;
        # another granted target must not legitimize crossing this pinned root.
        if root.resolve(strict=False) != self.root:
            raise WorkspacePermissionError("workspace root address changed; rebuild its boundary")
        return len(root.parts), path.relative_to(root).as_posix()


def _matches(relative: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        prefix == "." or relative == prefix or relative.startswith(f"{prefix}/")
        for prefix in prefixes
    )


def _version(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns]


class DirectoryChangedError(ValueError):
    """A paged directory observation changed; restart its listing."""


def _directory_scope(grants: list[WorkspaceGrant]) -> str:
    """Bind pagination to read authority, without publishing its private rules."""
    scopes = [{"root": os.path.abspath(os.path.expanduser(grant.root)),
               "read": sorted(grant.access.read_paths),
               "denied": sorted(grant.access.denied_paths)} for grant in grants]
    raw = json.dumps(sorted(scopes, key=lambda value: json.dumps(value, sort_keys=True)),
                     sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _directory_observation(path: str, resolved: Path, scope: str) -> str:
    info = resolved.stat()
    # Entry creation/removal/renaming changes the directory itself; editing an
    # existing file's contents is intentionally not a listing-version change.
    version = [info.st_dev, info.st_ino, info.st_mode, info.st_mtime_ns, info.st_ctime_ns]
    # Some filesystems give several mutations the same timestamp. Include a
    # bounded-memory, order-independent digest of this directory's names and
    # inode identities, without stat-ing descendants or reading file contents.
    count, digest = 0, 0
    with os.scandir(resolved) as entries:
        for entry in entries:
            identity = os.fsencode(entry.name) + b"\0" + str(entry.inode()).encode()
            digest = (digest + int.from_bytes(hashlib.sha256(identity).digest(), "big")) % (1 << 256)
            count += 1
    raw = json.dumps([path, str(resolved), version, count, f"{digest:064x}", scope], ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _listing_cursor(observation: str, after: str) -> str:
    raw = json.dumps({"v": 1, "observation": observation, "after": after},
                     ensure_ascii=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _read_listing_cursor(cursor: str, observation: str) -> str:
    try:
        if not isinstance(cursor, str) or len(cursor) > 8192:
            raise ValueError("invalid cursor")
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        value = json.loads(raw)
        if (not isinstance(value, dict) or set(value) != {"v", "observation", "after"}
                or type(value["v"]) is not int or value["v"] != 1
                or not isinstance(value["observation"], str)
                or not isinstance(value["after"], str)
                or not value["after"] or "/" in value["after"] or "\x00" in value["after"]):
            raise ValueError("invalid cursor")
    except (ValueError, UnicodeError, TypeError) as error:
        raise ValueError("Invalid directory cursor; restart the directory listing") from error
    if value["observation"] != observation:
        raise DirectoryChangedError("Directory or read scope changed; restart the directory listing")
    return value["after"]


def _directory_names(directory: Path, after: str | None, limit: int) -> list[str]:
    # Only names in this directory are scanned. The heap bounds retained names;
    # no descendant metadata or file contents are read to find a page.
    with os.scandir(directory) as entries:
        return heapq.nsmallest(limit, (entry.name for entry in entries
                                      if after is None or entry.name > after))


def _listing_name_requires_exact_path(name: str) -> bool:
    return name != name.strip() or "\\" in name or bool(PureWindowsPath(name).drive)


def _directory_entry(boundary: Any, path: str, name: str) -> dict[str, Any] | None:
    child = str(Path(path) / name)
    status = "available"
    error = None
    unsupported = False
    try:
        # Authorize the actual directory entry before applying the public text
        # path convention. Normalization must not authorize a different file,
        # hide a readable name, or reveal a denied symlink target.
        exact = boundary._resolve_listing_entry(child)
        kind = "directory" if exact.is_dir() else "file"
    except WorkspacePermissionError:
        return None
    except OSError as failure:
        kind, status, error = "entry", "read_error", type(failure).__name__
    else:
        try:
            addressed = (boundary.resolve_read_directory(child, traverse=True) if kind == "directory"
                         else boundary.resolve_read_file(child))
        except WorkspacePermissionError:
            unsupported = True
        except OSError as failure:
            # A noncanonical name may normalize to a missing or different kind
            # of entry. It is a path-language limitation, not missing metadata.
            unsupported = _listing_name_requires_exact_path(name)
            if not unsupported:
                status, error = "read_error", type(failure).__name__
        else:
            unsupported = addressed != exact or _listing_name_requires_exact_path(name)
    node = {"name": name, "type": kind, "location": {"path": child},
            "status": status, "content_nature": "directory_fact" if kind == "directory" else "live_file",
            "children": [], "expansion": {"state": "not_expanded" if kind == "directory" else "complete"}}
    if error is not None:
        node["error"] = error
    lexical = Path(child) if Path(child).is_absolute() else boundary.root / child
    if lexical.is_symlink():
        node["is_symlink"] = True
    try:
        child.encode("utf-8")
    except UnicodeEncodeError:
        unsupported = True
        error = "Filename is not valid UTF-8; current text path tools cannot address it"
        node["name"] = name.encode("utf-8", errors="backslashreplace").decode("utf-8")
    if unsupported:
        # Retain discoverable byte identity, never an address which the tools
        # would silently reinterpret as another file. This grants no new access.
        node["location"] = {"path_bytes_base64": base64.b64encode(os.fsencode(child)).decode()}
        node["status"] = "unsupported_filename"
        node["expansion"] = {"state": "complete"}
        node["error"] = error or "Current text path tools cannot address this filename exactly"
    return node


def _list_directory(boundary: Any, path: str, *, scope: str,
                    max_entries: int, cursor: str | None) -> dict[str, Any]:
    if type(max_entries) is not int or not 1 <= max_entries <= 2000:
        raise ValueError("max_entries must be an integer between 1 and 2000")
    directory = boundary.resolve_read_directory(path, traverse=True)
    observation = _directory_observation(path, directory, scope)
    after = _read_listing_cursor(cursor, observation) if cursor is not None else None
    nodes: list[dict[str, Any]] = []
    last_visible = None
    scanned = 0
    scan_limit = max(2000, max_entries * 4)
    partial, exhausted = False, False
    continuation = after
    while scanned < scan_limit:
        batch_limit = min(2001, scan_limit - scanned)
        names = _directory_names(directory, after, batch_limit)
        if not names:
            exhausted = True
            break
        for name in names:
            after = name
            scanned += 1
            node = _directory_entry(boundary, path, name)
            if node is None:
                continue
            if len(nodes) == max_entries:
                partial, continuation = True, last_visible
                break
            nodes.append(node)
            last_visible = name
        if partial:
            break
        if len(names) < batch_limit:
            exhausted = True
            break
    if not partial and not exhausted:
        # An empty visible page can still make progress through many denied
        # or escaping candidates. Never loop without bound to fill a page.
        partial, continuation = True, after
    if _directory_observation(path, boundary.resolve_read_directory(path, traverse=True), scope) != observation:
        raise DirectoryChangedError("Directory changed during listing; restart the directory listing")
    expansion = {"state": "partial" if partial else "complete"}
    if partial:
        assert continuation is not None
        expansion["next_cursor"] = _listing_cursor(observation, continuation)
    return {"name": directory.name, "type": "directory", "location": {"path": path},
            "status": "available", "content_nature": "directory_fact",
            "children": nodes[:max_entries], "expansion": expansion,
            "observation": {"directory_fingerprint": observation}}


class ReadOnlyWorkspaceBoundary:
    """Read absolute paths through the most specific trusted workspace grant.

    Boundaries are created when a file or directory is read, so missing output
    roots do not prevent the parent Agent from handling other work. A denied or
    missing specific root never falls back to a broader grant.
    """

    def __init__(self, grants: list[WorkspaceGrant], *, relative_root: str | Path | None = None) -> None:
        self.relative_root = (_WorkspaceAddress.from_root(relative_root).root
                              if relative_root is not None else None)
        self._grants = tuple(
            (address, grant.model_copy(update={"root": str(address.root)}, deep=True))
            for grant in grants for address in [_WorkspaceAddress.from_root(grant.root)]
        )

    @property
    def grants(self) -> tuple[WorkspaceGrant, ...]:
        """Canonical root descriptors; inspecting them cannot mutate authority."""
        return tuple(grant.model_copy(deep=True) for _, grant in self._grants)

    def _absolute_path(self, value: str, *, operation: str) -> Path:
        return _tool_absolute_path(value, relative_root=self.relative_root, operation=operation)

    def _matching_grants(self, path: Path) -> list[tuple[_WorkspaceAddress, WorkspaceGrant, str]]:
        matches = []
        for address, grant in self._grants:
            match = address.match(path)
            if match is not None:
                depth, relative = match
                matches.append((depth, address, grant, relative))
        if not matches:
            raise WorkspacePermissionError("path is outside the authorized read workspaces")
        depth = max(item[0] for item in matches)
        return [(address, grant, relative) for level, address, grant, relative in matches if level == depth]

    def _resolve_read_path(self, absolute_path: str, *, directory: bool,
                           traverse: bool = False, _listing_metadata: bool = False) -> Path:
        # Only internally generated directory candidates use exact spelling.
        # Public read/list inputs keep the existing path normalization rules.
        path = (Path(absolute_path) if _listing_metadata else
                self._absolute_path(absolute_path, operation="list_files" if directory else "read_file"))

        def resolve(grant: WorkspaceGrant, relative: str) -> Path:
            boundary = WorkspaceBoundary(grant)
            if _listing_metadata:
                return boundary._resolve_listing_entry(relative)
            return (boundary.resolve_read_directory(relative, traverse=traverse) if directory
                    else boundary.resolve_read_file(relative))

        resolved = None
        for _, grant, relative in self._matching_grants(path):
            resolved = resolve(grant, relative)
        assert resolved is not None
        # Recheck the actual target so an alias in a wide grant cannot bypass
        # a more specific grant, including differently spelled equal roots.
        targets = [(address.root, grant) for address, grant in self._grants
                   if resolved.is_relative_to(address.root)]
        if not targets:
            raise WorkspacePermissionError("resolved path is outside the authorized read workspaces")
        depth = max(len(root.parts) for root, _ in targets)
        for root, grant in targets:
            if len(root.parts) == depth:
                resolve(grant, resolved.relative_to(root).as_posix())
        return resolved

    def _resolve_listing_entry(self, path: str) -> Path:
        return self._resolve_read_path(path, directory=True, traverse=True, _listing_metadata=True)

    def resolve_read_file(self, absolute_path: str) -> Path:
        return self._resolve_read_path(absolute_path, directory=False)

    def resolve_read_directory(self, absolute_path: str, *, traverse: bool = False) -> Path:
        return self._resolve_read_path(absolute_path, directory=True, traverse=traverse)

    def list_directory(self, path: str, *, max_entries: int = 200,
                       cursor: str | None = None) -> dict[str, Any]:
        requested = self._absolute_path(path, operation="list_files")
        self._resolve_read_path(str(requested), directory=True, traverse=True)
        address, _, relative = self._matching_grants(requested)[0]
        absolute = str(address.root / relative)
        return _list_directory(self, absolute, scope=_directory_scope([grant for _, grant in self._grants]),
                               max_entries=max_entries, cursor=cursor)


class WorkspaceBoundary:
    """Resolve paths while enforcing one read/write/exclusion scope."""

    ignored_parts = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})

    def __init__(self, grant: WorkspaceGrant) -> None:
        self.grant = grant
        self._address = _WorkspaceAddress.from_root(grant.root)
        self.root = self._address.root.resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspacePermissionError("workspace root must be an existing directory")
        self.read_paths = tuple(grant.access.read_paths)
        self.write_paths = tuple(grant.access.write_paths)
        self.denied_paths = tuple(grant.access.denied_paths)

    def normalize_path(self, path: str, *, allow_root: bool = False) -> str:
        """Map a tool address to this grant's lexical relative namespace.

        This does not authorize access: callers must still apply lexical and
        resolved checks. Internal grant prefixes and snapshots stay relative.
        """
        absolute = _tool_absolute_path(path, relative_root=self.root, operation="filesystem tool")
        match = self._address.match(absolute)
        if match is not None:
            return _normalize_relative(match[1], allow_root=allow_root)
        raise WorkspacePermissionError("path is outside the main workspace root")

    def absolute_path(self, path: str, *, allow_root: bool = False) -> str:
        """Canonical root address, retaining lexical suffixes for permission checks."""
        return str(self.root / self.normalize_path(path, allow_root=allow_root))

    def _check_lexical(self, relative: str, *, write: bool = False, traverse: bool = False) -> None:
        parts = PurePosixPath(relative).parts
        if ".resagent2" in parts or (write and ".git" in parts):
            raise WorkspacePermissionError("path enters a runtime-reserved directory")
        if _matches(relative, self.denied_paths):
            raise WorkspacePermissionError("path is inside WorkspaceAccess.denied_paths")
        scope = self.write_paths if write else self.read_paths
        if _matches(relative, scope):
            return
        if traverse and any(_matches(prefix, (relative,)) for prefix in scope):
            return
        kind = "write" if write else "read"
        raise WorkspacePermissionError(f"path is outside the workspace {kind} scope")

    def allows_read(self, relative: str) -> bool:
        """Return whether a relative path is readable under the complete scope."""
        try:
            self._check_lexical(_normalize_relative(relative, allow_root=True))
        except WorkspacePermissionError:
            return False
        return True

    def _ensure_contained(self, path: Path) -> None:
        if not path.is_relative_to(self.root):
            raise WorkspacePermissionError("resolved path escapes workspace root")

    def _check_resolved(self, resolved: Path, *, write: bool = False, traverse: bool = False) -> None:
        self._ensure_contained(resolved)
        relative = resolved.relative_to(self.root).as_posix()
        self._check_lexical(relative, write=write, traverse=traverse)

    def _resolve_listing_entry(self, relative_path: str) -> Path:
        """Authorize metadata for an exact, internally formed directory entry."""
        candidate = Path(relative_path)
        if candidate.is_absolute() and candidate.is_relative_to(self.root):
            candidate = candidate.relative_to(self.root)
        relative = candidate.as_posix()
        if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise WorkspacePermissionError("listing candidate must stay inside its workspace")
        self._check_lexical(relative, traverse=True)
        resolved = (self.root / relative).resolve(strict=True)
        self._check_resolved(resolved, traverse=True)
        if resolved.is_dir():
            return resolved
        self._check_lexical(relative)
        self._check_resolved(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(relative)
        return resolved

    def resolve_read_file(self, relative_path: str) -> Path:
        relative = self.normalize_path(relative_path)
        self._check_lexical(relative)
        resolved = (self.root / relative).resolve(strict=True)
        self._ensure_contained(resolved)
        self._check_resolved(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(relative)
        return resolved

    def resolve_read_directory(self, relative_path: str = ".", *, traverse: bool = False) -> Path:
        relative = self.normalize_path(relative_path, allow_root=True)
        self._check_lexical(relative, traverse=traverse)
        resolved = (self.root / relative).resolve(strict=True)
        self._ensure_contained(resolved)
        self._check_resolved(resolved, traverse=traverse)
        if not resolved.is_dir():
            raise NotADirectoryError(relative)
        return resolved

    def resolve_write_file(self, relative_path: str, *, must_be_new: bool = False) -> Path:
        relative = self.normalize_path(relative_path)
        self._check_lexical(relative, write=True)
        candidate = self.root / relative
        resolved = candidate.resolve(strict=False)
        self._check_resolved(resolved, write=True)
        if candidate.exists() or candidate.is_symlink():
            if must_be_new:
                raise FileExistsError(relative)
            if not resolved.is_file():
                raise WorkspacePermissionError("write target must be a file")
        return resolved

    def resolve_system_write(self, relative_path: str) -> Path:
        """Resolve trusted framework output, separately from model file grants."""
        relative = _normalize_relative(relative_path)
        if not relative.startswith(".resagent2/"):
            raise WorkspacePermissionError("system output must be below .resagent2")
        candidate = (self.root / relative).resolve(strict=False)
        self._ensure_contained(candidate)
        if not candidate.is_relative_to(self.root / ".resagent2"):
            raise WorkspacePermissionError("system output escapes .resagent2")
        return candidate

    def list_directory(self, path: str = ".", *, max_entries: int = 200,
                       cursor: str | None = None) -> dict[str, Any]:
        absolute = self.absolute_path(path, allow_root=True)
        return _list_directory(self, absolute, scope=_directory_scope([self.grant]),
                               max_entries=max_entries, cursor=cursor)

    def iter_files(self, relative_path: str = ".") -> list[str]:
        start = self.resolve_read_directory(relative_path, traverse=True)
        files: list[str] = []
        for current, directories, names in os.walk(start, followlinks=False):
            current_path = Path(current)
            kept: list[str] = []
            for name in directories:
                if name in self.ignored_parts:
                    continue
                relative = (current_path / name).relative_to(self.root).as_posix()
                try:
                    self.resolve_read_directory(relative, traverse=True)
                except (OSError, WorkspacePermissionError):
                    continue
                kept.append(name)
            directories[:] = kept
            for name in names:
                relative = (current_path / name).relative_to(self.root).as_posix()
                try:
                    self.resolve_read_file(relative)
                except (OSError, WorkspacePermissionError):
                    continue
                files.append(relative)
        return sorted(files)

    def relative(self, path: Path) -> str:
        resolved = path.resolve(strict=True)
        self._ensure_contained(resolved)
        return resolved.relative_to(self.root).as_posix()

    def _delete_target(self, relative_path: str) -> Path:
        relative = _normalize_relative(relative_path)
        self._check_lexical(relative, write=True)
        lexical = self.root / relative
        parent = lexical.parent.resolve(strict=True)
        self._ensure_contained(parent)
        target = parent / lexical.name
        self._check_resolved(target, write=True)
        return target

    def prepare_delete(self, relative_path: str, *, recursive: bool = False) -> dict[str, Any]:
        """Snapshot exact entries without following the final symlink."""
        relative = self.normalize_path(relative_path)
        entries: list[dict[str, Any]] = []

        def visit(path: str) -> None:
            target = self._delete_target(path)
            info = target.lstat()
            if stat.S_ISLNK(info.st_mode):
                kind = "symlink"
            elif stat.S_ISREG(info.st_mode):
                kind = "file"
            elif stat.S_ISDIR(info.st_mode):
                kind = "directory"
            else:
                raise WorkspacePermissionError("delete target must be a file, directory or symlink")
            entry = {
                "path": path,
                "resolved_path": target.relative_to(self.root).as_posix(),
                "kind": kind,
                "version": _version(info),
            }
            if kind == "file":
                with self._delete_parent(target) as parent:
                    descriptor = os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
                    with os.fdopen(descriptor, "rb") as handle:
                        entry["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
                        if _version(os.fstat(handle.fileno())) != entry["version"]:
                            raise DeleteChangedError("delete target changed while preparing its snapshot")
            entries.append(entry)
            if kind == "directory":
                children = sorted(target.iterdir(), key=lambda child: child.name)
                if children and not recursive:
                    raise WorkspacePermissionError("nonempty directory deletion requires recursive=True")
                for child in children:
                    visit(f"{path}/{child.name}")
                if _version(target.lstat()) != _version(info):
                    raise DeleteChangedError("delete target changed while preparing its snapshot")

        visit(relative)
        return {
            "path": relative,
            "recursive": recursive,
            "requires_confirmation": len(entries) > 1,
            "entries": entries,
        }

    @contextmanager
    def _delete_parent(self, target: Path):
        # Anchor unlink/rmdir to directories opened without following replacement links.
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self.root, flags)
        try:
            for part in target.parent.relative_to(self.root).parts:
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            yield descriptor
        finally:
            os.close(descriptor)

    def delete_prepared(self, snapshot: dict[str, Any]) -> list[str]:
        """Delete only a previously inspected, unchanged and still authorized set."""
        current = self.prepare_delete(snapshot["path"], recursive=snapshot["recursive"])
        if current != snapshot:
            raise DeleteChangedError("delete target changed; prepare and confirm the new snapshot")
        pending = list(reversed(current["entries"]))
        deleted: list[str] = []
        try:
            for entry in pending:
                target = self._delete_target(entry["path"])
                if target.relative_to(self.root).as_posix() != entry["resolved_path"]:
                    raise DeleteChangedError("delete target parent changed during execution")
                with self._delete_parent(target) as parent:
                    actual = _version(os.stat(target.name, dir_fd=parent, follow_symlinks=False))
                    expected = entry["version"]
                    # Child removal changes directory times, but not its identity or mode.
                    if entry["kind"] == "directory":
                        changed = actual[:3] != expected[:3]
                    else:
                        changed = actual != expected
                    if changed:
                        raise DeleteChangedError("delete target changed during execution")
                    if entry["kind"] == "directory":
                        os.rmdir(target.name, dir_fd=parent)
                    else:
                        os.unlink(target.name, dir_fd=parent)
                deleted.append(entry["path"])
        except OSError as error:
            if not deleted:
                raise
            remaining = [entry["path"] for entry in pending[len(deleted):]]
            raise DeleteInterruptedError(str(error), deleted, remaining) from error
        return deleted
