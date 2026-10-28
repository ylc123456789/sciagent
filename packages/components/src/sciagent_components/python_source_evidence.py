"""Bounded, read-only Python source evidence for a proposed shell command.

This collects evidence without executing or simulating Bash/Python, installing
dependencies, or deciding safety. Unsupported behavior is reported as a gap.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._bash_source_structure import (
    MAX_AST_DEPTH, MAX_AST_NODES, MAX_COMMAND_BYTES, parse_bash_structure,
)
from sciagent_runtime.cancellation import check_cancelled

from .text import decode_text
from .workspace import WorkspaceBoundary, WorkspacePermissionError


MAX_FILES = 32
MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 512 * 1024
MAX_IMPORT_DEPTH = 8
MAX_IMPORTS = 128
MAX_GAPS = 64
MAX_REVIEW_INSPECTIONS = 3
_PYTHON = re.compile(r"python(?:\d+(?:\.\d+)*)?\Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SourceEvidence:
    """Source text for a reviewer and a stable content identity for preparation."""

    fingerprint: str
    _payload: dict[str, Any]

    def _inspection_budget(self) -> dict[str, int]:
        used = len(self._payload.get("review_inspections", []))
        return {"limit": MAX_REVIEW_INSPECTIONS, "used": used,
                "remaining": MAX_REVIEW_INSPECTIONS - used}

    def review_payload(self) -> dict[str, Any]:
        return {**copy.deepcopy(self._payload), "inspection_budget": self._inspection_budget()}

    def inspection_payload(self, previous: SourceEvidence) -> dict[str, Any]:
        """Project this observation into its receipt without resending old sources."""
        # Recollection can race with a file change after the prefix check.
        # Never bind changed older evidence while omitting its body as "known".
        current_files = {item["path"]: item for item in self._payload["files"]}
        if (any(current_files.get(item["path"]) != item for item in previous._payload["files"])
                or self._payload["review_inspections"][:-1] != previous._payload.get("review_inspections", [])
                or any(self._payload[key] != previous._payload[key] for key in ("cwd", "coverage"))
                or any(self._payload["limits"].get(key) != value
                       for key, value in previous._payload["limits"].items())
                or any(not {_canonical(item) for item in previous._payload[key]}.issubset(
                    {_canonical(item) for item in self._payload[key]}) for key in ("imports", "gaps"))):
            raise ValueError("Previously provided source evidence changed during inspection")
        observation = self._payload["review_inspections"][-1]
        known_paths = {item["path"] for item in previous._payload["files"]}
        result = {
            "inspection": observation,
            # An explicit reread still returns the requested file. Other source
            # bodies appear only when first discovered by this inspection.
            "files": [item for item in self._payload["files"]
                      if item["path"] == observation.get("source_path")
                      or item["path"] not in known_paths],
            "limits": self._payload["limits"],
            "inspection_budget": self._inspection_budget(),
            "source_fingerprint": self.fingerprint,
        }
        for key in ("imports", "gaps"):
            known = {_canonical(item) for item in previous._payload[key]}
            result[key] = [item for item in self._payload[key] if _canonical(item) not in known]
        return copy.deepcopy(result)


class _Collector:
    def __init__(self, boundary: WorkspaceBoundary) -> None:
        self.boundary = boundary
        self.files: dict[str, dict[str, Any]] = {}
        self.visited: set[tuple[str, str, str, str]] = set()
        self.imports: list[dict[str, Any]] = []
        self.gaps: dict[str, dict[str, Any]] = {}
        self.coverage: list[dict[str, Any]] = []
        self.total_bytes = 0
        self.observed_files: set[str] = set()
        self.file_failures: dict[str, dict[str, Any]] = {}
        self.import_attempts = 0
        self.shell_writes: set[str] = set()
        self.shell_write_unknown = False

    def gap(self, code: str, **details: Any) -> None:
        value = {"code": code, **details}
        if len(self.gaps) >= MAX_GAPS:
            if code.endswith("_limit"):
                self.gaps.setdefault(f"limit:{code}", value)
            else:
                self.gaps.setdefault("limit:gap_count", {"code": "gap_count_limit", "limit": MAX_GAPS})
            return
        self.gaps[_canonical(value)] = value

    def relative(self, value: str) -> str:
        if value != value.strip() or "\\" in value:
            raise WorkspacePermissionError("path spelling is not preserved by workspace normalization")
        path = Path(value)
        if path.is_absolute():
            try:
                return path.relative_to(self.boundary.root).as_posix()
            except ValueError:
                raise WorkspacePermissionError("path is outside workspace") from None
        return path.as_posix()

    def read(self, relative: str, *, module: str, package: str,
             search_root: str, depth: int) -> None:
        if depth > MAX_IMPORT_DEPTH:
            self.gap("import_depth_limit", path=relative, limit=MAX_IMPORT_DEPTH)
            return
        try:
            path = self.boundary.resolve_read_file(relative)
        except (OSError, WorkspacePermissionError) as exc:
            self.gap("source_unavailable", path=relative, reason=type(exc).__name__)
            return
        if module == "__main__" and not package:
            # CPython resolves the entrypoint before setting sys.path[0]. A
            # symlink's lexical directory must not supply different evidence.
            search_root = path.parent.relative_to(self.boundary.root).as_posix()
        canonical = path.as_posix()
        if self.shell_write_unknown or canonical in self.shell_writes:
            self.gap("source_version_unestablished", path=canonical,
                     reason="preceding_or_attached_shell_write")
            return
        identity = (canonical, module, package, search_root)
        if identity in self.visited:
            return
        self.visited.add(identity)
        if self.load_file(path, python_source=True) is None:
            return
        content = self.files[canonical]["content"]
        try:
            tree = ast.parse(content, filename=canonical)
        except (SyntaxError, ValueError, RecursionError):
            self.gap("source_parse_failed", path=canonical)
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.resolve(alias.name, search_root=search_root, source=canonical,
                                 line=node.lineno, depth=depth + 1)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parents = package.split(".") if package else []
                    if node.level > len(parents):
                        self.gap("relative_import_unresolved", path=canonical,
                                 line=node.lineno, module=base, level=node.level)
                        continue
                    prefix = parents[:len(parents) - node.level + 1]
                    base = ".".join([*prefix, *([base] if base else [])])
                if base:
                    self.resolve(base, search_root=search_root, source=canonical,
                                 line=node.lineno, depth=depth + 1)
                    # "from package import member" may load a submodule. If no
                    # local submodule exists, member may simply be an attribute.
                    for alias in node.names:
                        if alias.name != "*":
                            self.resolve(f"{base}.{alias.name}", search_root=search_root,
                                         source=canonical, line=node.lineno,
                                         depth=depth + 1, optional=True)
            elif isinstance(node, ast.Call):
                name = node.func
                if ((isinstance(name, ast.Name) and name.id in {"__import__", "eval", "exec"})
                        or (isinstance(name, ast.Attribute) and name.attr in {
                            "import_module", "spec_from_file_location", "run_module", "run_path",
                        })):
                    self.gap("dynamic_code_or_import", path=canonical, line=node.lineno)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "sys" and node.attr in {"path", "meta_path", "path_hooks"}:
                    self.gap("runtime_import_path", path=canonical, line=node.lineno)

    def load_file(self, path: Path, *, python_source: bool) -> dict[str, Any] | None:
        """Acquire a complete unique file under the shared source evidence quota."""
        canonical = path.as_posix()
        if canonical in self.files:
            return self.files[canonical]
        if canonical in self.file_failures:
            return None
        remaining = MAX_TOTAL_BYTES - self.total_bytes

        def fail(code: str, **details: Any) -> None:
            failure = {"code": code, "path": canonical, **details}
            self.file_failures[canonical] = failure
            self.gap(**failure)

        if len(self.observed_files) >= MAX_FILES:
            fail("file_count_limit", limit=MAX_FILES, files_collected=len(self.observed_files),
                 remaining_bytes=remaining)
            return None
        allowance = min(MAX_FILE_BYTES, remaining)
        try:
            with path.open("rb") as stream:
                raw = stream.read(allowance + 1)
        except OSError as exc:
            fail("source_unavailable", reason=type(exc).__name__)
            return None
        if len(raw) > allowance:
            per_file = MAX_FILE_BYTES <= remaining
            fail("source_file_bytes_limit" if per_file else "source_total_bytes_limit",
                 limit=MAX_FILE_BYTES if per_file else MAX_TOTAL_BYTES,
                 remaining_bytes=remaining, size_bytes_at_least=len(raw))
            return None
        # Even undecodable bytes were observed; rereads must not charge them
        # again or count one canonical file multiple times.
        self.observed_files.add(canonical)
        self.total_bytes += len(raw)
        try:
            if python_source:
                encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                content = raw.decode(encoding)
            else:
                encoding, content = "utf-8", decode_text(raw)
        except (SyntaxError, UnicodeError, LookupError, ValueError):
            fail("source_encoding_unavailable", sha256=hashlib.sha256(raw).hexdigest(),
                 size_bytes=len(raw))
            return None
        record = {
            "path": canonical, "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw), "encoding": encoding, "content": content,
        }
        self.files[canonical] = record
        return record

    def candidate(self, relative: str) -> tuple[str, str | None]:
        try:
            path = self.boundary.resolve_read_file(relative)
        except FileNotFoundError:
            return "absent", None
        except (OSError, WorkspacePermissionError) as exc:
            return "unavailable", type(exc).__name__
        return "source", path.as_posix()

    def resolve(self, module: str, *, search_root: str, source: str,
                line: int, depth: int, optional: bool = False) -> list[str]:
        if self.import_attempts >= MAX_IMPORTS:
            self.gap("import_count_limit", limit=MAX_IMPORTS)
            return []
        self.import_attempts += 1
        parts = module.split(".")
        if not all(part.isidentifier() for part in parts):
            self.gap("module_name_unresolved", path=source, module=module, line=line)
            return []
        record: dict[str, Any] = {"source": source, "line": line, "module": module,
                                  "search_root": search_root, "optional_member": optional}
        loaded: list[str] = []
        for number in range(1, len(parts) + 1):
            prefix = Path(search_root).joinpath(*parts[:number])
            # FileFinder checks a regular package before the module .py file.
            init = (prefix / "__init__.py").as_posix()
            py = prefix.with_suffix(".py").as_posix()
            init_status, init_value = self.candidate(init)
            py_status, py_value = self.candidate(py)
            if init_status == "source":
                selected, target, is_package = init, init_value, True
            elif py_status == "source":
                selected, target, is_package = py, py_value, False
                if init_status == "unavailable":
                    record.setdefault("unavailable_candidates", []).append(init)
                    self.gap("import_candidate_unavailable", path=source, module=module,
                             candidate=init, reason=init_value)
            elif init_status == "unavailable":
                record.update(status="unavailable", candidate=init, reason=init_value)
                self.gap("import_source_unavailable", path=source, module=module,
                         candidate=init, reason=init_value)
                break
            elif py_status == "unavailable":
                record.update(status="unavailable", candidate=py, reason=py_value)
                self.gap("import_source_unavailable", path=source, module=module,
                         candidate=py, reason=py_value)
                break
            else:
                try:
                    self.boundary.resolve_read_directory(prefix.as_posix(), traverse=True)
                except FileNotFoundError:
                    record.update(status="not_local", candidates=[init, py])
                    if not optional:
                        self.gap("external_or_unresolved_import", path=source,
                                 module=module, line=line)
                    break
                except (OSError, WorkspacePermissionError) as exc:
                    record.update(status="unavailable", candidate=prefix.as_posix(),
                                  reason=type(exc).__name__)
                    self.gap("import_source_unavailable", path=source, module=module,
                             candidate=prefix.as_posix(), reason=type(exc).__name__)
                    break
                else:
                    record["namespace_packages"] = [*record.get("namespace_packages", []),
                                                      prefix.as_posix()]
                    continue
            assert target is not None
            loaded.append(target)
            name = ".".join(parts[:number])
            self.read(selected, module=name, package=name if is_package else name.rpartition(".")[0],
                      search_root=search_root, depth=depth)
            if not is_package and number < len(parts):
                # Attribute imports need no extra file, but a dotted import of
                # a non-package cannot be resolved from this local source.
                record.update(status="local_attribute_or_unresolved")
                if not optional:
                    self.gap("module_parent_not_package", path=source, module=module)
                break
        else:
            record["status"] = "local" if loaded else "namespace_package"
        record["local_sources"] = loaded
        self.imports.append(record)
        return loaded

    def shell_redirects(self, redirects: list[dict[str, Any]]) -> None:
        """Do not read old bytes as a version changed by the proposed script."""
        for redirect in redirects:
            if redirect["kind"] != "file_write":
                continue
            destination = redirect.get("destination")
            if destination is None:
                self.shell_write_unknown = True
                continue
            try:
                relative = self.relative(destination)
                path = self.boundary.resolve_read_file(relative)
            except FileNotFoundError:
                # resolve_read_file already checked the lexical read grant.
                path = (self.boundary.root / relative).resolve(strict=False)
                if (not path.is_relative_to(self.boundary.root)
                        or not self.boundary.allows_read(path.relative_to(self.boundary.root).as_posix())):
                    self.shell_write_unknown = True
                    continue
            except (OSError, WorkspacePermissionError):
                self.shell_write_unknown = True
                continue
            self.shell_writes.add(path.as_posix())

    def command(self, command: str) -> None:
        structure = parse_bash_structure(command)
        for gap in structure.gaps:
            self.gap(**gap)
        context_changes: list[dict[str, Any]] = []
        prior_effect: list[int] | None = None
        for index, item in enumerate(structure.commands):
            args = item.get("argv", [])
            coverage: dict[str, Any] = {"command_index": index, **item}
            self.coverage.append(coverage)
            if item.get("status") == "unsupported_shell_structure":
                # Do not walk bodies of functions, control flow or subshells
                # as though they were top-level executed Python statements.
                if not context_changes:
                    context_changes.append({"span": item["span"], "node_type": item["node_type"]})
                continue
            redirects = item["redirections"]
            self.shell_redirects(redirects)
            if (any(word["evaluated_expressions"] for word in item["opaque_words"])
                    or any(redirect.get("evaluated_expressions") for redirect in redirects)):
                if not context_changes:
                    context_changes.append({"span": item["span"], "reason": "shell_expansion_effects_unestablished"})
            current_changes = list(context_changes)
            if item["assignments"]:
                current_changes.append({"span": item["span"], "reason": "command_environment_assignment"})
            if any(value["kind"] == "unsupported" for value in redirects):
                current_changes.append({"span": item["span"], "reason": "unsupported_redirection"})
            if not args or args[0] is None:
                coverage["status"] = "command_name_unresolved"
                if not context_changes:
                    context_changes.append({"span": item["span"], "reason": "dynamic_command"})
                self.gap("shell_command_unresolved", command_index=index)
                continue
            same_cwd_cd = (args == ["cd", self.boundary.root.as_posix()]
                           and not context_changes and prior_effect is None
                           and not item["assignments"] and not item["opaque_words"]
                           and not redirects)
            if not same_cwd_cd and (args[0] in {"cd", "pushd", "popd", "source", ".", "eval", "alias", "unalias",
                           "export", "declare", "typeset", "readonly", "unset", "read", "mapfile",
                           "shopt", "enable", "hash", "trap", "exec"} or (
                    args[0] == "set" and args != ["set", "-o", "pipefail"])):
                if not context_changes:
                    context_changes.append({"span": item["span"], "reason": "shell_context_change"})
                current_changes = list(context_changes)
            if not _PYTHON.fullmatch(Path(args[0]).name):
                coverage["status"] = "shell_option_only" if args == ["set", "-o", "pipefail"] else "unsupported"
                if same_cwd_cd:
                    coverage.update(status="same_working_directory", cwd_unchanged=True)
                    # This establishes only the directory used for source
                    # lookup. cd can still change PWD/OLDPWD and shell state.
                    self.gap("shell_directory_metadata_effects_unestablished", command_index=index,
                             variables=["PWD", "OLDPWD"])
                    prior_effect = item["span"]
                elif coverage["status"] == "unsupported":
                    self.gap("unsupported_command", command_index=index)
                    prior_effect = item["span"]
                continue
            coverage["source_context"] = {
                "view": "current_pre_operation_bytes",
                "prior_effects_unestablished": prior_effect is not None,
                "last_preceding_command_span": prior_effect,
            }
            if prior_effect is not None:
                self.gap("preceding_command_effects_unestablished", command_index=index,
                         last_preceding_command_span=prior_effect)
            if current_changes:
                coverage.update(status="source_context_unestablished", context_changes=current_changes)
                self.gap("shell_source_context_unestablished", command_index=index,
                         changes=current_changes)
                prior_effect = item["span"]
                continue
            rest = args[1:]
            if rest in (["--version"], ["-V"]):
                coverage["status"] = "interpreter_version_only"
                continue
            # Later Python commands are current-file candidates: this one can
            # change files even when its entrypoint itself is fully literal.
            prior_effect = item["span"]
            while rest and rest[0] in {"-u", "-B"}:
                rest = rest[1:]
            if len(rest) >= 3 and rest[:2] == ["-m", "unittest"]:
                modules = [value for value in rest[2:] if value not in {"-v", "--verbose", "-q", "--quiet"}]
                if (not modules or any(value is None or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value)
                                       or value == "discover" for value in modules)):
                    coverage["status"] = "unsupported"
                    self.gap("unittest_targets_unresolved", command_index=index)
                    continue
                coverage["status"] = "explicit_unittest_modules"
                runner_sources = self.resolve("unittest.__main__", search_root=".",
                                              source="<command>", line=0, depth=0, optional=True)
                if runner_sources:
                    coverage["runner_local_sources"] = runner_sources
                    self.gap("local_unittest_runner", paths=runner_sources)
                for module in modules:
                    # A dotted test selector may end in class/method names.
                    loaded = self.resolve(module, search_root=".", source="<command>", line=0,
                                          depth=0, optional=True)
                    if not loaded:
                        self.gap("unittest_source_unresolved", module=module)
                continue
            if not rest or rest[0] is None or rest[0].startswith("-"):
                coverage["status"] = "unsupported"
                self.gap("python_entrypoint_unresolved", command_index=index)
                continue
            try:
                relative = self.relative(rest[0])
            except WorkspacePermissionError:
                coverage["status"] = "entrypoint_unresolved"
                self.gap("entrypoint_path_unresolved", path=rest[0])
                continue
            coverage.update(status="direct_script", entrypoint=relative)
            self.read(relative, module="__main__", package="",
                      search_root=Path(relative).parent.as_posix(), depth=0)


def validate_review_inspections(value: object) -> list[dict[str, str]]:
    """Validate system inspection descriptors before preparation or recovery."""
    if not isinstance(value, list) or len(value) > MAX_REVIEW_INSPECTIONS:
        raise ValueError("invalid review inspection list")
    result = []
    for item in value:
        if (not isinstance(item, dict) or set(item) != {"kind", "path"}
                or item["kind"] not in ("file", "directory")
                or not isinstance(item["path"], str) or not item["path"].strip()):
            raise ValueError("invalid review inspection descriptor")
        result.append(dict(item))
    return result


def _inspect(collector: _Collector, descriptor: dict[str, str]) -> dict:
    """Use the existing read grant directly; never request another approval."""
    check_cancelled()
    result = {**descriptor, "status": "unavailable"}
    try:
        relative = collector.relative(descriptor["path"])
        if descriptor["kind"] == "directory":
            result.update(status="available", listing=collector.boundary.list_directory(relative, max_entries=100))
            return result
        path = collector.boundary.resolve_read_file(relative)
        canonical = path.as_posix()
        if canonical not in collector.files:
            if path.suffix == ".py":
                # These are explicitly requested CURRENT bytes, not a claim that
                # a preceding Shell write cannot change the later executable.
                writes, unknown = collector.shell_writes, collector.shell_write_unknown
                collector.shell_writes, collector.shell_write_unknown = set(), False
                try:
                    collector.read(relative, module="__main__", package="",
                                   search_root=path.parent.relative_to(collector.boundary.root).as_posix(), depth=0)
                finally:
                    collector.shell_writes, collector.shell_write_unknown = writes, unknown
            else:
                collector.load_file(path, python_source=False)
        if canonical in collector.files:
            result.update(status="available", source_path=canonical)
        else:
            failure = collector.file_failures.get(canonical)
            result.update(reason=failure["code"] if failure else "source_unavailable")
            if failure is not None:
                result["gap"] = dict(failure)
    except (OSError, ValueError, WorkspacePermissionError) as exc:
        # Do not reflect an arbitrary filesystem error containing host details.
        result["reason"] = type(exc).__name__
    check_cancelled()
    return result


def collect_python_source_evidence(
    command: str, boundary: WorkspaceBoundary, *, inspections: list[dict] | None = None,
) -> SourceEvidence:
    """Read supported Python entrypoints and statically identifiable local imports.

    Relative paths use ``boundary.root`` as the command working directory. The
    caller must use that same directory for the proposed process. Standard
    library, installed/compiled modules, environment/PYTHONPATH changes, dynamic
    code and runtime file/network access are outside this static source view.
    No code from the workspace is imported or executed.
    """
    collector = _Collector(boundary)
    collector.command(command)
    inspected = [_inspect(collector, item) for item in validate_review_inspections(inspections if inspections is not None else [])]
    payload: dict[str, Any] = {
        "cwd": boundary.root.as_posix(),
        "files": [collector.files[path] for path in sorted(collector.files)],
        "imports": sorted(collector.imports, key=_canonical),
        "gaps": [collector.gaps[key] for key in sorted(collector.gaps)],
        "coverage": collector.coverage,
        "limits": {"max_files": MAX_FILES, "max_file_bytes": MAX_FILE_BYTES,
                   "max_total_bytes": MAX_TOTAL_BYTES, "max_import_depth": MAX_IMPORT_DEPTH,
                   "max_imports": MAX_IMPORTS, "max_gaps": MAX_GAPS,
                   "max_command_bytes": MAX_COMMAND_BYTES, "max_ast_nodes": MAX_AST_NODES,
                   "max_ast_depth": MAX_AST_DEPTH},
        "scope": "Static local Python source only; not a complete dependency or behavior proof. "
                 "Interpreter startup hooks, PYTHONPATH, installed/compiled modules, dynamic imports "
                 "and runtime file/network access are not established by this evidence. "
                 "Files are current pre-operation bytes; preceding commands may change them. "
                 "Bash structure and local gaps do not establish complete execution behavior.",
    }
    if inspected:
        payload["review_inspections"] = inspected
        payload["limits"].update(max_review_inspections=MAX_REVIEW_INSPECTIONS, max_directory_entries=100)
        payload["scope"] += (
            " Explicit review inspections add current read-only workspace file/directory observations; "
            "they do not resolve dynamic Shell execution context or prove future file versions."
        )
    identity = {**payload, "files": [{key: value for key, value in item.items() if key != "content"}
                                    for item in payload["files"]]}
    return SourceEvidence(hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest(), payload)
