"""Authorization-scoped material navigation; original sources stay authoritative."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from resagent2_contracts import AgentRequest, ArtifactIndexEntry, ArtifactRef
from resagent2_runtime import AgentState, ContextSection

from .artifacts import ArtifactReadError, RegisteredArtifactReader
from .dataset import DatasetAvailability
from .workspace import ReadOnlyWorkspaceBoundary, WorkspaceBoundary


# Only live-file expansion has a display quota. All registered material entries
# and authorized root locations remain mandatory and use normal context checks.
DEFAULT_FILE_INDEX_ENTRIES = 200
DEFAULT_FILE_INDEX_CHARS = 24_000


def _node(name: str, kind: str, *, status: str = "available", nature: str = "navigation",
          children: list[dict] | None = None, location: dict | None = None,
          metadata: dict | None = None) -> dict[str, Any]:
    value = {"name": name, "type": kind, "status": status, "content_nature": nature,
             "children": children or [], "expansion": {"state": "complete"}}
    if location is not None:
        value["location"] = location
    if metadata:
        value["metadata"] = metadata
    return value


def _artifact(ref: ArtifactRef) -> dict:
    entry = ArtifactIndexEntry.from_ref(ref).model_dump(mode="json", exclude_none=True)
    return _node(ref.output_name or ref.id, "artifact", nature="frozen",
                 location={"artifact_id": ref.id}, metadata=entry)


def _diagnostics(value: Any) -> Any:
    """Keep navigation facts, not nested tool bodies hidden in diagnostics."""
    if isinstance(value, list):
        return [_diagnostics(item) for item in value]
    if isinstance(value, dict):
        return {key: _diagnostics(item) for key, item in value.items() if key != "details"}
    return value


def _registered_materials(request: AgentRequest, state: AgentState,
                          reader: RegisteredArtifactReader | None) -> tuple[dict, int]:
    refs = {ref.id: ref for ref in request.input_artifacts}
    for artifact_id in state.authorized_artifact_ids:
        if artifact_id in refs:
            continue
        ref = reader.resolve_ref(artifact_id) if reader is not None else None
        if ref is None:
            raise ArtifactReadError(f"authorized artifact is unavailable: {artifact_id}")
        refs[artifact_id] = ref
    placed: set[str] = set()
    works = []
    navigation = [work.model_dump(mode="json", exclude_none=True,
                  exclude={"artifacts", "tasks"}) | {
                      "tasks": [task.model_dump(mode="json", exclude_none=True) for task in work.tasks],
                      "referenced_artifact_ids": [ref.id for ref in work.artifacts]}
                  for work in request.work_results]
    active = request.task_context.get("active_work_navigation")
    if active is not None:
        navigation.append(active)
    for work in navigation:
        tasks = []
        owned_ids: set[str] = set()
        for task in work.get("tasks", []):
            owned = [ref for ref in refs.values() if ref.task_id == task["task_id"] and ref.id not in placed]
            placed.update(ref.id for ref in owned)
            owned_ids.update(ref.id for ref in owned)
            # A navigation view must never publish another task's directed
            # business inputs or private Session if those models grow fields.
            metadata = {key: task[key] for key in (
                "task_id", "workflow_agent_kind", "execution_ref", "instruction", "status", "attempts", "error", "warnings"
            ) if key in task}
            tasks.append(_node(task["task_id"], "task", status=task["status"], nature="task_record",
                               children=[_artifact(ref) for ref in owned], metadata=_diagnostics(metadata)))
        metadata = {key: work[key] for key in (
            "work_request_id", "objective", "status", "workflow_revision", "error", "owner_task_id"
        ) if key in work}
        references = [key for key in work.get("referenced_artifact_ids", [])
                      if key in refs and key not in owned_ids]
        if references:
            metadata["referenced_artifact_ids"] = references
        if not tasks:
            metadata["note"] = "No tasks were created."
        works.append(_node(work["work_request_id"], "work_request", status=work["status"],
                           nature="delegation_record", children=tasks, metadata=_diagnostics(metadata)))
    categories: dict[str, list] = {"literature": [], "web": [], "other": []}
    for ref in refs.values():
        if ref.id in placed:
            continue
        category = ("literature" if ref.kind in {"literature_paper", "literature_pdf", "literature_fulltext"}
                    else "web" if ref.kind == "web_page" else "other")
        categories[category].append(_artifact(ref))
    groups = [_node("literature", "group", children=categories["literature"]),
              _node("web", "group", children=categories["web"]),
              _node("work_requests", "group", children=works),
              _node("other", "group", children=categories["other"])]
    return _node("registered_materials", "group", children=groups), len(refs)


def _default_boundary(request: AgentRequest) -> ReadOnlyWorkspaceBoundary:
    grants = list(request.read_workspaces)
    if request.workspace is not None:
        grants.insert(0, request.workspace)
    return ReadOnlyWorkspaceBoundary(grants,
        relative_root=request.workspace.root if request.workspace is not None else None)


def _directory_location(path: str) -> dict:
    return {"state": "not_expanded", "listing": {"path": path}}


def _file_spaces(boundary: WorkspaceBoundary | ReadOnlyWorkspaceBoundary,
                 max_entries: int, max_chars: int) -> dict:
    if isinstance(boundary, WorkspaceBoundary):
        roots = [(boundary.absolute_path(".", allow_root=True), boundary.grant)]
    else:
        roots = [(grant.root, grant) for grant in boundary.grants]
    nodes, observed = [], set()
    queue = deque()
    for path, grant in roots:
        if path in observed or not grant.access.read_paths:
            continue
        observed.add(path)
        node = _node(Path(grant.root).name or path, "directory", nature="directory_fact",
                     location={"path": path}, metadata={"source_kind": grant.source.value})
        node["expansion"] = _directory_location(path)
        try:
            boundary.resolve_read_directory(path, traverse=True)
        except FileNotFoundError:
            node["status"] = "missing"
            node["expansion"] = {"state": "complete"}
        except (OSError, ValueError) as error:
            node["status"] = "read_error"
            node["error"] = type(error).__name__
            node["expansion"] = {"state": "complete"}
        else:
            queue.append((node, path))
        nodes.append(node)
    remaining_entries, remaining_chars = max_entries, max_chars
    while queue and remaining_entries > 0 and remaining_chars > 0:
        node, path = queue.popleft()
        try:
            page = boundary.list_directory(path, max_entries=min(2000, remaining_entries))
        except FileNotFoundError:
            node["status"], node["expansion"] = "missing", {"state": "complete"}
            continue
        except (OSError, ValueError) as error:
            node["status"], node["error"] = "read_error", type(error).__name__
            continue
        node["observation"] = page["observation"]
        accepted = []
        for child in page["children"]:
            if child["type"] == "directory" and child["status"] == "available":
                child["expansion"] = _directory_location(child["location"]["path"])
            size = len(json.dumps(child, ensure_ascii=False))
            if size > remaining_chars:
                break
            accepted.append(child)
            remaining_chars -= size
            remaining_entries -= 1
        node["children"] = accepted
        if len(accepted) == len(page["children"]):
            node["expansion"] = dict(page["expansion"])
            if node["expansion"]["state"] == "partial":
                node["expansion"]["listing"] = {
                    "path": path, "cursor": node["expansion"]["next_cursor"]}
        else:
            # Character-limited previews cannot reuse a cursor after entries
            # that were not displayed. Start a real tool enumeration instead.
            node["expansion"] = {"state": "partial" if accepted else "not_expanded",
                                 "listing": {"path": path}, "reason": "display_budget"}
            remaining_chars = 0
        for child in accepted:
            if (child["type"] == "directory" and child["status"] == "available"
                    and not child.get("is_symlink") and child["name"] not in WorkspaceBoundary.ignored_parts):
                queue.append((child, child["location"]["path"]))
    return _node("file_spaces", "group", children=nodes)


def _datasets(availability: DatasetAvailability) -> dict:
    entries = []
    for item in sorted(availability.available, key=lambda value: value["dataset_id"]):
        entries.append(_node(item["dataset_id"], "dataset", nature="dataset_resource",
            location={"dataset_id": item["dataset_id"]}, metadata={
                "access": "read_only", "availability_basis": "registered directory exists; contents not validated",
                "access_method": "existing dataset environment mapping; not workspace file authorization",
                "directory_listing_supported": False}))
    for dataset_id in sorted(availability.unavailable_ids):
        entries.append(_node(dataset_id, "dataset", status="missing", nature="dataset_resource",
            location={"dataset_id": dataset_id}, metadata={"directory_listing_supported": False}))
    return _node("datasets", "catalog", nature="catalog_record", children=entries)


def material_index_context(
    request: AgentRequest, state: AgentState, *, reader: RegisteredArtifactReader | None = None,
    read_boundary: WorkspaceBoundary | ReadOnlyWorkspaceBoundary | None = None,
    datasets: DatasetAvailability | None = None,
    max_file_entries: int = DEFAULT_FILE_INDEX_ENTRIES,
    max_file_chars: int = DEFAULT_FILE_INDEX_CHARS,
) -> ContextSection:
    """Complete artifact entries and roots, with explicit bounded file previews."""
    if type(max_file_entries) is not int or max_file_entries < 0 or type(max_file_chars) is not int or max_file_chars < 0:
        raise ValueError("file index display limits must be nonnegative integers")
    registered, artifact_count = _registered_materials(request, state, reader)
    files = _file_spaces(read_boundary or _default_boundary(request), max_file_entries, max_file_chars)
    value = _node("materials", "group", children=[registered, files, _datasets(datasets or DatasetAvailability())])
    value["artifact_count"] = artifact_count
    return ContextSection(name="material_index", content=json.dumps(value, ensure_ascii=False))
