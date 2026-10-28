"""Derive the file-reading view from an invocation's existing authority."""
import hashlib
from pathlib import Path
import tempfile
from uuid import uuid4
from resagent2_contracts import AgentRequest, WorkspaceAccess, WorkspaceGrant, WorkspaceSourceKind
from .workspace import ReadOnlyWorkspaceBoundary


def request_read_boundary(
    request: AgentRequest, *, command_log_dir: str | None = None,
    tool_result_dir: str | None = None, execution_environment_dir: str | None = None,
) -> ReadOnlyWorkspaceBoundary:
    grants = list(request.read_workspaces)
    if request.workspace is not None:
        grants.insert(0, WorkspaceGrant(
            root=request.workspace.root, source=request.workspace.source,
            access=WorkspaceAccess(read_paths=list(request.workspace.access.read_paths),
                                   denied_paths=list(request.workspace.access.denied_paths)),
        ))
    # Process output is already owned by this invocation. Expose only its command
    # logs and runtime-owned environment, not the enclosing Run, Session stores,
    # or another attempt's output.
    for directory in (command_log_dir, tool_result_dir, execution_environment_dir):
        if directory is not None:
            grants.append(WorkspaceGrant(
                root=str(Path(directory).absolute()), source=WorkspaceSourceKind.GENERATED,
                access=WorkspaceAccess(read_paths=['.']),
            ))
    return ReadOnlyWorkspaceBoundary(grants, relative_root=request.workspace.root if request.workspace else None)


def tool_result_directory(output_dir: str | None, session_id: str,
                          *, previous: str | None = None) -> str:
    """Keep generated receipts in one narrow, stable Session-owned directory."""
    if previous is not None:
        return previous
    if output_dir is not None:
        identity = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return str((Path(output_dir).expanduser() / "tool-results" / identity).resolve())
    # Standalone callers may have no output layout. Persist this unique location
    # with the Session; create it only when an oversized receipt actually occurs.
    return str(Path(tempfile.gettempdir()) / "resagent2-tool-results" / uuid4().hex)
