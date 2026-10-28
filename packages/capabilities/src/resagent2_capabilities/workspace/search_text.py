"""Search text tool."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from resagent2_runtime import AgentState, ToolObservation
from resagent2_runtime.models import NonEmptyStr, RuntimeModel
from resagent2_components.text import (
    MAX_WORKSPACE_TEXT_BYTES,
    TextFileTooLargeError,
    read_text_file,
)
from resagent2_components.workspace import (
    MAIN_WORKSPACE_PATH_DESCRIPTION,
    MAIN_WORKSPACE_PATH_GUIDANCE,
    WorkspaceBoundary,
    WorkspacePermissionError,
)


_MAX_SKIPPED_FILES = 50


class SearchTextInput(RuntimeModel):
    """Case-insensitive bounded literal substring search, not a regex search."""

    query: NonEmptyStr = Field(
        description="Case-insensitive literal substring; no regular expressions."
    )
    path: str = Field(default=".", description=MAIN_WORKSPACE_PATH_DESCRIPTION)
    max_results: int = Field(default=20, ge=1, le=50)


class SearchTextTool:
    """Search readable text files without invoking a process."""

    name = "search_text"
    input_model = SearchTextInput
    model_guidance = MAIN_WORKSPACE_PATH_GUIDANCE + (
        "Search is a case-insensitive literal substring match, not regex. "
        "a|b matches the literal text a|b; search alternatives separately. "
        "path may identify a workspace file or directory. Result paths are "
        "copyable absolute addresses. max_results must be "
        "between 1 and 50. Use returned line numbers for a bounded read_file range. "
        "Workspace text files are limited to 10 MiB and must be UTF-8 without NUL "
        "bytes. skipped_count and skipped_files report files skipped because "
        "they are too large, not UTF-8 text, or could not be read; details are "
        "limited to 50 files. incomplete means files were skipped or the result "
        "limit stopped the search. At the result limit, further matches are "
        "not checked and the total match count is unknown. Zero matches with "
        "skipped files do not prove absence."
    )

    def __init__(
        self, boundary: WorkspaceBoundary, *,
        max_bytes: int = MAX_WORKSPACE_TEXT_BYTES,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.boundary = boundary
        self.max_bytes = max_bytes

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(SearchTextInput, arguments)
        matches: list[dict] = []
        observed: list[str] = []
        skipped_count = 0
        skipped_files: list[dict[str, str]] = []
        try:
            files = self.boundary.iter_files(args.path)
        except NotADirectoryError:
            # Directory resolution already enforced its grant; resolving the
            # file rechecks containment and symlinks. Never catch permission errors.
            path = self.boundary.resolve_read_file(args.path)
            files = [self.boundary.relative(path)]
        for relative in files:
            absolute = self.boundary.absolute_path(relative)
            reason = None
            try:
                path = self.boundary.resolve_read_file(relative)
                lines = read_text_file(path, max_bytes=self.max_bytes).splitlines()
            except WorkspacePermissionError:
                raise
            except TextFileTooLargeError:
                reason = "too_large"
            except ValueError:
                reason = "not_utf8_text"
            except OSError:
                reason = "read_error"
            if reason is not None:
                skipped_count += 1
                if len(skipped_files) < _MAX_SKIPPED_FILES:
                    skipped_files.append({"path": absolute, "reason": reason})
                continue
            file_matched = False
            for number, line in enumerate(lines, start=1):
                if args.query.lower() not in line.lower():
                    continue
                matches.append({"path": absolute, "line": number, "text": line[:200]})
                file_matched = True
                if len(matches) >= args.max_results:
                    break
            if file_matched:
                observed.append(absolute)
            if len(matches) >= args.max_results:
                break
        read_paths = list(state.memory.get("read_paths", []))
        for relative in observed:
            if relative not in read_paths:
                read_paths.append(relative)
        truncated = len(matches) >= args.max_results
        incomplete = bool(skipped_count) or truncated
        summary = f"Found {len(matches)} matches for {args.query!r}"
        if truncated:
            summary += (
                "; search incomplete: result limit reached; further matches not checked; "
                "total match count unknown"
            )
        if skipped_count:
            summary += f"; search incomplete: {skipped_count} files skipped"
        return ToolObservation(
            summary=summary,
            value={
                "matches": matches, "truncated": truncated,
                "skipped_count": skipped_count, "skipped_files": skipped_files,
                "skipped_files_truncated": skipped_count > len(skipped_files),
                "incomplete": incomplete,
            },
            memory_updates={"read_paths": read_paths},
        )
