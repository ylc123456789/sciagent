"""Workspace text tools share bounded UTF-8 handling and explicit search coverage."""

import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sciagent_capabilities import (
    CreateFileTool,
    ReadFileTool,
    ReplaceTextTool,
    SearchTextTool,
)
from sciagent_components import WorkspaceBoundary, WorkspacePermissionError
from sciagent_components.text import MAX_WORKSPACE_TEXT_BYTES
from sciagent_contracts import AgentOwner, WorkspaceAccess, WorkspaceGrant
from sciagent_runtime import AgentState


def _state():
    now = datetime.now(UTC)
    return AgentState(
        session_id="session_text", agent_name="coding", owner=AgentOwner.EXECUTOR,
        run_id="run_text", task_id="task_text", attempt_number=1,
        created_at=now, updated_at=now,
    )


def _boundary(root):
    return WorkspaceBoundary(WorkspaceGrant(
        root=str(root), source="local",
        access=WorkspaceAccess(
            read_paths=["."], write_paths=["."], denied_paths=["denied"],
        ),
    ))


def test_default_workspace_limit_accepts_large_research_text_across_tools(tmp_path):
    # Above the former 1 MB cap; only a small selected window is returned.
    text = "needle\n" + "x" * 1_100_000
    boundary = _boundary(tmp_path)
    created = CreateFileTool(boundary).execute(
        _state(), CreateFileTool.input_model(path="log.txt", content=text),
    )
    assert created.value["bytes"] == len(text.encode("utf-8"))
    read = ReadFileTool(boundary).execute(
        _state(), ReadFileTool.input_model(path="log.txt", start_line=1, end_line=1),
    )
    assert read.value["content"] == "needle\n"
    search = SearchTextTool(boundary).execute(
        _state(), SearchTextTool.input_model(path="log.txt", query="needle"),
    )
    assert search.value["matches"] == [
        {"path": str(tmp_path / "log.txt"), "line": 1, "text": "needle"},
    ]
    assert search.value["incomplete"] is False
    replaced = ReplaceTextTool(boundary).execute(
        _state(), ReplaceTextTool.input_model(
            path="log.txt", old_text="needle", new_text="target",
        ),
    )
    assert replaced.ok
    assert (tmp_path / "log.txt").read_bytes() == text.replace(
        "needle", "target", 1,
    ).encode("utf-8")


def test_default_workspace_limit_includes_exact_10_mib(tmp_path):
    content = "x" * MAX_WORKSPACE_TEXT_BYTES
    boundary = _boundary(tmp_path)
    CreateFileTool(boundary).execute(
        _state(), CreateFileTool.input_model(path="exact.txt", content=content),
    )
    assert (tmp_path / "exact.txt").stat().st_size == 10 * 1024 * 1024
    read = ReadFileTool(boundary).execute(
        _state(), ReadFileTool.input_model(path="exact.txt", start_char=len(content) - 1),
    )
    assert read.value["content"] == "x"
    assert read.value["truncated"] is False
    ReplaceTextTool(boundary).execute(
        _state(), ReplaceTextTool.input_model(
            path="exact.txt", old_text=content, new_text="small",
        ),
    )
    assert (tmp_path / "exact.txt").read_bytes() == b"small"


def test_read_file_character_offsets_count_unicode_after_selecting_lines(tmp_path):
    body = "header\r\n\u4e2d\u6587\U0001f600abc\r\ntail\r"
    (tmp_path / "unicode.txt").write_bytes(body.encode("utf-8"))
    tool = ReadFileTool(_boundary(tmp_path))
    result = tool.execute(_state(), tool.input_model(
        path="unicode.txt", start_line=2, end_line=2, start_char=1, end_char=4,
    ))
    assert result.value == {
        "path": "unicode.txt", "start_line": 2, "end_line": 2,
        "start_char": 1, "end_char": 4, "content": "\u6587\U0001f600a",
        "total_lines": 3, "selected_chars": 8,
        "truncated": False, "next_start_char": 4,
    }


def test_workspace_tool_pages_single_long_line_without_rewriting(tmp_path):
    body = '{"report":"' + "\u4e2d\u6587" * 150_000 + '","tail":"UNIQUE_TAIL"}'
    path = tmp_path / "long.json"
    original = body.encode("utf-8")
    path.write_bytes(original)
    tool = ReadFileTool(_boundary(tmp_path))
    default = tool.execute(_state(), tool.input_model(path=path.name))
    assert len(default.value["content"]) == 128_000
    assert default.value["truncated"] is True
    pieces = []
    for start in range(0, len(body), 100_000):
        page = tool.execute(_state(), tool.input_model(
            path=path.name, start_char=start, end_char=start + 100_000,
        ))
        assert page.value["truncated"] is False
        pieces.append(page.value["content"])
    assert "".join(pieces) == body
    assert path.read_bytes() == original


@pytest.mark.parametrize("arguments", [
    {"start_line": 2, "end_line": 1},
    {"start_char": 5, "end_char": 5},
    {"start_char": 5, "end_char": 4},
])
def test_read_file_invalid_window_fails_before_filesystem_resolution(
    tmp_path, monkeypatch, arguments,
):
    boundary = _boundary(tmp_path)

    def forbidden_resolve(*args, **kwargs):
        raise AssertionError("invalid range must fail before file IO")

    monkeypatch.setattr(boundary, "resolve_read_file", forbidden_resolve)
    tool = ReadFileTool(boundary)
    with pytest.raises(ValueError):
        tool.execute(_state(), tool.input_model(path="missing.txt", **arguments))


@pytest.mark.parametrize("content,error", [
    ("\u4e2d\u6587a", "too large"),
    ("a\x00b", "NUL"),
    ("a\ud800b", "UTF-8"),
])
def test_create_rejects_invalid_or_large_text_before_creating_parents(
    tmp_path, content, error,
):
    state = _state()
    tool = CreateFileTool(_boundary(tmp_path), max_bytes=6)
    with pytest.raises(ValueError, match=error):
        tool.execute(state, tool.input_model(path="nested/new.txt", content=content))
    assert not (tmp_path / "nested").exists()
    assert state.memory == {}


def test_create_limit_uses_encoded_bytes_and_preserves_newlines(tmp_path):
    content = "\u4e2d\u6587"
    tool = CreateFileTool(_boundary(tmp_path), max_bytes=6)
    result = tool.execute(_state(), tool.input_model(path="exact.txt", content=content))
    assert result.value["bytes"] == 6
    assert (tmp_path / "exact.txt").read_bytes() == content.encode("utf-8")
    tool = CreateFileTool(_boundary(tmp_path))
    result = tool.execute(_state(), tool.input_model(
        path="newlines.txt", content="one\r\ntwo\rthree\n",
    ))
    assert (tmp_path / "newlines.txt").read_bytes() == b"one\r\ntwo\rthree\n"
    assert result.value["bytes"] == 15


@pytest.mark.parametrize("original,old,new,max_bytes,error", [
    (b"a\xffb", "a", "A", 10, "UTF-8"),
    (b"a\x00b", "a", "A", 10, "NUL"),
    (b"abcd", "a", "\u4e2d", 5, "too large"),
    (b"abcdef", "a", "", 5, "too large"),
    (b"abc", "a", "\x00", 10, "NUL"),
    (b"abc", "a", "\ud800", 10, "UTF-8"),
    (b"abc", "\ud800", "x", 10, "UTF-8"),
])
def test_replace_rejection_leaves_bytes_mode_revision_and_directory_unchanged(
    tmp_path, original, old, new, max_bytes, error,
):
    path = tmp_path / "text.txt"
    path.write_bytes(original)
    path.chmod(0o640)
    state = _state()
    tool = ReplaceTextTool(_boundary(tmp_path), max_bytes=max_bytes)
    with pytest.raises(ValueError, match=error):
        tool.execute(state, tool.input_model(path=path.name, old_text=old, new_text=new))
    assert path.read_bytes() == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert state.memory == {}
    assert list(tmp_path.iterdir()) == [path]


def test_replace_preserves_unmodified_newlines_and_permissions(tmp_path):
    path = tmp_path / "mixed.txt"
    path.write_bytes(b"head\r\nTARGET\r\nnext\rtail\n")
    path.chmod(0o750)
    tool = ReplaceTextTool(_boundary(tmp_path))
    result = tool.execute(_state(), tool.input_model(
        path=path.name, old_text="TARGET\r\n", new_text="changed\r\n",
    ))
    assert path.read_bytes() == b"head\r\nchanged\r\nnext\rtail\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o750
    assert result.ok
    assert list(tmp_path.iterdir()) == [path]


def test_search_reports_large_non_text_and_unreadable_files(tmp_path, monkeypatch):
    from sciagent_capabilities.workspace import search_text

    (tmp_path / "large.txt").write_bytes(b"needle" * 5)
    (tmp_path / "nul.bin").write_bytes(b"needle\x00")
    (tmp_path / "invalid.bin").write_bytes(b"needle\xff")
    (tmp_path / "unreadable.txt").write_bytes(b"needle")
    (tmp_path / "ok.txt").write_bytes(b"needle")
    real_read = search_text.read_text_file

    def read(path, *, max_bytes):
        if path.name == "unreadable.txt":
            raise OSError("do not expose absolute internal paths")
        return real_read(path, max_bytes=max_bytes)

    monkeypatch.setattr(search_text, "read_text_file", read)
    tool = SearchTextTool(_boundary(tmp_path), max_bytes=10)
    result = tool.execute(_state(), tool.input_model(query="needle"))
    assert result.value["matches"] == [{"path": str(tmp_path / "ok.txt"), "line": 1, "text": "needle"}]
    assert result.value["skipped_count"] == 4
    assert result.value["skipped_files"] == [
        {"path": str(tmp_path / "invalid.bin"), "reason": "not_utf8_text"},
        {"path": str(tmp_path / "large.txt"), "reason": "too_large"},
        {"path": str(tmp_path / "nul.bin"), "reason": "not_utf8_text"},
        {"path": str(tmp_path / "unreadable.txt"), "reason": "read_error"},
    ]
    assert result.value["incomplete"] is True
    assert result.value["truncated"] is False
    assert result.value["skipped_files_truncated"] is False
    assert result.memory_updates["read_paths"] == [str(tmp_path / "ok.txt")]


def test_search_zero_matches_with_skipped_files_is_explicitly_incomplete(tmp_path):
    (tmp_path / "large.txt").write_bytes(b"x" * 21)
    tool = SearchTextTool(_boundary(tmp_path), max_bytes=20)
    result = tool.execute(_state(), tool.input_model(query="missing"))
    assert result.value["matches"] == []
    assert result.value["incomplete"] is True
    assert result.value["skipped_count"] == 1
    assert result.memory_updates["read_paths"] == []
    assert "incomplete" in result.summary


def test_search_skip_details_are_bounded_and_do_not_expose_denied_files(tmp_path):
    for number in range(55):
        (tmp_path / f"{number:02d}.bin").write_bytes(b"\x00")
    (tmp_path / "denied").mkdir()
    (tmp_path / "denied" / "secret.bin").write_bytes(b"\x00")
    tool = SearchTextTool(_boundary(tmp_path))
    result = tool.execute(_state(), tool.input_model(query="missing"))
    assert result.value["skipped_count"] == 55
    assert len(result.value["skipped_files"]) == 50
    assert result.value["skipped_files_truncated"] is True
    assert result.value["incomplete"] is True
    assert all(not Path(item["path"]).is_relative_to(tmp_path / "denied") for item in result.value["skipped_files"])


@pytest.mark.parametrize("match_count", [1, 2])
def test_search_result_limit_explicitly_marks_search_incomplete(tmp_path, match_count):
    (tmp_path / "text.txt").write_text("match\n" * match_count, encoding="utf-8")
    tool = SearchTextTool(_boundary(tmp_path))
    result = tool.execute(_state(), tool.input_model(query="match", max_results=1))
    assert result.value["matches"] == [{"path": str(tmp_path / "text.txt"), "line": 1, "text": "match"}]
    assert result.value["truncated"] is True
    assert result.value["incomplete"] is True
    assert result.value["skipped_count"] == 0
    assert "result limit reached" in result.summary
    assert "further matches not checked" in result.summary
    assert "total match count unknown" in result.summary


def test_search_does_not_swallow_changed_path_authorization(tmp_path, monkeypatch):
    boundary = _boundary(tmp_path)
    monkeypatch.setattr(boundary, "iter_files", lambda path: ["denied/secret.txt"])
    tool = SearchTextTool(boundary)
    with pytest.raises(WorkspacePermissionError):
        tool.execute(_state(), tool.input_model(query="missing"))


def test_search_reports_file_disappearing_after_authorized_enumeration(tmp_path, monkeypatch):
    boundary = _boundary(tmp_path)
    monkeypatch.setattr(boundary, "iter_files", lambda path: ["disappeared.txt"])
    tool = SearchTextTool(boundary)
    result = tool.execute(_state(), tool.input_model(query="missing"))
    assert result.value["skipped_files"] == [
        {"path": str(tmp_path / "disappeared.txt"), "reason": "read_error"},
    ]
    assert result.value["incomplete"] is True


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_search_invalid_limit_is_a_configuration_error(tmp_path, max_bytes):
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        SearchTextTool(_boundary(tmp_path), max_bytes=max_bytes)
