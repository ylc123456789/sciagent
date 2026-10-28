"""Bound workspace reads by actual bytes without narrowing frozen evidence."""

import hashlib
from pathlib import Path

import pytest

from sciagent_components import RegisteredArtifactReader
from sciagent_components.text import (
    MAX_WORKSPACE_TEXT_BYTES,
    TextFileTooLargeError,
    read_text_file,
    slice_text_lines,
)
from sciagent_contracts import AgentOwner, ArtifactRef


def test_workspace_read_rejects_growth_after_size_precheck(tmp_path, monkeypatch):
    path = tmp_path / "growing.txt"
    path.write_bytes(b"small")
    original_stat = Path.stat
    grown = False

    def growing_stat(source, *args, **kwargs):
        nonlocal grown
        result = original_stat(source, *args, **kwargs)
        if source == path and not grown:
            grown = True
            path.write_bytes(b"x" * 100)
        return result

    monkeypatch.setattr(Path, "stat", growing_stat)
    with pytest.raises(TextFileTooLargeError, match="too large"):
        read_text_file(path, max_bytes=10)
    assert path.read_bytes() == b"x" * 100


def test_workspace_processing_limit_does_not_restrict_registered_text(tmp_path):
    path = tmp_path / "frozen.txt"
    content = b"selected\r\n" + b"x" * MAX_WORKSPACE_TEXT_BYTES
    path.write_bytes(content)
    ref = ArtifactRef(
        id="artifact_text", kind="text", producer=AgentOwner.EXECUTOR,
        run_id="run_reader", task_id="task_reader", attempt_number=1,
        uri=path.as_uri(), sha256=hashlib.sha256(content).hexdigest(),
        media_type="text/plain", summary="Frozen evidence",
    )
    reader = RegisteredArtifactReader([ref], run_id=ref.run_id)
    result = reader.read_text(ref.id, start_line=1, end_line=1)
    assert result["content"] == "selected\r\n"
    assert result["truncated"] is False


@pytest.mark.parametrize("end_char", [None, 100, 10])
def test_clipped_window_continues_at_returned_boundary(end_char):
    body = "0123456789abcdefghij"
    first = slice_text_lines(body, start_char=2, end_char=end_char, max_chars=5)
    assert first["content"] == "23456"
    assert first["start_char"] == 2
    assert first["end_char"] == end_char
    assert first["truncated"] is True
    assert first["next_start_char"] == 7
    second = slice_text_lines(body, start_char=first["next_start_char"], max_chars=5)
    assert second["content"] == "789ab"


@pytest.mark.parametrize("max_chars", [5, 20])
def test_complete_explicit_window_is_not_the_end_of_selected_lines(max_chars):
    first = slice_text_lines("0123456789", end_char=5, max_chars=max_chars)
    assert first["content"] == "01234"
    assert first["truncated"] is False
    assert first["next_start_char"] == 5
    second = slice_text_lines("0123456789", start_char=first["next_start_char"], max_chars=5)
    assert second["content"] == "56789"
    assert second["next_start_char"] is None


@pytest.mark.parametrize("body,start_char,end_char", [
    ("01234", 0, 5),
    ("01234", 0, None),
    ("01234", 2, 100),
    ("01234", 5, None),
    ("01234", 10, None),
    ("", 0, None),
])
def test_selected_line_end_and_empty_windows_have_no_continuation(body, start_char, end_char):
    result = slice_text_lines(body, start_char=start_char, end_char=end_char, max_chars=5)
    assert result["content"] == body[start_char:end_char]
    assert result["truncated"] is False
    assert result["next_start_char"] is None


@pytest.mark.parametrize(
    "body,start_line,end_line,expected_lines,expected_chars",
    [
        ("", None, None, 0, 0),
        ("\n", None, None, 1, 1),
        ("one", None, None, 1, 3),
        ("one\r\ntwo\n", None, None, 2, 9),
        ("one\r\ntwo\n", 2, 2, 2, 4),
        ("one\r\ntwo\n", 8, 10, 2, 0),
    ],
)
def test_read_window_reports_source_and_selected_sizes(
    body, start_line, end_line, expected_lines, expected_chars,
):
    result = slice_text_lines(body, start_line=start_line, end_line=end_line)
    assert result["total_lines"] == expected_lines
    assert result["selected_chars"] == expected_chars


@pytest.mark.parametrize("window_size", [3, 7])
def test_continuation_rebuilds_unicode_with_fixed_line_coordinates(window_size):
    selected = "中文😀abc\r\nαβγdelta\r\n"
    body = "not selected\r\n" + selected + "also not selected\r\n"
    pieces = []
    start_char = 0
    while True:
        result = slice_text_lines(
            body, start_line=2, end_line=3,
            start_char=start_char, end_char=start_char + window_size, max_chars=3,
        )
        assert result["start_line"] == 2
        assert result["end_line"] == 3
        assert result["content"] == selected[start_char:start_char + 3]
        pieces.append(result["content"])
        next_start = result["next_start_char"]
        if next_start is None:
            break
        assert next_start == start_char + len(result["content"])
        assert next_start > start_char
        start_char = next_start
    assert "".join(pieces) == selected


@pytest.mark.parametrize(
    "start_char,end_char,max_chars",
    [
        (0, None, 3), (0, 3, 10), (3, 8, 3), (8, None, 3), (20, None, 3),
    ],
)
def test_range_sizes_are_not_clipped_by_character_window(start_char, end_char, max_chars):
    body = "header\r\n中文😀abc\r\nfooter\r\n"
    result = slice_text_lines(
        body, start_line=2, end_line=2, start_char=start_char,
        end_char=end_char, max_chars=max_chars,
    )
    assert result["total_lines"] == 3
    assert result["selected_chars"] == len("中文😀abc\r\n")


def test_continuation_is_relative_to_selected_lines_not_the_whole_file():
    result = slice_text_lines("header\r\n中文😀\r\nfooter\r\n", start_line=2, end_line=2, max_chars=3)
    assert result["content"] == "中文😀"
    assert result["next_start_char"] == 3
    tail = slice_text_lines(
        "header\r\n中文😀\r\nfooter\r\n", start_line=2, end_line=2,
        start_char=result["next_start_char"], max_chars=3,
    )
    assert tail["content"] == "\r\n"
    assert tail["next_start_char"] is None
