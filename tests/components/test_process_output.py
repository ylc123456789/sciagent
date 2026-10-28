"""Durable shell output previews distinguish source limits from display limits."""

from pathlib import Path

import pytest

from resagent2_components.process import output_preview
from resagent2_components.text import MAX_WORKSPACE_TEXT_BYTES


def test_output_preview_returns_short_source_complete_with_exact_scale(tmp_path):
    path = tmp_path / "stdout.log"
    text = "first\n中文: last\n"
    path.write_bytes(text.encode())
    receipt = output_preview(path, limit=len(text))
    assert receipt["path"] == str(path)
    assert receipt["total_bytes"] == len(text.encode())
    assert receipt["total_chars"] == len(text)
    assert receipt["content"] == text
    assert receipt["preview_chars"] == len(text)
    assert receipt["preview_kind"] == "complete"
    assert receipt["truncated"] is False
    assert receipt["omitted_chars"] == 0
    assert receipt["text_status"] == "supported"
    assert receipt["decoding_replaced"] is False
    assert "head" not in receipt and "tail" not in receipt


@pytest.mark.parametrize("limit", [1, 8, 9])
def test_output_preview_preserves_both_ends_and_counts_unicode_characters(tmp_path, limit):
    path = tmp_path / "stdout.log"
    text = "开始😀" + "middle" * 10 + "结束😀"
    path.write_bytes(text.encode())
    receipt = output_preview(path, limit=limit)
    head_chars, tail_chars = (limit + 1) // 2, limit // 2
    assert receipt["head"] == text[:head_chars]
    assert receipt["tail"] == (text[-tail_chars:] if tail_chars else "")
    assert receipt["preview_chars"] == limit
    assert receipt["omitted_chars"] == len(text) - limit
    assert receipt["preview_kind"] == "head_tail"
    assert receipt["truncated"] is True
    assert "content" not in receipt


@pytest.mark.parametrize("raw,status,replaced", [
    (b"head" + b"x" * 100 + b"\xff" + b"x" * 100 + b"tail", "invalid_utf8", True),
    (b"head" + b"x" * 100 + b"\x00" + b"x" * 100 + b"tail", "contains_nul", False),
    ("head�tail".encode(), "supported", False),
])
def test_output_preview_checks_full_bounded_source_not_only_visible_ends(tmp_path, raw, status, replaced):
    path = tmp_path / "stdout.log"
    path.write_bytes(raw)
    receipt = output_preview(path, limit=8)
    assert receipt["text_status"] == status
    assert receipt["decoding_replaced"] is replaced
    assert receipt["total_bytes"] == len(raw)
    assert path.read_bytes() == raw


def test_large_log_only_reads_bounded_windows_and_reports_text_limit(tmp_path, monkeypatch):
    path = tmp_path / "stdout.log"
    with path.open("wb") as handle:
        handle.write("开始😀".encode())
        handle.seek(MAX_WORKSPACE_TEXT_BYTES + 1)
        handle.write("结束😀".encode())
    size = path.stat().st_size
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __enter__(self):
            self.handle = original_open(path, "rb")
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def seek(self, *args):
            return self.handle.seek(*args)

        def read(self, size=-1):
            reads.append(size)
            assert 0 <= size <= 24
            return self.handle.read(size)

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: BoundedReader())
    receipt = output_preview(path, limit=9)
    assert receipt["total_bytes"] == size
    assert receipt["text_status"] == "too_large"
    assert receipt["total_chars"] is None and receipt["omitted_chars"] is None
    assert receipt["head"].startswith("开始😀")
    assert receipt["tail"].endswith("结束😀")
    assert receipt["preview_chars"] == 9
    assert receipt["text_max_bytes"] == MAX_WORKSPACE_TEXT_BYTES
    assert receipt["changed_during_read"] is False
    assert len(reads) == 2


def test_unavailable_log_is_not_reported_as_empty_output(tmp_path):
    receipt = output_preview(tmp_path / "missing.log")
    assert receipt["text_status"] == "unavailable"
    assert receipt["preview_kind"] == "unavailable"
    assert "FileNotFoundError" in receipt["error"]
    assert "content" not in receipt and "total_bytes" not in receipt


@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
def test_output_preview_rejects_invalid_limits(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        output_preview(tmp_path / "missing", limit=limit)
