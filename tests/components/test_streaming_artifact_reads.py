"""Frozen text windows validate every byte without retaining the whole source."""

import hashlib
import io
import random
import tracemalloc
from pathlib import Path

import pytest

from sciagent_components import ArtifactReadError, RegisteredArtifactReader
from sciagent_components.text import decode_text_chunks, slice_text_chunks, slice_text_lines
from sciagent_contracts import AgentOwner, ArtifactRef
from sciagent_runtime.cancellation import (
    CancellationToken, OperationCancelled, bind_cancellation,
)


def artifact(path, *, digest=None):
    if digest is None:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return ArtifactRef(
        id="artifact_text", kind="text", producer=AgentOwner.EXECUTOR,
        run_id="run_reader", task_id="task_reader", attempt_number=1,
        uri=path.as_uri(), sha256=digest, media_type="text/plain", summary="Frozen evidence",
    )


def read(path, **options):
    ref = artifact(path)
    return RegisteredArtifactReader([ref], run_id=ref.run_id).read_text(ref.id, **options)


def stream_window(text, size, **options):
    raw = text.encode("utf-8")
    chunks = (raw[index:index + size] for index in range(0, len(raw), size))
    return slice_text_chunks(decode_text_chunks(chunks), **options)


def test_stream_windows_match_original_algorithm_for_unicode_and_all_line_endings():
    randomizer = random.Random(20261021)
    alphabet = "ab中😀\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"
    sources = ["", "a", "\r", "\n", "\r\n", "\r\r\n\r", alphabet]
    sources += ["".join(randomizer.choices(alphabet, k=randomizer.randrange(100)))
                for _ in range(30)]
    windows = [
        {}, {"start_line": 1, "end_line": 1},
        {"start_line": 2, "end_line": 4, "start_char": 1},
        {"end_line": 3, "start_char": 2, "end_char": 9},
        {"start_line": 3, "start_char": 20, "end_char": 50},
        {"start_line": 100, "end_line": 200},
        {"start_char": 1000},
    ]
    # Tiny byte chunks deliberately split CRLF and every multibyte codepoint.
    for text in sources:
        for size in (1, 2, 3, 7, 64):
            for window in windows:
                for max_chars in (1, 5, 128):
                    options = {**window, "max_chars": max_chars}
                    assert stream_window(text, size, **options) == slice_text_lines(text, **options)


@pytest.mark.parametrize("boundary", [65533, 65534, 65535, 65536])
@pytest.mark.parametrize("separator", ["\r\n", "\u2028", "\u2029", "\x85", "😀"])
def test_artifact_windows_preserve_byte_chunk_boundaries(tmp_path, boundary, separator):
    body = "a" * boundary + separator + "中文\r\ntail\r"
    path = tmp_path / "boundary.txt"
    path.write_bytes(body.encode("utf-8"))
    options = {"start_line": 1, "end_line": 2, "start_char": boundary - 2, "max_chars": 12}
    expected = slice_text_lines(body, **options)
    actual = read(path, **options)
    assert {key: actual[key] for key in expected} == expected


@pytest.mark.parametrize("tail,error", [
    (b"\x00", "NUL bytes"), (b"\xff", "invalid UTF-8"),
    (b"\xf0\x9f\x98", "invalid UTF-8"),
    (b"\xff" + b"x" * 65536 + b"\x00", "NUL bytes"),
])
def test_window_still_validates_invalid_bytes_after_selected_content(tmp_path, tail, error):
    path = tmp_path / "invalid.bin"
    path.write_bytes(b"selected\r\n" + b"x" * 140000 + tail)
    ref = artifact(path)
    reader = RegisteredArtifactReader([ref], run_id=ref.run_id)
    with pytest.raises(ArtifactReadError, match=error):
        reader.read_text(ref.id, start_line=1, end_line=1, max_chars=3)
    # Integrity failure takes precedence over text format, even beyond the window.
    reader = RegisteredArtifactReader([ref.model_copy(update={"sha256": "0" * 64})], run_id=ref.run_id)
    with pytest.raises(ArtifactReadError, match="sha256"):
        reader.read_text(ref.id, start_line=1, end_line=1, max_chars=3)


def test_reader_uses_bounded_reads_and_never_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / "huge-line.txt"
    body = ("中文😀" * 100000 + "END").encode("utf-8")
    path.write_bytes(body)
    ref = artifact(path)
    sizes = []
    original_open = Path.open

    class BoundedFile(io.BufferedReader):
        def read(self, size=-1):
            assert 0 < size <= 65536
            sizes.append(size)
            return super().read(size)

    def opening(source, *args, **kwargs):
        if source == path and args == ("rb",):
            return BoundedFile(io.FileIO(source, "r"))
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("unbounded whole-file read"))
    actual = RegisteredArtifactReader([ref], run_id=ref.run_id).read_text(
        ref.id, start_char=299998, max_chars=5,
    )
    assert actual["content"] == "文😀END"
    assert actual["selected_chars"] == 300003
    assert actual["total_lines"] == 1
    assert actual["next_start_char"] is None
    assert len(sizes) > 1


@pytest.mark.parametrize("options", [
    {"start_line": 0}, {"start_line": 2, "end_line": 1},
    {"start_char": -1}, {"start_char": 4, "end_char": 4},
    {"max_chars": 0}, {"max_chars": -1}, {"max_chars": True},
])
def test_bad_windows_are_rejected_before_opening_source(tmp_path, monkeypatch, options):
    path = tmp_path / "text.txt"
    path.write_bytes(b"source")
    ref = artifact(path)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("invalid window read bytes"))
    with pytest.raises(ValueError):
        RegisteredArtifactReader([ref], run_id=ref.run_id).read_text(ref.id, **options)


def test_cancelled_stream_closes_the_file_before_returning(tmp_path, monkeypatch):
    path = tmp_path / "cancel.txt"
    path.write_bytes(b"x" * 200000)
    ref = artifact(path)
    token = CancellationToken()
    handles = []
    original_open = Path.open

    class CancellingFile(io.BufferedReader):
        def read(self, size=-1):
            chunk = super().read(size)
            token.cancel("stop reading")
            return chunk

    def opening(source, *args, **kwargs):
        if source == path and args == ("rb",):
            handle = CancellingFile(io.FileIO(source, "r"))
            handles.append(handle)
            return handle
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    with bind_cancellation(token), pytest.raises(OperationCancelled, match="stop reading"):
        RegisteredArtifactReader([ref], run_id=ref.run_id).read_text(ref.id, max_chars=3)
    assert handles and all(handle.closed for handle in handles)


def test_peak_memory_does_not_grow_with_unselected_giant_line(tmp_path):
    peaks = []
    for megabytes in (1, 32):
        path = tmp_path / f"giant-{megabytes}.txt"
        with path.open("wb") as handle:
            for _ in range(megabytes * 16):
                handle.write(b"x" * 65536)
        ref = artifact(path)
        reader = RegisteredArtifactReader([ref], run_id=ref.run_id)
        tracemalloc.start()
        try:
            result = reader.read_text(ref.id, start_char=megabytes * 1024 * 1024 - 256,
                                      max_chars=512)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result["content"] == "x" * 256
        assert result["selected_chars"] == megabytes * 1024 * 1024
        assert result["total_lines"] == 1
        assert result["truncated"] is False
        peaks.append(peak)
    assert peaks[1] < 2 * 1024 * 1024
    assert peaks[1] <= peaks[0] + 512 * 1024


@pytest.mark.parametrize("failure_after", [0, 1])
def test_stream_io_error_is_preserved_without_partial_success(tmp_path, monkeypatch, failure_after):
    path = tmp_path / "io-error.bin"
    path.write_bytes(b"selected\n" + b"x" * 200000 + b"\x00")
    ref = artifact(path)
    handles = []
    original_open = Path.open

    class FailingFile(io.BufferedReader):
        def __init__(self, raw):
            super().__init__(raw)
            self.calls = 0

        def read(self, size=-1):
            if self.calls == failure_after:
                raise OSError("injected source read failure")
            self.calls += 1
            return super().read(size)

    def opening(source, *args, **kwargs):
        if source == path and args == ("rb",):
            handle = FailingFile(io.FileIO(source, "r"))
            handles.append(handle)
            return handle
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    with pytest.raises(OSError, match="injected source read failure"):
        RegisteredArtifactReader([ref], run_id=ref.run_id).read_text(
            ref.id, start_line=1, end_line=1, max_chars=3,
        )
    assert handles and all(handle.closed for handle in handles)


def test_stream_checks_shared_deadline_without_charging_model_usage(tmp_path, monkeypatch):
    from sciagent_runtime.budget import DeadlineExceededError, execution_budget

    path = tmp_path / "deadline.txt"
    path.write_bytes(b"x" * 200000)
    ref = artifact(path)
    now = [0.0]
    handles = []
    original_open = Path.open

    class AdvancingFile(io.BufferedReader):
        def read(self, size=-1):
            value = super().read(size)
            now[0] = 2.0
            return value

    def opening(source, *args, **kwargs):
        if source == path and args == ("rb",):
            handle = AdvancingFile(io.FileIO(source, "r"))
            handles.append(handle)
            return handle
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    with execution_budget(max_llm_calls=0, timeout_seconds=1, clock=lambda: now[0]) as budget:
        with pytest.raises(DeadlineExceededError):
            RegisteredArtifactReader([ref], run_id=ref.run_id).read_text(ref.id, max_chars=3)
        assert budget.usage.used == 0
    assert handles and all(handle.closed for handle in handles)


def test_zero_remaining_model_calls_does_not_prevent_local_stream_read(tmp_path):
    from sciagent_runtime.budget import execution_budget

    path = tmp_path / "local.txt"
    path.write_bytes(b"local text")
    with execution_budget(max_llm_calls=0, timeout_seconds=60) as budget:
        assert read(path, max_chars=5)["content"] == "local"
        assert budget.usage.used == 0
