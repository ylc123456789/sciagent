"""Shared UTF-8 text limits, bounded workspace reads, and window projection."""

import codecs
from collections.abc import Iterable, Iterator
from pathlib import Path
import re


# Workspace processing bound, separate from tool output and model input.
MAX_WORKSPACE_TEXT_BYTES = 10 * 1024 * 1024


# Raw tool-result IO bound, not the model input budget. Context projections
# pack these records against each Agent's effective input limit.
MAX_READ_CHARS = 128_000


_NUL_TEXT_ERROR = (
    "Cannot read as UTF-8 text: file contains NUL bytes. "
    "Use a format-aware reader or authorized processing to inspect the file."
)
_INVALID_UTF8_TEXT_ERROR = (
    "Cannot read as UTF-8 text: file contains invalid UTF-8 bytes. "
    "Use a format-aware reader or authorized processing to inspect the file."
)
_LINE_ENDINGS = re.compile(r"\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]")


class TextFileTooLargeError(ValueError):
    """A workspace text file exceeds the configured processing bound."""


def decode_text(content: bytes) -> str:
    """Read UTF-8 without lossy substitution or NUL-bearing binary content."""
    if b"\x00" in content:
        raise ValueError(_NUL_TEXT_ERROR)
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(_INVALID_UTF8_TEXT_ERROR) from None


def read_text_file(
    path: Path, *, max_bytes: int = MAX_WORKSPACE_TEXT_BYTES,
) -> str:
    """Validate one bounded workspace text file while preserving its newlines."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if path.stat().st_size > max_bytes:
        raise TextFileTooLargeError(f"file is too large to process as text: limit {max_bytes} bytes")
    with path.open("rb") as handle:
        content = handle.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise TextFileTooLargeError(f"file is too large to process as text: limit {max_bytes} bytes")
    return decode_text(content)


def encode_text(
    text: str, *, max_bytes: int = MAX_WORKSPACE_TEXT_BYTES,
) -> bytes:
    """Validate UTF-8 workspace output before touching the destination."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if "\x00" in text:
        raise ValueError("Cannot write as UTF-8 text: content contains NUL bytes.")
    try:
        content = text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Cannot write as UTF-8 text: content contains invalid Unicode characters.") from None
    if len(content) > max_bytes:
        raise TextFileTooLargeError(f"file is too large to write as text: limit {max_bytes} bytes")
    return content


def wrap_text_lines(text: str, *, max_chars: int = 1_000) -> str:
    """Insert line breaks without dropping source characters or whitespace."""
    return "\n".join(
        "\n".join(line[start:start + max_chars] for start in range(0, len(line), max_chars))
        for line in text.split("\n")
    )


def validate_text_window(
    *, start_line: int | None = None, end_line: int | None = None,
    start_char: int = 0, end_char: int | None = None,
) -> None:
    """Check one-based inclusive lines and zero-based exclusive characters."""
    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be greater than or equal to 1")
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be greater than or equal to 1")
    if start_line is not None and end_line is not None and end_line < start_line:
        raise ValueError("end_line must be greater than or equal to start_line")
    if start_char < 0 or (end_char is not None and end_char <= start_char):
        raise ValueError("character range must satisfy 0 <= start_char < end_char")


def slice_text_lines(
    text: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    start_char: int = 0,
    end_char: int | None = None,
    max_chars: int = MAX_READ_CHARS,
) -> dict:
    """Select physical lines, then a character window, then bound tool output.

    Request coordinates remain unchanged. ``next_start_char`` identifies the
    next source character in the same line selection, regardless of truncation.
    ``total_lines`` counts all physical lines in the source and
    ``selected_chars`` counts characters in the selected line range before
    applying the character window.
    """
    validate_text_window(
        start_line=start_line, end_line=end_line,
        start_char=start_char, end_char=end_char,
    )
    lines = text.splitlines(keepends=True)
    selected_lines = "".join(lines[(start_line or 1) - 1 : end_line])
    selected = selected_lines[start_char:end_char]
    content = selected[:max_chars]
    returned_end = start_char + len(content)
    return {
        "start_line": start_line,
        "end_line": end_line,
        "start_char": start_char,
        "end_char": end_char,
        "total_lines": len(lines),
        "selected_chars": len(selected_lines),
        "content": content,
        "truncated": len(selected) > max_chars,
        "next_start_char": returned_end if returned_end < len(selected_lines) else None,
    }


def decode_text_chunks(chunks: Iterable[bytes]) -> Iterator[str]:
    """Decode bounded chunks, validating the whole source before accepting it.

    Consume the remaining bytes after a decoding failure so the source can
    finish its integrity check and a later NUL still takes precedence, as with
    decode_text. Source-iterator errors likewise precede text-format errors.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    contains_nul = invalid_utf8 = False
    for chunk in chunks:
        contains_nul |= b"\x00" in chunk
        if not invalid_utf8:
            try:
                decoded = decoder.decode(chunk)
            except UnicodeDecodeError:
                invalid_utf8 = True
            else:
                if decoded:
                    yield decoded
    if not invalid_utf8:
        try:
            tail = decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            invalid_utf8 = True
        else:
            if tail:
                yield tail
    if contains_nul:
        raise ValueError(_NUL_TEXT_ERROR)
    if invalid_utf8:
        raise ValueError(_INVALID_UTF8_TEXT_ERROR)


def slice_text_chunks(
    chunks: Iterable[str], *, start_line: int | None = None,
    end_line: int | None = None, start_char: int = 0,
    end_char: int | None = None, max_chars: int = MAX_READ_CHARS,
) -> dict:
    """Project the same window as slice_text_lines without retaining whole lines.

    Memory is bounded by the incoming chunk and requested output. A trailing CR
    is carried across chunks so CRLF remains one physical line ending; the other
    separators match str.splitlines(keepends=True), including Unicode endings.
    All chunks are consumed, even after the requested window has been filled.
    """
    validate_text_window(start_line=start_line, end_line=end_line,
                         start_char=start_char, end_char=end_char)
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    first_line = start_line or 1
    current_line = 1
    selected_chars = 0
    has_tail = False
    pieces = []
    output_end = start_char + max_chars
    if end_char is not None:
        output_end = min(output_end, end_char)

    def consume(text: str) -> None:
        nonlocal current_line, selected_chars, has_tail
        if not text:
            return
        begin = 0 if (current_line >= first_line
                      and (end_line is None or current_line <= end_line)) else None
        stop = len(text)
        last_break = 0
        for ending in _LINE_ENDINGS.finditer(text):
            last_break = ending.end()
            if current_line == end_line:
                stop = last_break
            current_line += 1
            if current_line == first_line:
                begin = last_break
        has_tail = last_break < len(text)
        if begin is None:
            return
        size = stop - begin
        lower = max(0, start_char - selected_chars)
        upper = min(size, output_end - selected_chars)
        if lower < upper:
            pieces.append(text[begin + lower:begin + upper])
        selected_chars += size

    pending_cr = ""
    for chunk in chunks:
        text = pending_cr + chunk
        pending_cr = "\r" if text.endswith("\r") else ""
        consume(text[:-1] if pending_cr else text)
    consume(pending_cr)
    content = "".join(pieces)
    requested_end = selected_chars if end_char is None else min(selected_chars, end_char)
    requested_chars = max(0, requested_end - start_char)
    returned_end = start_char + len(content)
    return {
        "start_line": start_line, "end_line": end_line,
        "start_char": start_char, "end_char": end_char,
        "total_lines": current_line - 1 + int(has_tail),
        "selected_chars": selected_chars, "content": content,
        "truncated": requested_chars > max_chars,
        "next_start_char": returned_end if returned_end < selected_chars else None,
    }
