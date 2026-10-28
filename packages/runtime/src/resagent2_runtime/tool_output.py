"""Bound model-visible receipts without changing their authoritative records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


# Leave room for an existing 128000-character read window and its metadata.
MAX_TOOL_RECEIPT_CHARS = 160_000
_MAX_PREVIEW_ITEMS = 50
_MAX_OMISSION_NOTES = 16
_STATUS_FIELDS = (
    "status", "execution_status", "exit_code", "timed_out", "stop_confirmed", "operation",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _preview(value: Any, *, text_limit: int, item_limit: int,
             omissions: list[dict], path: str = "", depth: int = 0) -> Any:
    if isinstance(value, str) and len(value) > text_limit:
        # A visibly different type prevents a shortened identity/path from
        # looking like an exact, usable ID. Text windows are handled below.
        return {"text_preview": value[:text_limit], "original_chars": len(value),
                "display_truncated": True}
    if isinstance(value, (dict, list)) and depth >= 12:
        return {"display_omitted": True, "reason": "nested display limit"}
    if isinstance(value, dict):
        return {key: _preview(item, text_limit=text_limit, item_limit=item_limit,
                              omissions=omissions, path=f"{path}/{key[:80]}", depth=depth + 1)
                for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > item_limit and len(omissions) < _MAX_OMISSION_NOTES:
            omissions.append({"field": path[:240], "total_items": len(value),
                              "shown_items": item_limit})
        return [_preview(item, text_limit=text_limit, item_limit=item_limit,
                         omissions=omissions, path=f"{path}/{index}", depth=depth + 1)
                for index, item in enumerate(value[:item_limit])]
    return value


def _read_window(original: Any, projected: Any) -> None:
    """Keep a shortened character window's continuation at the first unseen character."""
    if not isinstance(original, dict) or not isinstance(projected, dict):
        return
    content = original.get("content")
    shown = projected.get("content")
    if (isinstance(content, str) and isinstance(shown, dict)
            and "text_preview" in shown and type(original.get("start_char")) is int
            and "next_start_char" in original and "selected_chars" in original):
        projected["content"] = shown["text_preview"]
        projected["truncated"] = True
        projected["next_start_char"] = original["start_char"] + len(projected["content"])


def model_tool_receipt(receipt: str, full_result: dict | None = None) -> str:
    """Project a valid, bounded tool message; leave ordinary receipts byte-exact."""
    if len(receipt) <= MAX_TOOL_RECEIPT_CHARS:
        return receipt
    try:
        original = json.loads(receipt)
    except (ValueError, RecursionError):
        original = None
    if not isinstance(original, dict):
        return _json({"display": {"truncated": True, "original_chars": len(receipt),
                                  "full_result": full_result},
                      "text_preview": receipt[:MAX_TOOL_RECEIPT_CHARS // 8]})
    value = original.get("value")
    notice = {
        "truncated": True,
        "original_chars": len(receipt),
        "note": (
            "Partial model display only; the original receipt remains in the runtime record. "
            "Use full_result.path with read_file when available; read bounded windows and follow continuation positions. "
            "Do not treat previews as complete evidence or rerun side effects to recover omitted output."
        ),
    }
    if full_result is not None:
        notice["full_result"] = full_result
    # An opaque next-page cursor cannot be corrected after dropping entries.
    # Omit such a page rather than expose a cursor that skips unseen entries.
    paginated = isinstance(value, dict) and (
        "next_cursor" in value or (
            isinstance(value.get("expansion"), dict) and isinstance(value.get("children"), list)
        )
    )
    if not paginated:
        text_limit, item_limit = MAX_TOOL_RECEIPT_CHARS // 2, _MAX_PREVIEW_ITEMS
        while text_limit >= 128:
            omissions: list[dict] = []
            projected = _preview(original, text_limit=text_limit, item_limit=item_limit,
                                 omissions=omissions)
            _read_window(value, projected.get("value"))
            projected["display"] = {**notice, "array_previews": omissions}
            rendered = _json(projected)
            if len(rendered) <= MAX_TOOL_RECEIPT_CHARS:
                return rendered
            text_limit //= 2
            item_limit = max(1, item_limit // 2)

    # Pathological dictionaries (e.g. huge keys) must not bypass the hard bound.
    # Keep exact small execution facts, never fabricate completion or cursors.
    minimal = {key: original[key] for key in (
        "ok", "observed_at", "control", "execution_status",
    ) if key in original and len(_json(original[key])) <= 1024}
    summary = original.get("summary", "")
    minimal["summary"] = summary[:1024] if isinstance(summary, str) else "Oversized tool receipt"
    minimal["value"] = {
        key: value[key] for key in _STATUS_FIELDS
        if isinstance(value, dict) and key in value and len(_json(value[key])) <= 1024
    }
    if isinstance(value, dict) and isinstance(value.get("completed_command"), dict):
        completed = value["completed_command"]
        minimal["value"]["completed_command"] = {
            key: completed[key] for key in _STATUS_FIELDS
            if key in completed and len(_json(completed[key])) <= 1024
        }
    minimal["value"]["display_omitted"] = True
    if paginated:
        notice["retry_hint"] = (
            "The page was not displayed. Repeat this read-only listing with the same input cursor "
            "and a smaller page size; do not advance to an unseen page's next cursor."
        )
    if "user_comment" in original:
        notice["user_comment_omitted"] = True
    minimal["display"] = notice
    return _json(minimal)


# Each text part has at most 2M Unicode characters / 8MB UTF-8 bytes, below
# read_file's 10MiB processing bound. Parts concatenate byte-for-byte to the
# original public receipt. A small manifest is published only after every part.
_SNAPSHOT_PART_CHARS = 2_000_000


def _write_snapshot(path: Path, content: str) -> None:
    data = content.encode("utf-8")
    if path.is_symlink():
        raise OSError("Tool-result snapshot must not be a symbolic link")
    if path.exists():
        if path.read_bytes() != data:
            raise OSError("Tool-result snapshot content conflicts with its identity")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".receipt-", delete=False) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_tool_receipt(directory: str, receipt: str, *, turn_index: int, call_id: str) -> dict:
    """Publish a read-only snapshot reference; never export runtime-private data."""
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise OSError("Tool-result directory must not be a symbolic link")
    digest = hashlib.sha256()
    byte_count = 0
    for start in range(0, len(receipt), _SNAPSHOT_PART_CHARS):
        data = receipt[start:start + _SNAPSHOT_PART_CHARS].encode("utf-8")
        digest.update(data)
        byte_count += len(data)
    content_hash = digest.hexdigest()
    call_hash = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:20]
    stem = f"{turn_index:06d}-{call_hash}-{content_hash}"
    total_parts = max(1, (len(receipt) + _SNAPSHOT_PART_CHARS - 1) // _SNAPSHOT_PART_CHARS)
    parts = []
    for part in range(total_parts):
        content = receipt[part * _SNAPSHOT_PART_CHARS:(part + 1) * _SNAPSHOT_PART_CHARS]
        path = root / (f"{stem}.json" if total_parts == 1 else f"{stem}.part-{part + 1:06d}.txt")
        _write_snapshot(path, content)
        parts.append({"path": str(path), "chars": len(content)})
    reference = {"available": True, "path": parts[0]["path"], "format": "json",
                 "sha256": content_hash, "total_chars": len(receipt), "total_bytes": byte_count}
    if total_parts > 1:
        manifest = root / f"{stem}.manifest.json"
        body = _json({"format": "concatenated-json-text", "sha256": content_hash,
                      "total_chars": len(receipt), "total_bytes": byte_count, "parts": parts})
        if len(body.encode("utf-8")) > 8_000_000:
            raise OSError("Tool-result manifest exceeds the supported read bound")
        _write_snapshot(manifest, body)
        reference.update(path=str(manifest), format="manifest", part_count=total_parts)
    return reference


def sync_tool_receipt_files(state, memo: dict) -> bool:
    """Best-effort export after raw facts are durable; errors are display facts."""
    if state.tool_result_dir is None:
        return False
    changed = False
    for ordinal, turn in enumerate(state.tool_turns):
        for call_id, receipt in turn.tool_results.items():
            if len(receipt) <= MAX_TOOL_RECEIPT_CHARS:
                continue
            call = next((item for item in turn.tool_calls if item.id == call_id), None)
            if call is not None and call.name == "read_file":
                # The source snapshot already has an exact path and continuation.
                # Escaped read windows must not create a chain of wrapper copies.
                try:
                    value = json.loads(receipt).get("value")
                    source = value.get("path") if isinstance(value, dict) else None
                    if (isinstance(source, str) and Path(source).is_absolute()
                            and Path(source).resolve().is_relative_to(Path(state.tool_result_dir).resolve())):
                        continue
                except (OSError, ValueError, RecursionError):
                    pass
            key = (state.session_id, ordinal, call_id)
            if key in memo and memo[key] == receipt:
                continue
            try:
                reference = export_tool_receipt(state.tool_result_dir, receipt,
                                                turn_index=ordinal, call_id=call_id)
            except (OSError, UnicodeError, ValueError) as error:
                reference = {"available": False, "error": type(error).__name__,
                             "note": "Full-result export failed; the original receipt and actual outcome are retained."}
            if turn.tool_result_files.get(call_id) != reference:
                turn.tool_result_files[call_id] = reference
                changed = True
            memo[key] = receipt
    return changed
