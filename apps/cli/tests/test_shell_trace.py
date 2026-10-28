"""A live trace reader must retain records the writer has not finished."""

import json

import pytest

from resagent2_cli.shell import TraceTail


def encoded_record(**values):
    return json.dumps(values, ensure_ascii=False).encode("utf-8") + b"\n"


def test_trace_tail_waits_for_record_newline_and_returns_it_once(tmp_path):
    path = tmp_path / "trace.jsonl"
    record = {"run_id": "run_live", "message": "complete JSON, unfinished line"}
    line = encoded_record(**record)
    path.write_bytes(line[:-1])
    tail = TraceTail(path)

    assert tail.new_records() == []
    assert tail.new_records() == []
    assert tail.offset == 0

    with path.open("ab") as handle:
        handle.write(line[-1:])
    assert tail.new_records() == [record]
    assert tail.offset == len(line)
    assert tail.new_records() == []


@pytest.mark.parametrize("character", ["中", "😀"])
def test_trace_tail_keeps_partial_utf8_after_complete_records(tmp_path, character):
    path = tmp_path / "trace.jsonl"
    first = {"run_id": "run_live", "message": "first"}
    second = {"run_id": "run_live", "message": character}
    first_line = encoded_record(**first)
    second_line = encoded_record(**second)
    split = second_line.index(character.encode("utf-8")) + 1
    path.write_bytes(first_line + second_line[:split])
    tail = TraceTail(path)

    assert tail.new_records("run_live") == [first]
    assert tail.offset == len(first_line)
    assert tail.new_records("run_live") == []

    with path.open("ab") as handle:
        handle.write(second_line[split:])
    assert tail.new_records("run_live") == [second]
    assert tail.offset == len(first_line) + len(second_line)
    assert tail.new_records("run_live") == []


def test_trace_tail_consumes_filtered_records_and_retains_partial_line(tmp_path):
    path = tmp_path / "trace.jsonl"
    other = encoded_record(run_id="run_other", message="跳过")
    wanted = {"run_id": "run_live", "message": "wanted"}
    line = encoded_record(**wanted)
    path.write_bytes(other + line[:10])
    tail = TraceTail(path)

    assert tail.new_records("run_live") == []
    assert tail.offset == len(other)
    with path.open("ab") as handle:
        handle.write(line[10:])
    assert tail.new_records("run_live") == [wanted]
    assert tail.new_records("run_live") == []


def test_trace_tail_still_skips_blank_and_invalid_complete_json_lines(tmp_path):
    path = tmp_path / "trace.jsonl"
    skipped = b"invalid JSON\n\n"
    record = {"run_id": "run_live", "message": "valid"}
    line = encoded_record(**record)
    path.write_bytes(skipped + line + b'{"unfinished":')
    tail = TraceTail(path)

    assert tail.new_records() == [record]
    assert tail.offset == len(skipped + line)
    assert tail.new_records() == []


def test_trace_tail_skips_invalid_utf8_complete_line_and_reads_following_record(tmp_path):
    path = tmp_path / "trace.jsonl"
    invalid = b'{"message": "\xff"}\n'
    record = {"message": "完整记录"}
    line = encoded_record(**record)
    path.write_bytes(invalid + line)
    tail = TraceTail(path)

    assert tail.new_records() == [record]
    assert tail.offset == len(invalid + line)
    assert tail.new_records() == []


def test_trace_tail_restarts_after_file_is_truncated(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_bytes(encoded_record(message="old record " * 30))
    tail = TraceTail(path)
    assert len(tail.new_records()) == 1
    replacement = {"message": "new"}
    line = encoded_record(**replacement)
    path.write_bytes(line)

    assert tail.new_records() == [replacement]
    assert tail.offset == len(line)
    assert tail.new_records() == []


def test_trace_tail_reset_skips_existing_records(tmp_path):
    path = tmp_path / "trace.jsonl"
    old = encoded_record(message="old")
    path.write_bytes(old)
    tail = TraceTail(path)
    tail.reset()
    assert tail.new_records() == []
    record = {"message": "新的记录"}
    with path.open("ab") as handle:
        handle.write(encoded_record(**record))
    assert tail.new_records() == [record]
    assert tail.new_records() == []


def test_trace_tail_handles_missing_and_empty_files(tmp_path):
    path = tmp_path / "trace.jsonl"
    tail = TraceTail(path)
    tail.reset()
    assert tail.new_records() == []
    path.touch()
    assert tail.new_records() == []
    record = {"message": "first"}
    path.write_bytes(encoded_record(**record))
    assert tail.new_records() == [record]
