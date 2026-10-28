"""Durable controls are independent of owner-written Run/Session snapshots."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from resagent2_runtime.controls import RunControlInbox


def test_controls_persist_and_owner_cursor_filters_without_deleting(tmp_path):
    inbox = RunControlInbox(tmp_path)
    first = inbox.submit("run_a", "input", "  original text\nline two  ")
    second = inbox.submit("run_a", "interrupt")
    (tmp_path / "run_a" / ".unfinished.tmp").write_text("partial")
    restarted = RunControlInbox(tmp_path)
    assert restarted.pending("run_a") == [first, second]
    assert restarted.pending("run_a", [first.event_id]) == [second]
    assert len(list((tmp_path / "run_a").glob("*.json"))) == 2
    assert restarted.pending("run_b") == []


def test_concurrent_submitters_publish_unique_complete_records(tmp_path):
    inbox = RunControlInbox(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        submitted = list(pool.map(lambda n: inbox.submit("run_a", "input", str(n)), range(24)))
    received = inbox.pending("run_a")
    assert len(received) == 24
    assert len({record.event_id for record in received}) == 24
    assert {record.text for record in received} == {str(n) for n in range(24)}
    assert set(received_record.event_id for received_record in received) == {r.event_id for r in submitted}
    assert [r.event_id for r in received] == sorted(r.event_id for r in received)


@pytest.mark.parametrize("run_id", ["../run", "/absolute", "a/b", "a\\b", ".", ".."])
def test_controls_cannot_escape_root(tmp_path, run_id):
    with pytest.raises(ValueError, match="invalid run ID"):
        RunControlInbox(tmp_path).submit(run_id, "input", "text")
    assert list(tmp_path.iterdir()) == []


def test_input_rejects_empty_text_without_publishing(tmp_path):
    with pytest.raises(ValueError, match="contain text"):
        RunControlInbox(tmp_path).submit("run_a", "input", "  ")
    assert list(tmp_path.iterdir()) == []


def test_identity_mismatch_is_not_silently_consumed(tmp_path):
    inbox = RunControlInbox(tmp_path)
    record = inbox.submit("run_a", "input", "text")
    path = tmp_path / "run_a" / f"{record.event_id}.json"
    data = json.loads(path.read_text())
    data["run_id"] = "run_other"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        inbox.pending("run_a")


@pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX concurrent publication ordering")
def test_slow_first_publication_cannot_be_overtaken(tmp_path, monkeypatch):
    import os
    from threading import Event

    inbox = RunControlInbox(tmp_path)
    first_publishing = Event()
    release_first = Event()
    second_started = Event()
    reader_started = Event()
    replace = os.replace

    def delayed_replace(source, destination):
        if json.loads(source.read_text())["text"] == "first":
            first_publishing.set()
            assert release_first.wait(2)
        replace(source, destination)

    monkeypatch.setattr("resagent2_runtime.controls.os.replace", delayed_replace)

    def second_submitter():
        second_started.set()
        return inbox.submit("run_a", "input", "second")

    def snapshot_reader():
        reader_started.set()
        return inbox.pending("run_a")

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(inbox.submit, "run_a", "input", "first")
        assert first_publishing.wait(2)
        second = pool.submit(second_submitter)
        reader = pool.submit(snapshot_reader)
        try:
            assert second_started.wait(2) and reader_started.wait(2)
            assert not second.done() and not reader.done()
        finally:
            release_first.set()
        first_record, second_record = first.result(timeout=2), second.result(timeout=2)
        snapshot = reader.result(timeout=2)
    assert first_record.event_id < second_record.event_id
    assert snapshot[0] == first_record
    assert inbox.pending("run_a") == [first_record, second_record]
