"""A second CLI owner cannot consume input or overwrite paused Run state."""

from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from resagent2_contracts import PendingQuestion, ResearchRequest, RunStatus, UserAnswer, RunBudget, RunPermissions
from resagent2_orchestrator import JsonRunStore, ResearchController, ResearchRun
from resagent2_runtime.controls import RunControlInbox


@pytest.mark.skipif(os.name != "posix", reason="Controller POSIX owner lock")
@pytest.mark.parametrize("operation", ["resume", "answer"])
def test_second_process_cannot_mutate_owner_state(tmp_path, operation):
    now = datetime.now(UTC)
    run = ResearchRun(run_id="run_locked", request=ResearchRequest(goal="Keep original goal", budget=RunBudget(max_llm_calls=10, timeout_seconds=60), permissions=RunPermissions()),
        status=RunStatus.PAUSED, created_at=now, updated_at=now,
        pending_question=PendingQuestion(id="question_locked", run_id="run_locked", session_id="session_locked",
            text="Choose direction", requested_fields=["direction"], created_at=now))
    store = JsonRunStore(tmp_path / "state")
    store.save(run)
    snapshot = tmp_path / "state" / "run_locked.json"
    before = snapshot.read_bytes()
    inbox = RunControlInbox(tmp_path / "controls")
    event = inbox.submit(run.run_id, "input", "Queued supplement")
    lock_path = inbox.root / run.run_id / "owner.lock"
    script = """
import fcntl, sys
with open(sys.argv[1], 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    print('ready', flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(lock_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        controller = ResearchController.__new__(ResearchController)
        controller.control_inbox = inbox
        controller.scheduler = SimpleNamespace(store=store)
        with pytest.raises(ValueError, match="active execution owner"):
            if operation == "resume":
                controller.resume_run(run.run_id)
            else:
                controller.answer_question(run.run_id, UserAnswer(question_id=run.pending_question.id,
                    values={"direction": "Continue"}, answered_at=now))
        assert snapshot.read_bytes() == before
        assert inbox.pending(run.run_id) == [event]
    finally:
        process.communicate(input="done\n", timeout=2)
