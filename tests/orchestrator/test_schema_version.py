"""A schema break rejects old runs without rewriting their audit records."""

from resagent2_contracts import RunPermissions, ExecutionLimits

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from resagent2_contracts import (
    ResearchRequest,
    RunBudget,
    RunStatus,
    SCHEMA_VERSION,
)
from resagent2_orchestrator import JsonRunStore, ResearchRun


@pytest.mark.parametrize("old_version", ["7.0", "13.0", "14.0", "15.0", "16.0", "20.0", "21.0", "22.0", "23.0", "24.0", "25.0", "26.0", "27.0", "28.0", "29.0", "30.0"])
def test_old_run_rejected_without_rewriting_file(tmp_path, old_version):
    now = datetime.now(UTC)
    run = ResearchRun(run_id='run_schema', status=RunStatus.RUNNING, request=ResearchRequest(goal='Version boundary', budget=RunBudget(max_llm_calls=1, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1)), created_at=now, updated_at=now)
    store = JsonRunStore(tmp_path)
    store.save(run)
    assert store.load(run.run_id) == run

    path = tmp_path / "run_schema.json"
    legacy = path.read_bytes().replace(f'"{SCHEMA_VERSION}"'.encode(), f'"{old_version}"'.encode())
    path.write_bytes(legacy)
    with pytest.raises(ValidationError):
        store.load(run.run_id)
    assert path.read_bytes() == legacy
