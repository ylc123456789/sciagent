"""Trusted deterministic fixtures stay separate from production preset resolution."""
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from resagent2_contracts import ExecutionLimits, WorkRequest, WorkRequestDraft, Workflow, WorkflowTask
from resagent2_executor import default_execution_registry
from resagent2_orchestrator import CompilationError, DeterministicWorkflowCompiler, FixedWorkflowCompiler


def request(**task_changes):
    now = datetime.now(UTC)
    task = {"key": "work", "instruction": "Preserve this exact task",
            "selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Implementation required"},
            **task_changes}
    return WorkRequest(id="work_fixture", run_id="run_fixture",
        request=WorkRequestDraft(objective="Trusted fixture", tasks=[task]), created_at=now, updated_at=now)


def test_deterministic_compiler_preserves_trusted_submission():
    work = request()
    registry = default_execution_registry()
    args = {"registry": registry, "limits": ExecutionLimits()}
    proposal = FixedWorkflowCompiler().compile(work, current=None, **args).output
    compiler = DeterministicWorkflowCompiler(proposal)
    assert compiler.compile(work, current=None, **args).output == proposal
    current = Workflow(run_id=work.run_id, revision=1, created_from=work.id,
        tasks=[WorkflowTask(**task.model_dump(exclude={"acceptance_spec", "output_names"})) for task in proposal.tasks])
    with pytest.raises(CompilationError, match="no patch"):
        compiler.compile(work, current=current, **args)


@pytest.mark.parametrize("depends_on,match", [(["missing"], "unknown"), (["work"], "cycle")])
def test_invalid_original_graph_is_rejected_before_compilation(depends_on, match):
    with pytest.raises(ValidationError, match=match):
        request(depends_on=depends_on)
