"""Real E2E uses the production fixed resolver, with no routing model call."""
from datetime import UTC, datetime
import pytest
from e2e import real_e2e
from resagent2_cli import composition
from resagent2_contracts import ExecutionLimits, WorkRequest, WorkRequestDraft
from resagent2_orchestrator import FixedWorkflowCompiler


@pytest.mark.parametrize("repeats", [1, 50000])
def test_fixed_compiler_preserves_instruction_without_model_budget(monkeypatch, tmp_path, repeats):
    class NoModel:
        def next_tool_call(self, *args, **kwargs):
            pytest.fail("configuration binding must not call an LLM")
    monkeypatch.setattr(composition, "_client", lambda: NoModel())
    monkeypatch.setenv("RESAGENT2_WEB_SEARCH_PROVIDER", "off")
    controller = real_e2e._application(tmp_path).controller
    assert isinstance(controller.compiler, FixedWorkflowCompiler)
    instruction = "  Measure the requested result.\n" + " detail" * repeats
    now = datetime.now(UTC)
    work = WorkRequest(id="work_binding", run_id="run_binding",
        request=WorkRequestDraft(objective="Measure", tasks=[{
            "key": "measure", "instruction": instruction,
            "selection": {"kind": "preset", "preset_ref": "experiment@1", "reason": "Measurement"},
        }]), created_at=now, updated_at=now)
    result = controller.compiler.compile(work, current=None,
        registry=controller.registry, limits=ExecutionLimits())
    assert result.llm_calls == 0
    assert result.output.tasks[0].instruction == instruction
    assert result.output.tasks[0].execution_ref == "experiment@1"
