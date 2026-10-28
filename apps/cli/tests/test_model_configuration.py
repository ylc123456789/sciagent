"""Deployment output headroom must not expand module input or retry policies."""

from e2e.native_fixtures import tool_turns

from datetime import UTC, datetime
import httpx
import json

import pytest

from sciagent_cli import composition
from sciagent_contracts import ExecutionLimits, WorkRequest, WorkRequestDraft
from sciagent_runtime.budget import execution_budget


@pytest.fixture
def model_config_scope():
    with execution_budget(max_llm_calls=10, timeout_seconds=900):
        yield
from sciagent_orchestrator import DeterministicWorkInterpreter
from sciagent_orchestrator import FixedWorkflowCompiler
from sciagent_runtime import DEFAULT_AGENT_CONTEXT_TOKENS, ComposedContext, ScriptedLLMClient


@pytest.fixture
def defaults(monkeypatch):
    for name in (
        "CONTEXT_WINDOW", "RESERVED_OUTPUT_TOKENS", "CONTEXT_SAFETY_MARGIN_TOKENS",
        "LLM_TIMEOUT_SECONDS", "MODEL", "API_BASE", "API_KEY_ENV", "LLM_TRACE_LEVEL",
        "LLM_TRACE_DIR", "SCIENTIFIC_CONTEXT_TOKENS", "CODING_CONTEXT_TOKENS",
        "EXPERIMENT_CONTEXT_TOKENS", "COMPILER_CONTEXT_TOKENS",
        "PERMISSION_REVIEWER_CONTEXT_TOKENS",
    ):
        monkeypatch.delenv(f"SCIAGENT_{name}", raising=False)
    monkeypatch.setenv("SCIAGENT_API_KEY_ENV", "TEST_CONFIG_KEY")
    monkeypatch.setenv("TEST_CONFIG_KEY", "test-only")


@pytest.mark.parametrize("component,limit", [
    ("scientific", 256_000), ("coding", 256_000),
    ("experiment", 256_000),
    ("permission_reviewer", 256_000),
])
def test_output_headroom_preserves_every_module_input_limit(defaults, component, limit):
    client = composition._client()
    assert client.model_profile.context_window == 1_000_000
    assert client.model_profile.reserved_output_tokens == 256_000
    assert client.model_profile.safety_margin_tokens == 1024
    assert client.timeout_seconds == 600
    assert composition._component_context_limit(component) == limit
    assert client.tool_input_limit(limit) == limit


def test_real_e2e_translation_uses_the_shared_cli_default(defaults, monkeypatch, tmp_path):
    from e2e import real_e2e

    monkeypatch.setattr(composition, "_client", lambda: ScriptedLLMClient([]))
    controller = real_e2e._application(tmp_path).controller

    assert isinstance(controller.compiler, FixedWorkflowCompiler)
    assert not hasattr(controller.compiler, "_client")
    assert {d.config.max_context_tokens for d in controller.registry.execution_definitions.values()} == {DEFAULT_AGENT_CONTEXT_TOKENS}


@pytest.mark.parametrize("entry", ["cli", "real_e2e"])
def test_composition_uses_fixed_interpreter_without_a_model_client(defaults, monkeypatch, tmp_path, entry):
    clients = []

    def client():
        value = ScriptedLLMClient([])
        clients.append(value)
        return value

    if entry == "cli":
        monkeypatch.setattr(composition, "_client", client)
        controller = composition.build_application(data_root=tmp_path).controller
    else:
        from e2e import real_e2e
        monkeypatch.setattr(composition, "_client", client)
        controller = real_e2e._application(tmp_path).controller

    assert isinstance(controller.interpreter, DeterministicWorkInterpreter)
    # Scientific and reviewer are eager; execution profile is resolved per invocation.
    assert len(clients) == 2
    assert isinstance(controller.compiler, FixedWorkflowCompiler)
    assert not hasattr(controller.compiler, "_client")


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_cli_wire_uses_configured_output_cap_not_an_input_cap(defaults, monkeypatch, model, model_config_scope):
    monkeypatch.setenv("SCIAGENT_MODEL", model)
    seen = []

    def respond(request, *, timeout, max_response_bytes):
        assert max_response_bytes is None
        seen.append((json.loads(request.content), timeout))
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
            "id": "call_config", "type": "function",
            "function": {"name": "finish", "arguments": '{"report":"done"}'},
        }]}}]}, request=request)

    monkeypatch.setattr("sciagent_runtime.llm.send_request", respond)
    client = composition._client()
    client.next_tool_call(ComposedContext(
        text="Select the next tool", included_sections=["system"], omitted_sections=[], estimated_tokens=3,
    ), [], [], max_input_tokens=256_000)
    assert len(seen) == client.last_attempts == 1
    body, timeout = seen[0]
    assert body["max_tokens"] == 256_000
    assert body["model"] == model
    assert timeout == 600
    assert "thinking" not in body and "reasoning_effort" not in body


def test_custom_model_can_override_capacity_output_and_timeout(defaults, monkeypatch):
    for key, value in {
        "MODEL": "custom-model", "CONTEXT_WINDOW": "32000",
        "RESERVED_OUTPUT_TOKENS": "3000", "LLM_TIMEOUT_SECONDS": "90",
        "CODING_CONTEXT_TOKENS": "12000",
    }.items():
        monkeypatch.setenv(f"SCIAGENT_{key}", value)
    client = composition._client()
    assert client.model == "custom-model"
    assert client.model_profile.context_window == 32000
    assert client.model_profile.reserved_output_tokens == 3000
    assert client.timeout_seconds == 90
    assert client.tool_input_limit(composition._component_context_limit("coding")) == 12000


@pytest.mark.parametrize("key,value", [
    ("RESERVED_OUTPUT_TOKENS", "0"), ("RESERVED_OUTPUT_TOKENS", "invalid"),
    ("LLM_TIMEOUT_SECONDS", "0"), ("LLM_TIMEOUT_SECONDS", "invalid"),
    ("CONTEXT_WINDOW", "65536"),  # A stale small capacity cannot fit the new reservation.
])
def test_invalid_capacity_configuration_fails_before_network(defaults, monkeypatch, key, value):
    monkeypatch.setenv(f"SCIAGENT_{key}", value)
    monkeypatch.setattr("sciagent_runtime.llm.send_request", lambda *a, **k: pytest.fail("no network"))
    with pytest.raises(ValueError):
        composition._client()


def test_full_cli_compilation_is_fixed_and_has_no_model_request(defaults, monkeypatch, tmp_path, model_config_scope):
    monkeypatch.setenv("SCIAGENT_LLM_TRACE_LEVEL", "full")
    monkeypatch.setenv("SCIAGENT_LLM_TRACE_DIR", str(tmp_path / "trace"))
    # An obsolete compiler override must not affect composition or create a client.
    monkeypatch.setenv("SCIAGENT_COMPILER_CONTEXT_TOKENS", "invalid")
    monkeypatch.setattr("sciagent_runtime.llm.send_request", lambda *a, **k: pytest.fail("no compiler HTTP"))
    controller = composition.build_application(data_root=tmp_path / "data").controller
    now = datetime.now(UTC)
    instruction = "  Measure the method using the existing code and report uncertainty.\n"
    request = WorkRequest(id="work_config", run_id="run_config", owner_session_id="session_config",
        owner_task_id="task_root", owner_attempt_number=1, depth=2,
        request=WorkRequestDraft(objective="Measure the method", tasks=[{
            "key": "work", "instruction": instruction,
            "selection": {"kind": "preset", "preset_ref": "experiment@1", "reason": "Uses existing code for measurement"},
        }]), created_at=now, updated_at=now)
    result = controller.compiler.compile(request, current=None, registry=controller.registry,
                                         limits=ExecutionLimits())
    assert result.llm_calls == 0
    assert result.output.tasks[0].instruction == instruction
    assert result.output.tasks[0].execution_ref == "experiment@1"
    assert not (tmp_path / "trace" / "llm_traces.jsonl").exists()


@pytest.mark.parametrize("level", ["off", "metadata", "full"])
def test_composition_reports_effective_trace_configuration(defaults, monkeypatch, tmp_path, capsys, level):
    monkeypatch.setenv("SCIAGENT_LLM_TRACE_LEVEL", level)
    monkeypatch.setenv("SCIAGENT_LLM_TRACE_DIR", str(tmp_path / "traces"))
    composition.build_application(data_root=tmp_path / "data")
    assert capsys.readouterr().err == (
        f"LLM trace: {level}; directory: {tmp_path / 'traces'}\n"
    )
