"""Configuration identity extends prompt/schema matching without changing lookup."""
from dataclasses import replace

import pytest

from e2e.native_fixtures import tool_turns
from sciagent_contracts import AgentRequest, ExecutionConfig, ExecutionDefinition
from sciagent_runtime import (AgentDefinition, AgentLoop, AllowListPermissionPolicy,
    CompletionDecision, FinishTool, InMemorySessionStore, ScriptedLLMClient)


def definition(completion="artifacts@1", ref="analysis@1"):
    return ExecutionDefinition(ref=ref, config=ExecutionConfig(
        system_prompt="Finish the bounded task", tools=["finish"], preparation="workspace@1",
        completion=completion, finalizer="execution@1", context_builder="executor@1",
        implementation_version="1", model_profile="fixture@1", max_context_tokens=256000))


def request(frozen, parent=None):
    return AgentRequest(run_id="run_config", work_request_id="work_config", task_id="task_config",
        attempt_number=1, agent="executor", instruction="Preserve the result", permissions={},
        budget={"max_llm_calls": 3, "timeout_seconds": 30}, execution_definition=frozen,
        parent_session_id=parent)


class Complete:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=True, report=candidate.report)


def profile(frozen, client, finalizer=None):
    return AgentDefinition(name="executor", owner="executor", system_prompt=frozen.config.system_prompt,
        tools=(FinishTool(),), llm_client=client, context_builder=lambda *args: [],
        permission_policy=AllowListPermissionPolicy({"finish"}), completion_check=Complete(),
        finalize_result=finalizer, execution_ref=frozen.ref,
        execution_config_hash=frozen.config_hash)


@pytest.mark.parametrize("changed", ["ref", "strategy"])
def test_resume_rejects_config_drift_even_when_prompt_and_schema_unchanged(changed):
    original = definition()
    store = InMemorySessionStore()
    client = ScriptedLLMClient(tool_turns([{"tool": "finish", "arguments": {"report": "Original"}}]))
    loop = AgentLoop(store=store)
    first = loop.run(profile(original, client), request(original), session_id="session_config")
    assert first.status == "completed"
    before = store.load("session_config").model_dump(mode="json")
    altered = definition(ref="other@1") if changed == "ref" else definition(completion="artifacts@2")
    empty_client = ScriptedLLMClient([])
    result = loop.run(profile(altered, empty_client), request(altered, "session_config"), session_id="session_config")
    assert result.status == "failed"
    assert "resume target" in result.report
    assert empty_client.contexts == []
    assert store.load("session_config").model_dump(mode="json") == before


def test_request_and_factory_binding_must_match_before_any_model_or_session_write():
    original = definition()
    store = InMemorySessionStore()
    client = ScriptedLLMClient([])
    wrong = replace(profile(original, client), execution_config_hash="0" * 64)
    result = AgentLoop(store=store).run(wrong, request(original), session_id="session_config")
    assert result.status == "failed"
    assert "frozen request" in result.report
    assert client.contexts == []
    assert not store.exists("session_config")


def test_saved_terminal_lookup_preserves_configuration_without_rerunning_factory_or_finalizer():
    frozen = definition()
    calls = []
    def finalize(result, state):
        calls.append(state.execution_ref)
        return result
    client = ScriptedLLMClient(tool_turns([{"tool": "finish", "arguments": {"report": "Original"}}]))
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    loop.run(profile(frozen, client, finalize), request(frozen), session_id="session_config")
    assert calls == [frozen.ref]
    before = store.load("session_config").model_dump(mode="json")
    assert before["execution_ref"] == frozen.ref
    assert before["execution_config_hash"] == frozen.config_hash
    result = loop.read_terminal_result(session_id="session_config", run_id="run_config",
        task_id="task_config", attempt_number=1, owner="executor")
    assert result.report == "Original"
    assert result.llm_calls == 0
    assert calls == [frozen.ref]
    assert store.load("session_config").model_dump(mode="json") == before
