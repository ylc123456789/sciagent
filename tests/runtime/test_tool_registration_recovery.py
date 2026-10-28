"""Reopen real Run/Session stores across registration and receipt cut points."""

import json
from datetime import UTC, datetime

import pytest

from sciagent_components import RegisteredArtifactReader, material_index_context, request_task_context
from sciagent_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ArtifactCandidate, ResearchRequest, RunBudget, RunPermissions, TaskBudget,
)
from sciagent_orchestrator import JsonRunStore, ResearchRun
from sciagent_orchestrator.artifacts import ArtifactRegistry, ScientificArtifactRegistration
from sciagent_runtime import (
    AgentDefinition, AgentLoop, AllowListPermissionPolicy, CompletionDecision,
    ContextSection, FinishTool, JsonSessionStore, NativeToolCall, ScriptedLLMClient, ToolCallTurn, ToolObservation,
)
from tool_fixtures import WriteValueTool


class _ProcessStopped(BaseException):
    pass


class _Complete:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None, report="done")


@pytest.mark.parametrize("cut_point", ["allocation_before_freeze", "receipt_before_ack"])
def test_reopen_consumes_registration_once_without_replaying_the_tool(tmp_path, monkeypatch, cut_point):
    run_root, session_root, artifact_root = (tmp_path / name for name in ("runs", "sessions", "artifacts"))
    run_store = JsonRunStore(run_root)
    now = datetime.now(UTC)
    run_store.save(ResearchRun(
        run_id="run_recovery", request=ResearchRequest(goal="Preserve one source",
            budget=RunBudget(max_llm_calls=10, timeout_seconds=60), permissions=RunPermissions()),
        status="running", created_at=now, updated_at=now,
    ))
    dispatched = []

    def definition(port, client, acknowledge):
        class RegisterTool(WriteValueTool):
            def execute(self, state, arguments):
                dispatched.append(state.current_operation_key)
                ref = port.register_scientific(
                    ArtifactCandidate(kind="web_page", path="source.md", media_type="text/markdown",
                                      summary="Original source", content="EXACT_SAVED_SOURCE"),
                    run_id=state.run_id, session_id=state.session_id,
                    operation_key=state.current_operation_key, output_position=0,
                )
                return ToolObservation(summary="Material saved", artifacts=[ref], value={"artifact_id": ref.id})

        tools = (RegisterTool(), FinishTool())
        reader = RegisteredArtifactReader([], run_id="run_recovery", resolve=lambda artifact_id: port.resolve(
            artifact_id, run_id="run_recovery", session_id="session_recovery",
        ))

        def context(request, state, limit):
            return [
                ContextSection(name="task", content=json.dumps(request_task_context(request))),
                material_index_context(request, state, reader=reader),
            ]

        return AgentDefinition(
            name="scientific", owner=AgentOwner.SCIENTIFIC, system_prompt="Fixed source preservation task",
            tools=tools, llm_client=client, context_builder=context,
            permission_policy=AllowListPermissionPolicy({tool.name for tool in tools}), completion_check=_Complete(),
            acknowledge_registration=acknowledge, recover_registration=port.recover_registration,
        )

    request = AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1,
        run_id="run_recovery", agent="scientific", instruction="Preserve one source", permissions=AgentPermissions(),
        budget=TaskBudget(max_llm_calls=10, timeout_seconds=60),
    )
    port = ScientificArtifactRegistration(ArtifactRegistry(artifact_root), run_store)
    client = ScriptedLLMClient([ToolCallTurn(tool_calls=[NativeToolCall(
        id="save", name="write_value", arguments='{"key":"source","value":"source"}',
    )])])

    def stop(**kwargs):
        raise _ProcessStopped()

    with monkeypatch.context() as patch:
        acknowledge = port.acknowledge_registration
        if cut_point == "allocation_before_freeze":
            patch.setattr("sciagent_orchestrator.artifacts.tempfile.mkdtemp", lambda **kwargs: stop())
        else:
            acknowledge = stop
        with pytest.raises(_ProcessStopped):
            AgentLoop(store=JsonSessionStore(session_root)).run(
                definition(port, client, acknowledge), request, session_id="session_recovery",
            )
    before = JsonSessionStore(session_root).load("session_recovery")
    old_receipt = before.tool_turns[0].tool_results.get("save")
    allocated = JsonRunStore(run_root).load(request.run_id)
    assert len(allocated.pending_registrations) == 1
    assert not allocated.scientific_artifact_ids

    reopened_run_store = JsonRunStore(run_root)
    reopened_port = ScientificArtifactRegistration(ArtifactRegistry(artifact_root), reopened_run_store)
    resumed_client = ScriptedLLMClient([ToolCallTurn(tool_calls=[NativeToolCall(
        id="finish", name="finish", arguments='{"report":"done"}',
    )])])
    result = AgentLoop(store=JsonSessionStore(session_root)).run(
        definition(reopened_port, resumed_client, reopened_port.acknowledge_registration),
        request.model_copy(update={"parent_session_id": "session_recovery"}), session_id="unused",
    )
    assert result.status == "completed"
    assert len(dispatched) == 1
    final = reopened_run_store.load(request.run_id)
    state = JsonSessionStore(session_root).load("session_recovery")
    receipt = state.tool_turns[0].tool_results["save"]
    assert not final.pending_registrations
    assert len(state.tool_turns[0].tool_results) == 1
    if cut_point == "receipt_before_ack":
        assert receipt == old_receipt
        assert len(final.scientific_artifact_ids) == len(state.authorized_artifact_ids) == 1
        assert final.scientific_artifact_ids == state.authorized_artifact_ids
        assert state.authorized_artifact_ids[0] in resumed_client.contexts[0].material_index
    else:
        assert json.loads(receipt)["ok"] is False
        assert not final.artifacts
        assert not final.scientific_artifact_ids
        assert not state.authorized_artifact_ids
