"""Scientific can inspect current files without registering them as evidence."""

import json
from datetime import UTC, datetime

from e2e.native_fixtures import tool_turns
from sciagent_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, RecordedAnswer, TaskBudget,
    WorkspaceAccess, WorkspaceGrant,
)
from sciagent_runtime import ScriptedLLMClient
from sciagent_scientific import ScientificAgent


def request(*, grants=(), **kwargs):
    return AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1,
        run_id="run_parent_read", agent=AgentOwner.SCIENTIFIC,
        instruction="Resolve the active task's question", input_artifacts=[],
        budget=TaskBudget(max_llm_calls=5, timeout_seconds=60),
        permissions=AgentPermissions(), read_workspaces=list(grants), **kwargs,
    )


def grant(root):
    return WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(read_paths=["."]))


def ask():
    return {"tool": "ask_user", "arguments": {
        "assessment": {"statement": "Need a user decision"},
        "text": "Should the unfinished task continue?", "requested_fields": ["decision"],
    }}


def read(path, **kwargs):
    return {"tool": "read_file", "arguments": {"path": str(path), **kwargs}}


def receipt(state, turn):
    item = state.tool_turns[turn]
    return json.loads(item.tool_results[item.tool_calls[0].id])


def test_scientific_reads_current_file_in_third_part_without_creating_artifact(tmp_path):
    path = tmp_path / "partial.log"
    path.write_bytes(b"first\r\nsecond\r\n")
    client = ScriptedLLMClient(tool_turns([read(path, start_line=2, end_line=2), ask()]))
    agent = ScientificAgent(client)
    result = agent.invoke(request(grants=[grant(tmp_path)]))
    assert result.status == "needs_user_input", result.report
    state = agent.store.load(result.session.id)
    observed = receipt(state, 0)
    assert observed["ok"] is True
    assert observed["value"]["content"] == "second\r\n"
    assert result.artifacts == []
    assert state.authorized_artifact_ids == []
    assert "second" not in client.contexts[-1].material_index
    assert "second" not in client.contexts[-1].task_context
    assert "second" in str(client.histories[-1][0].tool_results)
    names = {schema["function"]["name"] for schema in client.schemas[0]}
    assert "read_file" in names
    assert "list_files" in names
    assert not {"create_file", "replace_text", "delete_path", "run_shell", "search_text"} & names


def test_read_tool_contract_stays_fixed_when_runtime_grants_change(tmp_path):
    path = tmp_path / "partial.txt"
    path.write_text("current")
    client = ScriptedLLMClient(tool_turns([ask(), read(path), ask()]))
    agent = ScientificAgent(client)
    first = agent.invoke(request())
    question = first.control.question
    answer = RecordedAnswer(task_id=question.task_id, attempt_number=question.attempt_number,
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"decision": "inspect the current file"}, answered_at=datetime.now(UTC),
        run_id=question.run_id, session_id=question.session_id,
    )
    resumed = agent.invoke(request(
        grants=[grant(tmp_path)], parent_session_id=first.session.id,
        input_ids=[answer.question_id], user_answers=[answer], resume_answers=[answer], control_ack=first.control.origin,
    ))
    assert resumed.status == "needs_user_input", resumed.report
    assert receipt(agent.store.load(first.session.id), 1)["value"]["content"] == "current"
    assert client.schemas[0] == client.schemas[1]


def test_empty_grants_deny_path_but_keep_scientific_available(tmp_path):
    path = tmp_path / "ungranted.txt"
    path.write_text("private")
    client = ScriptedLLMClient(tool_turns([read(path), ask()]))
    agent = ScientificAgent(client)
    result = agent.invoke(request())
    assert result.status == "needs_user_input", result.report
    observed = receipt(agent.store.load(result.session.id), 0)
    assert observed["ok"] is False
    assert "private" not in json.dumps(observed)
    assert result.artifacts == []


def test_missing_output_root_is_a_read_error_not_agent_initialization_failure(tmp_path):
    missing = tmp_path / "missing"
    client = ScriptedLLMClient(tool_turns([read(missing / "partial.txt"), ask()]))
    agent = ScientificAgent(client)
    result = agent.invoke(request(grants=[grant(missing)]))
    assert result.status == "needs_user_input", result.report
    assert receipt(agent.store.load(result.session.id), 0)["ok"] is False
    assert not missing.exists()
