"""Deletion must be reachable through Coding's actual native invocation path."""

from datetime import UTC, datetime
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from resagent2_coding import NativeCodingAgent
from resagent2_contracts import (
    AgentPermissions, AgentRequest, ArtifactCandidate, ModuleStatus, QuestionDraft,
    RecordedAnswer, TaskBudget, WorkspaceAccess, WorkspaceGrant,
)
from resagent2_components import RegisteredArtifactReader, ResourceLayout
from resagent2_orchestrator.artifacts import ArtifactRegistry
from resagent2_runtime import JsonSessionStore, OpenAICompatibleClient


def setup(tmp_path, monkeypatch, actions):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "keep.txt").write_text("keep\n")
    replies = iter(actions)
    sent_tools = []
    sequence = 0
    monkeypatch.setenv("DELETE_TEST_KEY", "test-only")

    def send(request, **kwargs):
        nonlocal sequence
        sequence += 1
        body = json.loads(request.content)
        sent_tools.append({tool["function"]["name"] for tool in body["tools"]})
        reply = next(replies)
        tool, arguments = reply(body) if callable(reply) else reply
        return httpx.Response(200, request=request, json={"choices": [{
            "finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
                "id": f"call_{sequence}", "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments)},
            }]},
        }]})

    monkeypatch.setattr("resagent2_runtime.llm.send_request", send)
    layout = ResourceLayout.from_env(data_root=tmp_path / "data")

    def agent():
        return NativeCodingAgent(
            OpenAICompatibleClient(
                model="delete-test", api_base="https://example.invalid/v1",
                api_key_env="DELETE_TEST_KEY",
            ),
            store=JsonSessionStore(tmp_path / "sessions"), resource_layout=layout,
        )

    request = AgentRequest(work_request_id="work_test",
        agent="executor", run_id="run_delete", task_id="task_delete", attempt_number=1,
        instruction="Delete the requested obsolete path; preserve keep.txt.",
        budget=TaskBudget(max_llm_calls=6, timeout_seconds=30),
        permissions=AgentPermissions(),
        workspace=WorkspaceGrant(
            root=str(repo), source="local",
            access=WorkspaceAccess(read_paths=["."], write_paths=["obsolete"]),
        ),
        output_dir=str(tmp_path / "out"),
    )
    return repo, request, agent, sent_tools


FINISH = ("finish", {"report": "Removed obsolete files; no commands executed."})


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_native_coding_deletes_file_or_empty_directory_without_asking(
    tmp_path, monkeypatch, directory, absolute,
):
    repo, request, agent, sent = setup(tmp_path, monkeypatch, [
        ("delete_path", {"path": str(tmp_path / "repo/obsolete") if absolute else "obsolete"}), FINISH,
    ])
    target = repo / "obsolete"
    if directory:
        target.mkdir()
    else:
        target.write_text("obsolete\n")

    result = agent().invoke(request)

    assert result.status == ModuleStatus.COMPLETED
    assert not target.exists()
    assert (repo / "keep.txt").read_text() == "keep\n"
    assert result.control is None
    assert sent and {"delete_path", "finish"} <= sent[0]
    assert all(names == sent[0] for names in sent)
    if not directory:
        patch = next(item for item in result.artifacts if item.kind == "code_patch")
        assert "deleted file mode" in (Path(request.output_dir) / patch.path).read_text()
        assert "-obsolete" in (Path(request.output_dir) / patch.path).read_text()


@pytest.mark.parametrize("absolute", [False, True])
def test_native_coding_recursive_delete_resumes_from_approved_snapshot(tmp_path, monkeypatch, absolute):
    target_address = str(tmp_path / "repo/obsolete") if absolute else "obsolete"
    deletion = ("delete_path", {"path": target_address, "recursive": True})
    def after_execution(body):
        assert not target.exists()
        task_message = next(message["content"] for message in body["messages"]
                            if message["role"] == "user" and message["content"].startswith("## task"))
        task = json.loads(task_message.split("## task\n",1)[1])
        assert [event["record"] for event in task["business_dialogue"] if event["type"] == "answer"] == []
        receipts = [json.loads(message["content"]) for message in body["messages"] if message["role"] == "tool"]
        assert len(receipts) == 1 and receipts[0]["ok"]
        assert "pending_operation" not in json.dumps(body)
        return FINISH
    repo, request, agent, _ = setup(tmp_path, monkeypatch, [deletion, after_execution])
    target = repo / "obsolete"
    target.mkdir()
    for name in ("first.txt", "second.txt"):
        (target/name).write_text(name)
    first = agent().invoke(request)
    assert first.status == ModuleStatus.NEEDS_USER_INPUT
    assert first.artifacts == []
    question = first.control.question
    assert question.action.tool == "delete_path"
    assert question.action.arguments == {"path": target_address, "recursive": True}
    recorded = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve":"yes"}, answered_at=datetime.now(UTC),
        run_id=request.run_id, task_id=request.task_id, attempt_number=1,
        session_id=first.session.id, action=question.action,
    )
    second = agent().invoke(request.model_copy(update={
        "parent_session_id":first.session.id, "resume_answers":[recorded], "control_ack":first.control.origin,
    }))
    assert second.status == ModuleStatus.COMPLETED, second.report
    assert not target.exists() and (repo/"keep.txt").read_text() == "keep\n"
    state = JsonSessionStore(tmp_path/"sessions").load(second.session.id)
    assert state.pending_action is None
    assert len([event for event in state.events if event.tool == "delete_path" and event.type == "observation"]) == 1
