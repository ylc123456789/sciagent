"""Public CLI, real stores, two native work rounds and explicit Task report reads."""

import hashlib
import io
import json
import re
import subprocess
from collections import Counter

import httpx

from resagent2_cli.main import EXIT_COMPLETED, EXIT_PAUSED, cli
from resagent2_cli.shell import Shell
from resagent2_cli.composition import build_application
from resagent2_contracts import ArtifactIndexEntry
from resagent2_coding import NativeCodingAgent
from resagent2_experiment import NativeExperimentAgent
from resagent2_orchestrator import JsonRunStore
from resagent2_scientific import ScientificAgent


def test_cli_rebuilds_and_finishes_two_native_work_rounds(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    obsolete = repo / "obsolete"
    obsolete.mkdir()
    (obsolete / "old.py").write_text("OLD = True\n")
    metrics = repo / "metrics.json"
    metrics.write_text('{"baseline": 0.45, "candidate": 0.52}')
    original_hash = hashlib.sha256(metrics.read_bytes()).hexdigest()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    data, run_id = tmp_path / "data", "run_public_chain"
    store = JsonRunStore(data / "state")
    counts, requests, scientific_requests, scientific_agents = Counter(), [], [], []
    directories = []
    monkeypatch.setenv("RESAGENT2_MODEL", "chain-test")
    monkeypatch.setenv("RESAGENT2_API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("RESAGENT2_API_KEY_ENV", "CHAIN_TEST_KEY")
    monkeypatch.setenv("CHAIN_TEST_KEY", "test-only")
    monkeypatch.setenv("RESAGENT2_LLM_TRACE_LEVEL", "off")
    monkeypatch.setenv("RESAGENT2_RESOURCE_ROOT", str(tmp_path / "resources"))
    invoke = ScientificAgent.invoke
    def capture(self, request):
        scientific_requests.append(request)
        scientific_agents.append(self)
        return invoke(self, request)
    monkeypatch.setattr(ScientificAgent, "invoke", capture)

    def walk(node):
        yield node
        for child in node.get("children", []):
            yield from walk(child)
    def work_groups(directory):
        return next(node["children"] for node in walk(directory) if node["name"] == "work_requests" and node["type"] == "group")
    def root_session(current):
        return next(task for task in current.workflow.tasks if task.id == "task_root").attempts[-1].session
    def worker_task(current, kind):
        return next(task for task in current.workflow.tasks if task.execution_ref == f"{kind}@1")
    def run():
        return store.load(run_id)
    def evidence_id():
        return next(ref.id for ref in run().artifacts.values() if ref.kind == "metrics")
    def report_id(kind):
        task = next(task for task in run().workflow.tasks if task.execution_ref == f"{kind}@1")
        return task.attempts[-1].report_artifact_id
    def read_ids():
        state = scientific_agents[-1].store.load(root_session(run()).id)
        return {event.data["value"]["artifact_id"] for event in state.events
                if event.type == "observation" and event.tool == "read_artifact" and event.data.get("ok")}
    def work(objective, preset):
        return "request_work", {"assessment": {"statement": objective},
                                "work_request": {'objective': objective, 'tasks': [{"selection": {"kind": "preset", "preset_ref": preset, "reason": "Suitable configured executor"}, 'key': 'work', 'instruction': objective}]}}
    def paired_receipt(body, call):
        return json.loads(next(message["content"] for message in body["messages"]
                               if message.get("tool_call_id") == call))

    def reply(request, **kwargs):
        body = json.loads(request.content)
        requests.append(body)
        names = {tool["function"]["name"] for tool in body.get("tools", [])}
        prompt = "\n".join(message.get("content") or "" for message in body["messages"])
        assert "route_tasks" not in names
        role = ("coding" if "delete_path" in names
                else "experiment" if "run_shell" in names else "scientific")
        counts[role] += 1
        index = counts[role]
        if role == "scientific":
            current = body["messages"][-1]["content"]
            directory = json.loads(current.split("\n", 1)[-1]) if current.startswith("## material_index\n") else json.loads(current)
            directories.append(directory)
            req = scientific_requests[-1]
            by_id = {ref.id: ref for ref in req.input_artifacts}
            entries = [node["metadata"] for node in walk(directory) if node["type"] == "artifact"]
            ids = [entry["artifact_id"] for entry in entries]
            assert len(ids) == len(set(ids)) == directory["artifact_count"]
            assert set(ids) == set(by_id)
            for entry in entries:
                ref = by_id[entry["artifact_id"]]
                assert entry == ArtifactIndexEntry.from_ref(ref).model_dump(mode="json", exclude_none=True)
                assert ref.uri not in current and ref.sha256 not in current
            assert all(ref.kind not in {"research_index", "work_record", "work_feedback", "question", "answer"}
                       for ref in by_id.values())
            if req.child_questions:
                child = req.child_questions[0]
                if not req.user_answers:
                    tool, args = "ask_user", {"assessment": {"statement": "The child needs a user choice"},
                        "text": child.text, "requested_fields": child.requested_fields}
                else:
                    tool, args = "update_work", {"work_request_id": req.task_context["active_work_request_id"],
                        "answers": [{"task_id": child.task_id, "question_id": child.id,
                                     "values": req.user_answers[-1].values}]}
            elif index == 1:
                tool, args = work("Remove the obsolete source directory with approval", "coding@1")
            elif index == 2:
                group, = work_groups(directory)
                assert len(group["children"]) == 1 and group["children"][0]["status"] == "completed"
                assert "Removed obsolete source" not in current
                assert report_id("coding") not in read_ids()
                tool, args = "read_artifact", {"artifact_id": report_id("coding")}
            elif index == 3:
                assert "Removed obsolete source" in paired_receipt(body, "call_scientific_2")["value"]["content"]
                tool, args = work("Analyze existing baseline and candidate metrics; ask which metric", "experiment@1")
            elif index == 6:
                assert len(work_groups(directory)) == 2
                assert "Accuracy difference is 0.07" not in current
                assert evidence_id() not in read_ids()
                assert len(req.user_answers) == 1 and req.user_answers[0].source == "user"
                assert req.user_answers[0].task_id == "task_root"
                tool, args = "read_artifact", {"artifact_id": report_id("experiment")}
            elif index == 7:
                assert "Accuracy difference is 0.07" in paired_receipt(body, "call_scientific_6")["value"]["content"]
                tool, args = "read_artifact", {"artifact_id": evidence_id()}
            elif index in (8, 9):
                if index == 9:
                    rejected = paired_receipt(body, "call_scientific_8")
                    assert not rejected["ok"] and "data" in json.dumps(rejected)
                    assert run().status != "failed"
                tool, args = "finish", {"report": "The recorded candidate exceeds baseline by 0.07", "artifacts": [{
                    "kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json",
                    "summary": "Analysis of existing measurements", "content": {
                        "verdict": "inconclusive", "statement": "Recorded accuracy difference is 0.07",
                        "evidence_artifact_ids": [evidence_id()], "limitations": ["Existing results only; no new experiment executed"],
                    },
                }]}
                if index == 8:
                    args["artifacts"].insert(0, {"kind": "data", "path": evidence_id(),
                                               "media_type": "application/json", "summary": "Existing input evidence"})
            else:
                raise AssertionError(body)
        elif role == "coding":
            if index == 1:
                tool, args = "delete_path", {"path": "obsolete", "recursive": True}
            elif index == 2:
                assert not obsolete.exists()
                tool, args = "finish", {"report": "Removed obsolete source"}
            else:
                raise AssertionError(body)
            task_payload = next(json.loads(message["content"].split("\n", 1)[1]) for message in body["messages"]
                                if message.get("role") == "user" and message.get("content", "").startswith("## task\n"))
            assert [event["record"] for event in task_payload["business_dialogue"] if event["type"] == "answer"] == []
        else:
            if index == 1:
                tool, args = "ask_user", {"text": "Which metric should be primary?", "requested_fields": ["metric"]}
            elif index == 2:
                assert "accuracy" in prompt and "Which metric should be primary?" in prompt
                tool, args = "read_file", {"path": "metrics.json"}
            elif index == 3:
                tool, args = "finish", {"report": "Accuracy difference is 0.07", "artifacts": [{
                    "kind": "metrics", "path": "metrics.json", "media_type": "application/json",
                    "summary": "Existing baseline and candidate values",
                }]}
            else:
                raise AssertionError(body)
        message = {"content": None, "tool_calls": [{"id": f"call_{role}_{index}", "type": "function",
                                                    "function": {"name": tool, "arguments": json.dumps(args)}}]}
        reason = "tool_calls"
        return httpx.Response(200, request=request, json={"choices": [{"finish_reason": reason, "message": message}]})

    monkeypatch.setattr("resagent2_runtime.llm.send_request", reply)
    assert cli(["run", "--run-id", run_id, "--goal", "Clean obsolete source and analyze existing accuracy",
                "--workspace", str(repo), "--data-root", str(data), "--no-execute-commands",
                "--max-llm-calls", "30", "--max-tasks", "3"]) == EXIT_PAUSED
    first = run()
    assert obsolete.is_dir() and first.pending_question.action.tool == "delete_path"
    coding_session = worker_task(first, "coding").attempts[0].session.id
    scientific_session = root_session(first).id
    shell = Shell(data_root=data, application_builder=build_application, store=store, stream=io.StringIO())
    shell.current_run_id = run_id
    shell._dispatch("/answer yes")
    assert shell.runner.done
    second = run()
    assert not obsolete.exists() and second.status == "paused"
    assert second.pending_question.requested_fields == ["metric"] and second.workflow.revision == 3
    assert worker_task(second, "coding").attempts[0].session.id == coding_session
    experiment_session = worker_task(second, "experiment").attempts[0].session.id
    used_before_answer = second.llm_calls_used
    first_outcome = second.work_requests[1].outcome
    assert first_outcome is not None and "interpreter" not in counts
    assert cli(["answer", run_id, "--field", "metric=accuracy", "--data-root", str(data)]) == EXIT_COMPLETED
    final = run()
    assert final.status == "completed", final.terminal_error
    assert all(re.fullmatch(r"artifact_[0-9a-f]{32}", ref.id) for ref in final.artifacts.values())
    assert all(ref.id == key for key, ref in final.artifacts.items()) and final.pending_question is None
    assert root_session(final).id == scientific_session
    assert worker_task(final, "experiment").attempts[0].session.id == experiment_session
    assert all(len(task.attempts) == 1 and task.status == "completed" for task in final.workflow.tasks)
    assert all(work.status == "consumed" for work in final.work_requests)
    assert final.work_requests[1].outcome == first_outcome
    assert final.llm_calls_used == len(requests) == sum(counts.values()) > used_before_answer
    assert counts == {"scientific": 9, "coding": 2, "experiment": 3}
    assert len(scientific_requests) == 5 and final.usage.outcomes == {"succeeded": len(requests), "failed": 0, "unknown": 0}
    assert len(final.answers) == 3 and len([answer for answer in final.answers if answer.action is not None]) == 1
    assert {answer.question_text for answer in final.answers} == {first.pending_question.text, second.pending_question.text}
    assert final.final_opinion.evidence_artifact_ids == [evidence_id()]
    assert {evidence_id(), report_id("coding"), report_id("experiment")} <= read_ids()
    assert final.artifacts[final.final_report_artifact_id].kind == "final_report"
    assert hashlib.sha256(metrics.read_bytes()).hexdigest() == original_hash
    assert not any(ref.kind in {"research_index", "work_record", "work_feedback", "question", "answer", "execution_record"}
                   for ref in final.artifacts.values())
    final_directory = directories[-1]
    assert len(work_groups(final_directory)) == 2
    for work_group, record in zip(work_groups(final_directory), final.work_requests[1:], strict=True):
        assert work_group["metadata"]["work_request_id"] == record.id and work_group["metadata"]["objective"] == record.request.objective
        task_group, = work_group["children"]
        actual = record.outcome.tasks[0]
        assert task_group["metadata"]["task_id"] == actual.task_id
        ids = {entry["location"]["artifact_id"] for entry in task_group["children"]}
        assert actual.attempts[0].report_artifact_id in ids
        assert set(actual.attempts[0].artifact_ids) <= ids
        report_ref = final.artifacts[actual.attempts[0].report_artifact_id]
        assert report_ref.kind == "module_report" and report_ref.task_id == actual.task_id
        from resagent2_components import RegisteredArtifactReader
        text = RegisteredArtifactReader([report_ref], run_id=run_id).read_text(report_ref.id)["content"]
        expected = "Removed obsolete source" if record == final.work_requests[1] else "Accuracy difference is 0.07"
        assert text == expected
