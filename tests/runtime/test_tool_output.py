"""Durable public receipts remain complete behind bounded native-tool displays."""

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from resagent2_capabilities import ReadFileTool
from resagent2_components import ReadOnlyWorkspaceBoundary
from resagent2_contracts import AgentRequest, TaskBudget, WorkspaceAccess, WorkspaceGrant
from resagent2_runtime import (
    AgentDefinition, AgentLoop, AgentState, AllowListPermissionPolicy, AskUserTool,
    CompletionDecision, ContextSection, FinishTool, InMemorySessionStore,
    JsonSessionStore, ScriptedLLMClient, ToolObservation,
)
from resagent2_runtime.models import NativeToolCall, RuntimeModel, ToolCallTurn
from resagent2_runtime.tool_calling import native_input_text, tool_messages, tool_receipt
from resagent2_runtime.tool_output import (
    MAX_TOOL_RECEIPT_CHARS, export_tool_receipt, model_tool_receipt, sync_tool_receipt_files,
)


def state(directory=None, turns=()):
    now = datetime.now(UTC)
    return AgentState(session_id="session_receipt", agent_name="receipt-test", owner="executor",
        run_id="run_receipt", task_id="task_receipt", attempt_number=1,
        created_at=now, updated_at=now, tool_result_dir=str(directory) if directory else None,
        tool_turns=list(turns))


def turn(receipt, *, call_id="call_receipt", name="emit_report"):
    return ToolCallTurn(tool_calls=[NativeToolCall(id=call_id, name=name, arguments="{}")],
                        tool_results={call_id: receipt})


def public_receipt(content):
    return tool_receipt(ToolObservation(ok=False, summary="The command failed after producing output",
        value={"exit_code": 7, "execution_status": "completed", "content": content},
        user_comment="Original approval: preserve the exact output",
        memory_updates={"private_runtime_value": "NOT_PUBLIC"}), sequence=19)


def read_all_through_model(tool, saved, path):
    """Follow only positions actually shown to the model, including shortened windows."""
    chunks, offset, pages, shortened = [], 0, 0, False
    while True:
        observation = tool.execute(saved, tool.input_model(path=path, start_char=offset))
        raw = tool_receipt(observation)
        shown_text = model_tool_receipt(raw)
        shown = json.loads(shown_text)
        assert len(shown_text) <= MAX_TOOL_RECEIPT_CHARS
        assert shown["ok"] is True
        value = shown["value"]
        assert value["path"] == path and value["start_char"] == offset
        assert value["content"]
        chunks.append(value["content"])
        pages += 1
        shortened |= len(value["content"]) < len(observation.value["content"])
        following = value["next_start_char"]
        if following is None:
            break
        assert following == offset + len(value["content"])
        assert following > offset
        offset = following
        assert pages < 100
    return "".join(chunks), pages, shortened


def readonly_tool(directory):
    grant = WorkspaceGrant(root=str(directory), source="local",
                           access=WorkspaceAccess(read_paths=["."], write_paths=[]))
    return ReadFileTool(ReadOnlyWorkspaceBoundary([grant]))


def test_export_contains_the_exact_public_json_and_utf8_hash(tmp_path):
    receipt = public_receipt("原始中文 / emoji 😃 / \\ / \" / \n / \u0001" * 7000)
    reference = export_tool_receipt(str(tmp_path), receipt, turn_index=3, call_id="call_exact")
    exported = Path(reference["path"]).read_bytes()
    assert reference["available"] is True and reference["format"] == "json"
    assert exported == receipt.encode("utf-8")
    assert json.loads(exported) == json.loads(receipt)
    assert reference["sha256"] == hashlib.sha256(exported).hexdigest()
    assert reference["total_chars"] == len(receipt)
    assert reference["total_bytes"] == len(exported)
    assert b"private_runtime_value" not in exported and b"NOT_PUBLIC" not in exported
    assert json.loads(exported)["user_comment"].startswith("Original approval")
    assert json.loads(exported)["observed_at"] == 19


def test_export_identity_is_stable_and_distinguishes_turns_and_content(tmp_path):
    receipt = public_receipt("result " * 30_000)
    unsafe_id = "../../outside/同名 call"
    first = export_tool_receipt(str(tmp_path), receipt, turn_index=4, call_id=unsafe_id)
    path = Path(first["path"])
    path_before = path.stat().st_mtime_ns
    repeated = export_tool_receipt(str(tmp_path), receipt, turn_index=4, call_id=unsafe_id)
    next_turn = export_tool_receipt(str(tmp_path), receipt, turn_index=5, call_id=unsafe_id)
    revised = export_tool_receipt(str(tmp_path), receipt + " ", turn_index=4, call_id=unsafe_id)
    assert first == repeated and path.stat().st_mtime_ns == path_before
    assert len({first["path"], next_turn["path"], revised["path"]}) == 3
    assert first["sha256"] == next_turn["sha256"] != revised["sha256"]
    assert all(Path(ref["path"]).parent == tmp_path for ref in (first, next_turn, revised))
    assert path.read_text() == receipt


def test_receipt_over_ten_mib_reassembles_from_bounded_parts_readable_to_the_end(tmp_path):
    # Four-byte characters exercise the byte limit independently of character count.
    receipt = public_receipt("😃" * 2_800_000 + "EXACT_TRAILING_MARKER")
    assert len(receipt.encode("utf-8")) > 10 * 1024 * 1024
    reference = export_tool_receipt(str(tmp_path), receipt, turn_index=0, call_id="call_large")
    assert reference["format"] == "manifest"
    manifest = json.loads(Path(reference["path"]).read_bytes())
    assert manifest["format"] == "concatenated-json-text"
    assert len(manifest["parts"]) == reference["part_count"] > 1
    expected_bytes = receipt.encode("utf-8")
    assert manifest["sha256"] == reference["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert manifest["total_bytes"] == reference["total_bytes"] == len(expected_bytes)
    assert manifest["total_chars"] == reference["total_chars"] == len(receipt)
    tool, saved, contents = readonly_tool(tmp_path), state(tmp_path), []
    for part in manifest["parts"]:
        path = Path(part["path"])
        assert path.parent == tmp_path
        assert path.stat().st_size <= 8_000_000
        assert len(path.read_text(encoding="utf-8")) == part["chars"]
        content, pages, _ = read_all_through_model(tool, saved, str(path))
        assert pages > 1
        contents.append(content)
    reconstructed = "".join(contents)
    assert reconstructed.encode("utf-8") == expected_bytes
    assert json.loads(reconstructed)["value"]["content"].endswith("EXACT_TRAILING_MARKER")
    assert not list(tmp_path.glob(".receipt-*"))


def test_small_receipts_do_not_create_export_roots_or_change_projection(tmp_path):
    directory = tmp_path / "not-created"
    receipt = public_receipt("small exact content")
    saved = state(directory, [turn(receipt)])
    assert sync_tool_receipt_files(saved, {}) is False
    assert not directory.exists()
    assert not saved.tool_turns[0].tool_result_files
    assert model_tool_receipt(receipt) == receipt


def test_projection_uses_saved_reference_without_any_file_io_or_mutation(tmp_path, monkeypatch):
    from resagent2_runtime import tool_output
    receipt = public_receipt("complete output " * 30_000)
    reference = export_tool_receipt(str(tmp_path), receipt, turn_index=0, call_id="call_receipt")
    native_turn = turn(receipt)
    native_turn.tool_result_files["call_receipt"] = reference
    before = native_turn.model_dump_json()

    def forbidden(*args, **kwargs):
        pytest.fail("Model projection must not read or write files")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbidden)
        for name in ("open", "read_bytes", "read_text", "write_bytes", "write_text", "stat", "mkdir"):
            patch.setattr(Path, name, forbidden)
        patch.setattr(tool_output, "export_tool_receipt", forbidden)
        shown_text = tool_messages([native_turn])[-1]["content"]
        wire = json.loads(native_input_text("task", [], [native_turn]))
    shown = json.loads(shown_text)
    assert len(shown_text) <= MAX_TOOL_RECEIPT_CHARS
    assert shown["display"]["full_result"] == reference
    assert shown["ok"] is False and shown["value"]["exit_code"] == 7
    assert wire["messages"][-1]["content"] == shown_text
    assert native_turn.model_dump_json() == before


class EmptyInput(RuntimeModel):
    pass


class EmitReport:
    name = "emit_report"
    input_model = EmptyInput

    def __init__(self, content):
        self.content, self.effects = content, []

    def execute(self, saved, args):
        self.effects.append("operation actually executed")
        return ToolObservation(summary="Produced a complete command result",
            value={"exit_code": 0, "output": self.content},
            memory_updates={"execution_preserved": True, "private_runtime_value": "NOT_PUBLIC"})


class Complete:
    def evaluate(self, saved, candidate):
        return CompletionDecision(complete=candidate is not None, report="Done")


def call(name, identifier, arguments=None):
    return ToolCallTurn(tool_calls=[NativeToolCall(id=identifier, name=name,
        arguments=json.dumps(arguments or {}, ensure_ascii=False))])


def request(parent=None):
    return AgentRequest(run_id="run_receipt", work_request_id="work_receipt",
        task_id="task_receipt", attempt_number=1, agent="executor", instruction="Preserve actual output",
        parent_session_id=parent, permissions={}, budget=TaskBudget(max_llm_calls=5, timeout_seconds=30))


def definition(tool, directory, client):
    return AgentDefinition(name="receipt-test", owner="executor", system_prompt="Use the existing tools",
        tools=(tool, AskUserTool(), FinishTool()), llm_client=client,
        permission_policy=AllowListPermissionPolicy({"emit_report", "ask_user", "finish"}),
        context_builder=lambda req, *_: [ContextSection(name="task", content=req.instruction)],
        completion_check=Complete(), tool_result_dir=str(directory))


def pause_after_output(tmp_path, *, store=None, content=None):
    store = store or JsonSessionStore(tmp_path / "sessions")
    tool = EmitReport(content or "The whole observed output " * 12_000)
    client = ScriptedLLMClient([
        call("emit_report", "call_emit"),
        call("ask_user", "call_question", {"text": "Which next step?", "requested_fields": ["step"]}),
    ])
    spec = definition(tool, tmp_path / "receipts", client)
    loop = AgentLoop(store=store)
    result = loop.run(spec, request(), session_id="session_receipt")
    assert result.status == "needs_user_input"
    return loop, spec, tool, store, result


def test_export_failure_happens_after_durable_execution_facts_and_stays_model_visible(tmp_path, monkeypatch):
    from resagent2_runtime import tool_output
    store = InMemorySessionStore()
    observed_receipts = []

    def unavailable(directory, receipt, *, turn_index, call_id):
        durable = store.load("session_receipt")
        assert durable.memory["execution_preserved"] is True
        assert durable.tool_turns[turn_index].tool_results[call_id] == receipt
        assert json.loads(receipt)["ok"] is True
        observed_receipts.append(receipt)
        raise OSError("sensitive internal details must not enter the model display")

    monkeypatch.setattr(tool_output, "export_tool_receipt", unavailable)
    _, spec, tool, store, result = pause_after_output(tmp_path, store=store)
    saved = store.load(result.session.id)
    assert len(tool.effects) == len(observed_receipts) == 1
    receipt = saved.tool_turns[0].tool_results["call_emit"]
    assert receipt == observed_receipts[0]
    shown = json.loads(tool_messages(saved.tool_turns)[1]["content"])
    reference = shown["display"]["full_result"]
    assert reference["available"] is False and reference["error"] == "OSError"
    assert "sensitive" not in json.dumps(reference)
    assert shown["ok"] is True and shown["value"]["exit_code"] == 0
    assert not saved.authorized_artifact_ids and not result.artifacts
    assert {item.name for item in spec.tools} == {"emit_report", "ask_user", "finish"}
    assert "NOT_PUBLIC" not in receipt


@pytest.mark.parametrize("multipart", [False, True], ids=["single-file", "manifest-and-part"])
def test_disk_resume_rebuilds_missing_export_files_without_model_or_tool_replay(tmp_path, multipart):
    content = "界" * 750_000 if multipart else "The original complete output " * 12_000
    # JSON must exceed 2M characters to exercise the part manifest.
    if multipart:
        content *= 3
    _, spec, tool, store, paused = pause_after_output(tmp_path, content=content)
    saved = store.load(paused.session.id)
    original = saved.tool_turns[0].tool_results["call_emit"]
    reference = saved.tool_turns[0].tool_result_files["call_emit"].copy()
    if multipart:
        assert reference["format"] == "manifest"
        manifest_path = Path(reference["path"])
        manifest = json.loads(manifest_path.read_text())
        Path(manifest["parts"][-1]["path"]).unlink()
        manifest_path.unlink()
    else:
        Path(reference["path"]).unlink()
    fresh_client = ScriptedLLMClient([])
    spec = replace(spec, llm_client=fresh_client)
    reopened = JsonSessionStore(store.root)
    result = AgentLoop(store=reopened).run(spec, request(paused.session.id), session_id="unused")
    recovered = reopened.load(paused.session.id)
    assert result.status == "needs_user_input" and result.llm_calls == 0
    assert not fresh_client.contexts and len(tool.effects) == 1
    assert recovered.tool_turns[0].tool_results["call_emit"] == original
    assert recovered.tool_turns[0].tool_result_files["call_emit"] == reference
    if multipart:
        manifest = json.loads(Path(reference["path"]).read_text())
        restored = "".join(Path(part["path"]).read_text() for part in manifest["parts"])
    else:
        restored = Path(reference["path"]).read_text()
    assert restored == original
    assert recovered.control_history == saved.control_history
    assert not recovered.authorized_artifact_ids and not result.artifacts


def test_resume_cannot_rebind_export_directory_or_change_the_saved_session(tmp_path):
    _, spec, tool, store, paused = pause_after_output(tmp_path)
    original_path = store.root / f"{paused.session.id}.json"
    original_bytes = original_path.read_bytes()
    rebound = replace(spec, tool_result_dir=str(tmp_path / "other-receipts"),
                      llm_client=ScriptedLLMClient([]))
    result = AgentLoop(store=JsonSessionStore(store.root)).run(
        rebound, request(paused.session.id), session_id="unused")
    assert result.status == "failed" and result.error.code == "contract_error"
    assert original_path.read_bytes() == original_bytes
    assert not (tmp_path / "other-receipts").exists()
    assert len(tool.effects) == 1 and not rebound.llm_client.contexts


def test_reading_exported_high_escape_windows_reaches_tail_without_secondary_exports(tmp_path):
    receipt = public_receipt("\\\"" * 180_000 + "EXACT_END")
    saved = state(tmp_path, [turn(receipt)])
    memo = {}
    assert sync_tool_receipt_files(saved, memo)
    reference = saved.tool_turns[0].tool_result_files["call_receipt"]
    original_files = set(tmp_path.iterdir())
    tool = readonly_tool(tmp_path)
    chunks, offset, page_count, shortened = [], 0, 0, False
    while True:
        observation = tool.execute(saved, tool.input_model(path=reference["path"], start_char=offset))
        raw = tool_receipt(observation)
        read_turn = turn(raw, call_id=f"read_{page_count}", name="read_file")
        saved.tool_turns.append(read_turn)
        sync_tool_receipt_files(saved, memo)
        assert not read_turn.tool_result_files
        shown_text = tool_messages([read_turn])[-1]["content"]
        assert len(shown_text) <= MAX_TOOL_RECEIPT_CHARS
        shown = json.loads(shown_text)["value"]
        assert shown["path"] == reference["path"]
        assert shown["content"]
        shortened |= len(shown["content"]) < len(observation.value["content"])
        chunks.append(shown["content"])
        page_count += 1
        following = shown["next_start_char"]
        if following is None:
            break
        assert following == offset + len(shown["content"]) > offset
        offset = following
        assert page_count < 50
    assert shortened and page_count > 1
    assert "".join(chunks) == receipt
    assert set(tmp_path.iterdir()) == original_files
    assert not saved.authorized_artifact_ids



def test_export_failure_can_be_recovered_without_changing_the_original_receipt(tmp_path, monkeypatch):
    from resagent2_runtime import tool_output

    def unavailable(*args, **kwargs):
        raise OSError("Temporary export storage failure")

    with monkeypatch.context() as patch:
        patch.setattr(tool_output, "export_tool_receipt", unavailable)
        _, spec, tool, store, paused = pause_after_output(tmp_path)
    saved = store.load(paused.session.id)
    original = saved.tool_turns[0].tool_results["call_emit"]
    assert saved.tool_turns[0].tool_result_files["call_emit"]["available"] is False
    fresh_client = ScriptedLLMClient([])
    result = AgentLoop(store=JsonSessionStore(store.root)).run(
        replace(spec, llm_client=fresh_client), request(paused.session.id), session_id="unused")
    recovered = store.load(paused.session.id)
    reference = recovered.tool_turns[0].tool_result_files["call_emit"]
    assert result.status == "needs_user_input" and result.llm_calls == 0
    assert len(tool.effects) == 1 and not fresh_client.contexts
    assert reference["available"] is True
    assert Path(reference["path"]).read_bytes() == original.encode("utf-8")
    assert reference["sha256"] == hashlib.sha256(original.encode("utf-8")).hexdigest()
    assert recovered.tool_turns[0].tool_results["call_emit"] == original


def test_optional_reference_checkpoint_failure_does_not_turn_a_completed_tool_into_failure(tmp_path):
    class FailFirstReferenceSave(InMemorySessionStore):
        failed = False

        def save(self, saved):
            if not self.failed and any(t.tool_result_files for t in saved.tool_turns):
                self.failed = True
                # The raw receipt was saved by the preceding authoritative checkpoint.
                prior = self.load(saved.session_id)
                assert prior.memory["execution_preserved"] is True
                assert prior.tool_turns[0].tool_results == saved.tool_turns[0].tool_results
                assert not prior.tool_turns[0].tool_result_files
                raise OSError("Optional reference checkpoint could not be saved")
            super().save(saved)

    store = FailFirstReferenceSave()
    _, _, tool, _, paused = pause_after_output(tmp_path, store=store)
    assert store.failed and paused.status == "needs_user_input"
    saved = store.load(paused.session.id)
    receipt = saved.tool_turns[0].tool_results["call_emit"]
    assert json.loads(receipt)["ok"] is True
    assert len(tool.effects) == 1 and saved.memory["execution_preserved"] is True
    reference = saved.tool_turns[0].tool_result_files["call_emit"]
    assert reference["available"] is True
    assert Path(reference["path"]).read_text() == receipt
    assert not saved.authorized_artifact_ids and not paused.artifacts


def test_reviewed_operation_exports_its_original_public_review_receipt(tmp_path):
    from resagent2_runtime import PermissionDecision

    class Policy:
        def check(self, action, saved, req):
            return PermissionDecision(outcome="ask" if action.tool == "emit_report" else "allow",
                                      context={"exact_operation": "fixed"})

    class Reviewer:
        def review(self, operation, req):
            return {"decision": "allow", "reason": "The exact operation is authorized"}

    tool = EmitReport("Full approved output " * 12_000)
    client = ScriptedLLMClient([call("emit_report", "call_emit"),
                               call("finish", "call_finish", {"report": "Done"})])
    spec = replace(definition(tool, tmp_path / "receipts", client),
                   permission_policy=Policy(), operation_reviewer=Reviewer())
    req = request().model_copy(update={"permissions": request().permissions.model_copy(
        update={"approval_mode": "auto_review"})})
    store = JsonSessionStore(tmp_path / "sessions")
    result = AgentLoop(store=store).run(spec, req, session_id="session_receipt")
    saved = store.load(result.session.id)
    receipt = saved.tool_turns[0].tool_results["call_emit"]
    reference = saved.tool_turns[0].tool_result_files["call_emit"]
    exported = Path(reference["path"]).read_bytes()
    assert result.status == "completed" and len(tool.effects) == 1
    assert exported == receipt.encode("utf-8")
    assert reference["sha256"] == hashlib.sha256(exported).hexdigest()
    review = json.loads(exported)["permission_review"]
    assert review["source"] == "auto_review" and review["status"] == "allow"
    events = [event for event in saved.events if event.type == "permission_review"]
    assert review["audit_sequence"] == events[-1].sequence
    assert review["action_id"] == events[-1].data["operation"]["action_id"]
    assert "NOT_PUBLIC" not in exported.decode("utf-8")
    assert not result.artifacts and not saved.authorized_artifact_ids


def test_conflicting_export_bytes_are_not_overwritten_or_presented_as_original(tmp_path):
    receipt = public_receipt("The complete original " * 12_000)
    saved = state(tmp_path, [turn(receipt)])
    assert sync_tool_receipt_files(saved, {})
    path = Path(saved.tool_turns[0].tool_result_files["call_receipt"]["path"])
    path.write_text("conflicting bytes")
    # A fresh invocation must validate the existing snapshot against durable facts.
    assert sync_tool_receipt_files(saved, {})
    reference = saved.tool_turns[0].tool_result_files["call_receipt"]
    assert reference["available"] is False and reference["error"] == "OSError"
    assert path.read_text() == "conflicting bytes"
    assert saved.tool_turns[0].tool_results["call_receipt"] == receipt
    shown = json.loads(tool_messages(saved.tool_turns)[-1]["content"])
    assert shown["display"]["full_result"]["available"] is False
    assert shown["ok"] is False and shown["value"]["exit_code"] == 7
