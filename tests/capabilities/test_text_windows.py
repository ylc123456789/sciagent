"""Text windows work across workspace and frozen evidence without weakening grants."""

from e2e.native_fixtures import tool_turns

import hashlib
import io
import zipfile
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from sciagent_components import (
    ArtifactReadError,
    RegisteredArtifactReader,
    WorkspaceBoundary,
    WorkspacePermissionError,
)
from sciagent_capabilities import (
    ReadArtifactTool,
    ReadFileTool,
    SearchTextTool,
)
from sciagent_contracts import AgentOwner, ArtifactRef, WorkspaceAccess, WorkspaceGrant
from sciagent_runtime import AgentState


def _state():
    now = datetime.now(UTC)
    return AgentState(
        session_id="session_reader", agent_name="reader", owner=AgentOwner.EXECUTOR,
        run_id="run_reader", task_id="task_reader", attempt_number=1,
        created_at=now, updated_at=now,
    )

def _boundary(root, **kwargs):
    return WorkspaceBoundary(WorkspaceGrant(
        root=str(root), source="local", access=WorkspaceAccess(
            read_paths=kwargs.get("read_paths", ["."]),
            denied_paths=kwargs.get("denied_paths", []),
        ),
    ))


def _artifact(path):
    return ArtifactRef(
        id="artifact_text", kind="text", producer=AgentOwner.EXECUTOR,
        run_id="run_reader", task_id="task_reader", attempt_number=1,
        uri=path.as_uri(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        media_type="text/plain", summary="Frozen evidence",
    )


@pytest.mark.parametrize("start,end,expected", [
    (None, None, "one\ntwo\nthree\nfour\n"),
    (2, 3, "two\nthree\n"),
    (3, None, "three\nfour\n"),
    (None, 2, "one\ntwo\n"),
    (8, 10, ""),
])
def test_workspace_and_artifact_share_line_range_semantics(tmp_path, start, end, expected):
    path = tmp_path / "text.txt"
    path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    ref = _artifact(path)
    file_tool = ReadFileTool(_boundary(tmp_path))
    artifact_tool = ReadArtifactTool(RegisteredArtifactReader([ref], run_id=ref.run_id))
    file_result = file_tool.execute(_state(), file_tool.input_model(
        path="text.txt", start_line=start, end_line=end,
    ))
    artifact_result = artifact_tool.execute(_state(), artifact_tool.input_model(
        artifact_id=ref.id, start_line=start, end_line=end,
    ))
    for result in (file_result, artifact_result):
        assert result.value["start_line"] == start
        assert result.value["end_line"] == end
        assert result.value["total_lines"] == 4
        assert result.value["selected_chars"] == len(expected)
        assert result.value["content"] == expected
        assert result.value["truncated"] is False
    assert artifact_result.memory_updates.get("read_artifact_ids", []) == ([ref.id] if expected else [])
    assert artifact_result.value["provenance"] == {
        "producer": "executor", "task_id": "task_reader", "attempt_number": 1,
    }


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("body,options,total_lines,selected_chars", [
    ("", {}, 0, 0),
    ("one\ntwo\n", {"start_line": 8, "end_line": 10}, 2, 0),
    ("one\ntwo\n", {"start_char": 8}, 2, 8),
    ("one\ntwo\n", {"start_char": 20}, 2, 8),
])
def test_text_tools_distinguish_empty_source_and_ranges_beyond_end(
    tmp_path, registered, body, options, total_lines, selected_chars,
):
    path = tmp_path / "text.txt"
    path.write_bytes(body.encode("utf-8"))
    if registered:
        ref = _artifact(path)
        tool = ReadArtifactTool(RegisteredArtifactReader([ref], run_id=ref.run_id))
        source = {"artifact_id": ref.id}
    else:
        tool = ReadFileTool(_boundary(tmp_path))
        source = {"path": path.name}
    result = tool.execute(_state(), tool.input_model(**source, **options)).value
    assert result["content"] == ""
    assert result["total_lines"] == total_lines
    assert result["selected_chars"] == selected_chars
    assert result["truncated"] is False
    assert result["next_start_char"] is None


def test_artifact_range_can_recover_text_after_default_character_limit(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("first " + "x" * 128100 + "\nsecond\nrequired evidence\n", encoding="utf-8")
    ref = _artifact(path)
    reader = RegisteredArtifactReader([ref], run_id=ref.run_id)
    default = reader.read_text(ref.id)
    assert len(default["content"]) == 128_000
    assert default["truncated"] is True
    assert default["start_line"] is default["end_line"] is None
    ranged = reader.read_text(ref.id, start_line=2, end_line=3)
    assert ranged["content"] == "second\nrequired evidence\n"
    assert ranged["truncated"] is False
    assert reader.read_text(ref.id, start_line=2, end_line=3, max_chars=5)["truncated"] is True


def test_artifact_range_verifies_bytes_outside_selected_lines(tmp_path):
    path = tmp_path / "frozen.txt"
    path.write_text("selected line\nuntouched line\nlast line\n", encoding="utf-8")
    ref = _artifact(path)
    path.write_text("selected line\nuntouched line\nTAMPERED\n", encoding="utf-8")
    reader = RegisteredArtifactReader([ref], run_id=ref.run_id)
    with pytest.raises(ArtifactReadError, match="sha256"):
        reader.read_text(ref.id, start_line=1, end_line=1)


@pytest.mark.parametrize("live_resolver", [False, True])
def test_artifact_range_rejects_cross_run_before_reading(tmp_path, monkeypatch, live_resolver):
    path = tmp_path / "frozen.txt"
    path.write_text("must not read\n", encoding="utf-8")
    ref = _artifact(path)
    reader = RegisteredArtifactReader(
        [] if live_resolver else [ref], run_id="run_other",
        resolve=(lambda _: ref) if live_resolver else None,
    )
    def forbidden_read(self):
        raise AssertionError("Unauthorized artifact bytes were read")
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(ArtifactReadError, match="unknown artifact"):
        reader.read_text(ref.id, start_line=1, end_line=1)


def test_artifact_inverted_range_is_a_controlled_error(tmp_path):
    path = tmp_path / "frozen.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    ref = _artifact(path)
    tool = ReadArtifactTool(RegisteredArtifactReader([ref], run_id=ref.run_id))
    with pytest.raises(ValueError, match="end_line must be"):
        tool.execute(_state(), tool.input_model(artifact_id=ref.id, start_line=2, end_line=1))


def test_artifact_tool_rejects_zero_line_in_schema(tmp_path):
    tool = ReadArtifactTool(RegisteredArtifactReader([], run_id="run_reader"))
    with pytest.raises(ValidationError):
        tool.input_model(artifact_id="artifact_text", start_line=0)


def test_search_accepts_a_single_file_or_directory(tmp_path):
    (tmp_path / "train.py").write_text("nothing\nTarget = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("target = 2\n", encoding="utf-8")
    tool = SearchTextTool(_boundary(tmp_path))
    one = tool.execute(_state(), tool.input_model(path="train.py", query="target"))
    assert one.value["matches"] == [{"path": str(tmp_path / "train.py"), "line": 2, "text": "Target = 1"}]
    assert one.memory_updates["read_paths"] == [str(tmp_path / "train.py")]
    both = tool.execute(_state(), tool.input_model(path=".", query="target"))
    assert {item["path"] for item in both.value["matches"]} == {str(tmp_path / "train.py"), str(tmp_path / "other.py")}


@pytest.mark.parametrize("query_path", ["denied/secret.py", "alias.py", "escape.py", "../outside.py", ".sciagent/state"])
def test_single_file_search_cannot_bypass_path_scope(tmp_path, query_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "denied").mkdir()
    denied = root / "denied" / "secret.py"
    denied.write_text("target = secret\n")
    outside = tmp_path / "outside.py"
    outside.write_text("target = outside\n")
    (root / "alias.py").symlink_to(denied)
    (root / "escape.py").symlink_to(outside)
    tool = SearchTextTool(_boundary(root, denied_paths=["denied"]))
    with pytest.raises(WorkspacePermissionError):
        tool.execute(_state(), tool.input_model(path=query_path, query="target"))


def test_single_file_search_respects_read_paths(tmp_path):
    (tmp_path / "allowed.py").write_text("target = 1\n")
    (tmp_path / "other.py").write_text("target = 2\n")
    tool = SearchTextTool(_boundary(tmp_path, read_paths=["allowed.py"]))
    assert tool.execute(_state(), tool.input_model(path="allowed.py", query="target")).ok
    with pytest.raises(WorkspacePermissionError):
        tool.execute(_state(), tool.input_model(path="other.py", query="target"))


def test_search_guidance_matches_bounded_input_schema(tmp_path):
    tool = SearchTextTool(_boundary(tmp_path))
    assert "file or directory" in tool.model_guidance
    assert "50" in tool.model_guidance
    with pytest.raises(ValidationError):
        tool.input_model(query="target", max_results=51)


def test_search_pipe_is_literal_and_guidance_does_not_promise_regex(tmp_path):
    from sciagent_runtime.tool_calling import native_tool_schemas

    (tmp_path / "names.txt").write_text("alpha\nbeta\nALPHA|BETA\n", encoding="utf-8")
    tool = SearchTextTool(_boundary(tmp_path))
    result = tool.execute(_state(), tool.input_model(path="names.txt", query="alpha|beta"))
    assert result.value["matches"] == [
        {"path": str(tmp_path / "names.txt"), "line": 3, "text": "ALPHA|BETA"},
    ]
    contracts = native_tool_schemas((tool,))[0]["function"]["description"]
    assert "literal substring match, not regex" in contracts
    assert "search alternatives separately" in contracts
    assert "no regular expressions" in tool.input_model.model_json_schema()[
        "properties"
    ]["query"]["description"]

def test_artifact_tool_pages_long_json_without_rewriting_source(tmp_path):
    from sciagent_components.materials import read_artifact_json

    path = tmp_path / "feedback.json"
    body = json.dumps({"report": "原报告\\n" * 40_000 + "UNIQUE_TAIL"}, ensure_ascii=False, indent=2)
    path.write_text(body, encoding="utf-8")
    ref = _artifact(path)
    reader = RegisteredArtifactReader([ref], run_id=ref.run_id)
    tool = ReadArtifactTool(reader)
    default = tool.execute(_state(), tool.input_model(artifact_id=ref.id)).value
    assert default["truncated"] is True
    assert "UNIQUE_TAIL" not in default["content"]
    pieces = []
    for start in range(0, len(body), 4_000):
        result = tool.execute(_state(), tool.input_model(
            artifact_id=ref.id, start_char=start, end_char=start + 4_000,
        ))
        assert result.value["truncated"] is False
        assert result.memory_updates.get("read_artifact_ids", []) == ([ref.id] if body else [])
        pieces.append(result.value["content"])
    assert "".join(pieces) == body
    assert read_artifact_json(reader, ref.id) == json.loads(body)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ref.sha256
    # Physical lines retain their meaning; offsets are relative to that selection.
    line = body.splitlines(keepends=True)[1]
    result = tool.execute(_state(), tool.input_model(
        artifact_id=ref.id, start_line=2, end_line=2, start_char=130_000, end_char=131_000,
    ))
    assert result.value["content"] == line[130_000:131_000]
    path.write_text("X" + body[1:], encoding="utf-8")
    with pytest.raises(ArtifactReadError, match="sha256"):
        tool.execute(_state(), tool.input_model(artifact_id=ref.id, start_char=130_000))


@pytest.mark.parametrize("start,end", [(5, 5), (5, 4)])
def test_artifact_tool_rejects_inverted_character_window(tmp_path, start, end):
    path = tmp_path / "text.txt"
    path.write_text("source", encoding="utf-8")
    ref = _artifact(path)
    tool = ReadArtifactTool(RegisteredArtifactReader([ref], run_id=ref.run_id))
    with pytest.raises(ValueError, match="character range"):
        tool.execute(_state(), tool.input_model(artifact_id=ref.id, start_char=start, end_char=end))


@pytest.mark.parametrize("content", [
    b"first line\n" + b"a" * 140_000 + b"\x00",
    b"first line\n" + b"a" * 140_000 + b"\xff",
])
@pytest.mark.parametrize("registered", [False, True])
def test_text_reads_reject_binary_even_outside_selected_window(tmp_path, content, registered):
    path = tmp_path / "data.json"
    path.write_bytes(content)
    state = _state()
    if registered:
        ref = _artifact(path)
        tool = ReadArtifactTool(RegisteredArtifactReader([ref], run_id=ref.run_id))
        args = tool.input_model(artifact_id=ref.id, start_line=1, end_line=1)
        error_type = ArtifactReadError
    else:
        tool = ReadFileTool(_boundary(tmp_path))
        args = tool.input_model(path=path.name, start_line=1, end_line=1)
        error_type = ValueError
    with pytest.raises(error_type, match="Cannot read as UTF-8 text") as error:
        tool.execute(state, args)
    assert len(str(error.value)) < 200
    assert "\ufffd" not in str(error.value)
    assert not state.memory.get("read_artifact_ids")
    assert not state.memory.get("read_paths")
    assert path.read_bytes() == content


def test_zip_is_verifiable_but_cannot_be_read_as_text(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metrics.json", '{"accuracy": 0.95}')
    path = tmp_path / "code_bundle.zip"
    path.write_bytes(buffer.getvalue())
    ref = _artifact(path).model_copy(update={"media_type": "application/zip"})
    reader = RegisteredArtifactReader([ref], run_id=ref.run_id)
    reader.verify(ref.id)
    with pytest.raises(ArtifactReadError, match="Cannot read as UTF-8 text"):
        reader.read_text(ref.id)
    tool = ReadFileTool(_boundary(tmp_path))
    with pytest.raises(ValueError, match="Cannot read as UTF-8 text"):
        tool.execute(_state(), tool.input_model(path=path.name))
    # A binary format error must not hide a frozen-content integrity failure.
    path.write_bytes(buffer.getvalue() + b"changed")
    with pytest.raises(ArtifactReadError, match="sha256"):
        reader.read_text(ref.id)


@pytest.mark.parametrize("body", ["", "中文 α 😀\n", "one\r\ntwo\rthree\n", "literal \ufffd\n"])
def test_text_reads_preserve_valid_utf8_regardless_of_file_label(tmp_path, body):
    path = tmp_path / "unknown.bin"
    path.write_bytes(body.encode("utf-8"))
    ref = _artifact(path).model_copy(update={"media_type": "application/octet-stream"})
    file_tool = ReadFileTool(_boundary(tmp_path))
    artifact_tool = ReadArtifactTool(RegisteredArtifactReader([ref], run_id=ref.run_id))
    assert file_tool.execute(_state(), file_tool.input_model(path=path.name)).value["content"] == body
    result = artifact_tool.execute(_state(), artifact_tool.input_model(artifact_id=ref.id))
    assert result.value["content"] == body
    assert result.memory_updates.get("read_artifact_ids", []) == ([ref.id] if body else [])


@pytest.mark.parametrize("registered", [False, True])
def test_failed_binary_read_recovers_without_recording_it_as_read(tmp_path, registered):
    from sciagent_contracts import AgentPermissions, AgentRequest, ModuleStatus, TaskBudget
    from sciagent_runtime import (
        AgentAction, AgentDefinition, AgentLoop, AllowListPermissionPolicy,
        CompletionDecision, ContextSection, FinishTool, InMemorySessionStore,
        ScriptedLLMClient,
    )

    bad = tmp_path / "binary.dat"
    bad.write_bytes(b"binary\x00\xff")
    good = tmp_path / "report.txt"
    good.write_text("original evidence", encoding="utf-8")
    if registered:
        refs = [_artifact(path).model_copy(update={"id": id_}) for path, id_ in
                [(bad, "artifact_bad"), (good, "artifact_good")]]
        tool = ReadArtifactTool(RegisteredArtifactReader(refs, run_id="run_reader"))
        arguments = [{"artifact_id": ref.id} for ref in refs]
        memory_key, expected = "read_artifact_ids", [refs[1].id]
    else:
        tool = ReadFileTool(_boundary(tmp_path))
        arguments = [{"path": path.name} for path in (bad, good)]
        memory_key, expected = "read_paths", [good.name]

    class AcceptFinish:
        def evaluate(self, state, candidate):
            return CompletionDecision(complete=candidate is not None, report="Finished")

    llm = ScriptedLLMClient(tool_turns([
        *(AgentAction(tool=tool.name, arguments=args) for args in arguments),
        AgentAction(tool="finish", arguments={"report": "Finished"}),
    ]))
    store = InMemorySessionStore()
    result = AgentLoop(store=store).run(
        AgentDefinition(
            name="reader", owner=AgentOwner.EXECUTOR, system_prompt="Read the evidence.",
            tools=(tool, FinishTool()), llm_client=llm,
            context_builder=lambda request, state, limit: [
                ContextSection(name="task", content=request.instruction),
            ],
            permission_policy=AllowListPermissionPolicy({tool.name, "finish"}),
            completion_check=AcceptFinish(),
        ),
        AgentRequest(work_request_id="work_test",
            run_id="run_reader", task_id="task_reader", attempt_number=1,
            agent=AgentOwner.EXECUTOR, instruction="Read the evidence.",
            budget=TaskBudget(max_llm_calls=3, timeout_seconds=60), permissions=AgentPermissions(),
        ),
        session_id="session_reader",
    )
    assert result.status == ModuleStatus.COMPLETED
    from sciagent_runtime.tool_calling import native_input_text
    wire = native_input_text(llm.contexts[1], llm.schemas[1], llm.histories[1])
    assert "Cannot read as UTF-8 text" in wire
    assert "\ufffd" not in wire
    state = store.load("session_reader")
    assert state.memory[memory_key] == expected
    assert json.loads(state.tool_turns[0].tool_results[state.tool_turns[0].tool_calls[0].id])["ok"] is False


@pytest.mark.parametrize("start_line,end_line,start_char,end_char", [
    (None, None, 2, 7),
    (2, 2, 1, 5),
    (2, 3, 0, 100),
    (None, None, 100, None),
])
def test_workspace_and_artifact_share_character_windows_and_newlines(
    tmp_path, start_line, end_line, start_char, end_char,
):
    body = "header\r\n\u4e2d\u6587\U0001f600abc\r\ntail\r"
    path = tmp_path / "text.txt"
    path.write_bytes(body.encode("utf-8"))
    ref = _artifact(path)
    file_tool = ReadFileTool(_boundary(tmp_path), max_chars=5)
    artifact_tool = ReadArtifactTool(
        RegisteredArtifactReader([ref], run_id=ref.run_id),
    )
    options = {
        "start_line": start_line, "end_line": end_line,
        "start_char": start_char, "end_char": end_char,
    }
    file_value = file_tool.execute(
        _state(), file_tool.input_model(path=path.name, **options),
    ).value
    artifact_value = artifact_tool.reader.read_text(ref.id, max_chars=5, **options)
    for key in (
        "start_line", "end_line", "start_char", "end_char", "total_lines",
        "selected_chars", "next_start_char", "content", "truncated",
    ):
        assert file_value[key] == artifact_value[key]
    assert path.read_bytes() == body.encode("utf-8")


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("window_size", [128_000, 256_000, None])
def test_text_tools_continuation_recovers_every_source_character(tmp_path, registered, window_size):
    # Explicit windows which exactly fill the cap still have a continuation,
    # even though their requested content was not truncated.
    body = "中文😀abc\r\n" * 40_000 + "LAST_CHARACTER"
    path = tmp_path / "large.txt"
    path.write_bytes(body.encode("utf-8"))
    if registered:
        ref = _artifact(path)
        tool = ReadArtifactTool(RegisteredArtifactReader([ref], run_id=ref.run_id))
        source = {"artifact_id": ref.id}
    else:
        tool = ReadFileTool(_boundary(tmp_path))
        source = {"path": path.name}
    state = _state()
    pieces = []
    start = 0
    for _ in range(10):
        end = start + window_size if window_size is not None else None
        result = tool.execute(state, tool.input_model(
            **source, start_char=start, end_char=end,
        )).value
        assert result["start_char"] == start
        assert result["end_char"] == end
        assert result["content"] == body[start:end][:128_000]
        assert result["truncated"] is (len(body[start:end]) > 128_000)
        pieces.append(result["content"])
        next_start = result["next_start_char"]
        if next_start is None:
            break
        assert next_start == start + len(result["content"])
        assert next_start > start
        start = next_start
    else:
        pytest.fail("Text continuation did not reach the selected source end")
    assert "".join(pieces) == body
    assert path.read_bytes() == body.encode("utf-8")


@pytest.mark.parametrize("tool_type", [ReadFileTool, ReadArtifactTool])
def test_text_tools_explain_continuation_and_requested_window_scope(tool_type):
    guidance = tool_type.model_guidance
    assert "next_start_char" in guidance
    assert "total_lines" in guidance
    assert "selected_chars" in guidance
    assert "Keep the line range fixed" in guidance
    assert "truncated=False" in guidance
    assert "selected lines are exhausted" in guidance



def test_read_and_search_git_text_use_the_shared_workspace_boundary(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("ref: refs/heads/target\n")
    scope = _boundary(tmp_path)
    reader = ReadFileTool(scope)
    read = reader.execute(_state(), reader.input_model(path=".git/HEAD"))
    assert read.value["content"] == "ref: refs/heads/target\n"
    search = SearchTextTool(scope)
    for path in (".git/HEAD", ".git", "."):
        result = search.execute(_state(), search.input_model(path=path, query="target"))
        assert result.value["matches"] == [
            {"path": str(tmp_path / ".git/HEAD"), "line": 1, "text": "ref: refs/heads/target"}]
