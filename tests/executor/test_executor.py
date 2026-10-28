"""Frozen configurations preserve execution facts, authority and resumption."""
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from e2e.native_fixtures import tool_turns
from resagent2_components import runtime_authority_context
from resagent2_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ExecutionCatalogEntry,
    ExecutionDefinition, RecordedAnswer,
    TaskBudget, WorkspaceAccess, WorkspaceGrant,
)
from resagent2_executor import (
    NativeExecutorAgent, build_coding_definition, build_experiment_definition,
    default_execution_registry, validate_execution_definition,
)
from resagent2_runtime import JsonSessionStore, ScriptedLLMClient


def request(root, definition=None, **updates):
    values = dict(run_id="run_config", work_request_id="work_config", task_id="task_config",
        attempt_number=1, agent=AgentOwner.EXECUTOR, instruction="Complete the original task",
        execution_definition=definition, budget=TaskBudget(max_llm_calls=8, timeout_seconds=30),
        workspace=WorkspaceGrant(root=str(root), source="local",
            access=WorkspaceAccess(read_paths=["."], write_paths=["."])),
        output_dir=str(root.parent / "results"), permissions=AgentPermissions(execute_commands=True))
    values.update(updates)
    return AgentRequest(**values)


def init_repo(root):
    root.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "value.py").write_text("VALUE = 1\n")


def finish(**extra):
    return {"tool": "finish", "arguments": {"report": "Original report", **extra}}


def approve(req, result):
    question = result.control.question
    answer = RecordedAnswer(question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, answered_at=datetime.now(UTC), run_id=req.run_id,
        task_id=req.task_id, attempt_number=req.attempt_number, session_id=result.session.id,
        action=question.action)
    return req.model_copy(update={"parent_session_id": result.session.id,
        "resume_answers": [answer], "control_ack": result.control.origin})


def test_frozen_definition_is_required_before_model_resolution(tmp_path):
    resolved = []
    agent = NativeExecutorAgent(lambda profile: resolved.append(profile))
    result = agent.invoke(request(tmp_path))
    assert result.status == "blocked" and "frozen execution definition" in result.report
    assert resolved == []


@pytest.mark.parametrize("updates,reason", [
    ({"tools": ["fetch_secrets"]}, "Unknown executor tools"),
    ({"preparation": "arbitrary.module:prepare"}, "Unsupported or inconsistent"),
    ({"completion": "artifacts@99"}, "Unsupported or inconsistent"),
    ({"finalizer": "execution@99"}, "Unsupported or inconsistent"),
    ({"context_builder": "executor@99"}, "Unsupported context builder"),
    ({"implementation_version": "99"}, "Unsupported executor implementation"),
    ({"tools": ["create_file"]}, "require the Git"),
    ({"tools": ["git_diff"]}, "require the Git"),
])
def test_unknown_or_incompatible_components_do_not_touch_workspace(tmp_path, updates, reason):
    base = build_experiment_definition().config
    definition = ExecutionDefinition(ref="custom@1", config=base.model_copy(update=updates))
    resolved = []
    agent = NativeExecutorAgent(lambda profile: resolved.append(profile))
    root = tmp_path / "not-created"
    result = agent.invoke(request(root, definition))
    assert result.status == "blocked" and reason in result.report
    assert resolved == [] and not root.exists()


def test_unregistered_model_profile_has_no_fallback(tmp_path):
    def resolve(profile):
        raise KeyError(f"Unregistered model profile: {profile}")
    result = NativeExecutorAgent(resolve).invoke(request(tmp_path,
        build_experiment_definition(model_profile="not-authorized")))
    assert result.status == "blocked" and "Unregistered model profile" in result.report


def test_host_constructed_configuration_uses_same_factory_and_actual_tools(tmp_path):
    original = build_experiment_definition()
    definition = ExecutionDefinition(ref="host-analysis@1", config=original.config.model_copy(update={
        "system_prompt": "Analyze the authorized file and report its value.",
        "tools": ["read_file"], "model_profile": "host-allowed",
    }))
    (tmp_path / "result.json").write_text('{"value": 7}')
    client = ScriptedLLMClient(tool_turns([
        {"tool": "read_file", "arguments": {"path": "result.json"}}, finish(),
    ]))
    profiles = []
    def resolve(profile):
        profiles.append(profile)
        assert profile == "host-allowed"
        return client
    agent = NativeExecutorAgent(resolve)
    result = agent.invoke(request(tmp_path, definition))
    assert result.status == "completed" and result.session.module == AgentOwner.EXECUTOR
    assert profiles == ["host-allowed"]
    names = {item["function"]["name"] for item in client.schemas[0]}
    assert names == {"read_file", "ask_user", "finish", "update_work", "abandon_work"}
    state = agent.loop.store.load(result.session.id)
    assert state.execution_ref == definition.ref
    assert state.execution_config_hash == definition.config_hash


def test_declared_shell_does_not_grant_execution_permission(tmp_path):
    marker = tmp_path / "unauthorized"
    client = ScriptedLLMClient(tool_turns([
        {"tool": "run_shell", "arguments": {"command": "touch unauthorized"}},
    ]))
    agent = NativeExecutorAgent(lambda _: client)
    result = agent.invoke(request(tmp_path, build_experiment_definition(),
        permissions=AgentPermissions(execute_commands=False)))
    assert result.status == "failed" and not marker.exists()
    assert not any(item.kind == "execution_record" for item in result.artifacts)


def test_configuration_delegation_is_filtered_by_permission_and_depth(tmp_path):
    definition = build_experiment_definition()
    client = ScriptedLLMClient(tool_turns([finish()]))
    result = NativeExecutorAgent(lambda _: client).invoke(request(tmp_path, definition,
        work_depth=3, max_work_depth=3, permissions=AgentPermissions(request_work=False)))
    assert result.status == "completed"
    assert "request_work" not in {schema["function"]["name"] for schema in client.schemas[0]}


def test_recreated_executor_preserves_original_git_baseline_and_checkpoint(tmp_path):
    root = tmp_path / "repo"
    init_repo(root)
    store = JsonSessionStore(tmp_path / "sessions")
    definition = build_coding_definition()
    first_client = ScriptedLLMClient(tool_turns([
        {"tool": "replace_text", "arguments": {"path": "value.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"}},
        {"tool": "run_shell", "arguments": {"command": shlex.quote(sys.executable) + " -c 'pass'"}},
    ]))
    req = request(root, definition)
    first_agent = NativeExecutorAgent(lambda _: first_client, store=store)
    paused = first_agent.invoke(req)
    assert paused.status == "needs_user_input" and (root / "value.py").read_text() == "VALUE = 2\n"
    original_baseline = store.load(paused.session.id).memory["workspace_snapshot"]
    second_client = ScriptedLLMClient(tool_turns([finish()]))
    resumed_agent = NativeExecutorAgent(lambda _: second_client, store=JsonSessionStore(tmp_path / "sessions"))
    completed = resumed_agent.invoke(approve(req, paused))
    assert completed.status == "completed", completed.report
    assert resumed_agent.loop.store.load(completed.session.id).memory["workspace_snapshot"] == original_baseline
    patch = (root.parent / "results" / next(
        item.path for item in completed.artifacts if item.kind == "code_patch")).read_text()
    assert "-VALUE = 1" in patch and "+VALUE = 2" in patch
    assert len(json.loads(next(item.content for item in completed.artifacts if item.kind == "execution_record"))["results"]) == 1
    def no_resolution(_):
        raise AssertionError("checkpoint must not resolve clients or execute preparation")
    reader = NativeExecutorAgent(no_resolution, store=JsonSessionStore(tmp_path / "sessions"))
    checkpoint = reader.read_checkpoint_result(run_id=req.run_id, task_id=req.task_id, attempt_number=1)
    assert checkpoint.result.artifacts == completed.artifacts
    assert checkpoint.result.report == "Original report"


@pytest.mark.parametrize("change", ["config_hash", "execution_ref", "foreign_task"])
def test_resume_binding_is_rejected_before_model_resolution_or_workspace_preparation(tmp_path, monkeypatch, change):
    from resagent2_contracts import WorkspaceSpec

    root = tmp_path / "workspace"
    root.mkdir()
    store = JsonSessionStore(tmp_path / "sessions")
    client = ScriptedLLMClient(tool_turns([
        {"tool": "ask_user", "arguments": {"text": "Need a definition", "requested_fields": ["definition"]}},
    ]))
    agent = NativeExecutorAgent(lambda _: client, store=store)
    definition = build_experiment_definition()
    req = request(root, definition)
    paused = agent.invoke(req)
    assert paused.status == "needs_user_input"
    session_path = store.root / f"{paused.session.id}.json"
    original = session_path.read_bytes()
    updates = {"parent_session_id": paused.session.id,
        "workspace_spec": WorkspaceSpec(workspace_id="ws_config", source_kind="local",
            location=str(root), access=req.workspace.access)}
    if change == "config_hash":
        updates["execution_definition"] = ExecutionDefinition(ref=definition.ref,
            config=definition.config.model_copy(update={
                "max_context_tokens": definition.config.max_context_tokens + 1,
            }))
    elif change == "execution_ref":
        updates["execution_definition"] = ExecutionDefinition(ref="another@1", config=definition.config)
    else:
        # The other Task deliberately has the same valid definition and hash.
        updates["task_id"] = "task_foreign"

    def unexpected_call(*args, **kwargs):
        pytest.fail("A mismatched Session must be rejected before model resolution or materialization")

    resumed = NativeExecutorAgent(unexpected_call, store=JsonSessionStore(store.root))
    monkeypatch.setattr("resagent2_executor.agent.RepoMaterializer.materialize", unexpected_call)
    rejected = resumed.invoke(req.model_copy(update=updates))
    assert rejected.status == "failed"
    assert rejected.error.code == "contract_error"
    assert "resume target" in rejected.report
    assert len(client.contexts) == 1
    assert session_path.read_bytes() == original


def test_coding_task_implements_fits_and_evaluates_with_actual_result_artifacts(tmp_path):
    root = tmp_path / "repo"
    init_repo(root)
    out = tmp_path / "results"
    script = """import json, pathlib, sys
fit_x, fit_y = [1., 2., 3., 4.], [2., 4., 6., 8.]
slope = sum(x*y for x,y in zip(fit_x, fit_y)) / sum(x*x for x in fit_x)
test_x, test_y = [5., 6.], [10., 12.]
prediction = [slope*x for x in test_x]
mse = sum((a-b)**2 for a,b in zip(prediction, test_y)) / len(test_y)
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"slope": slope, "test_mse": mse, "prediction": prediction}))
dependency = pathlib.Path("environment/lib/site-packages/local_dependency")
dependency.mkdir(parents=True)
for index in range(128):
    (dependency / f"module_{index}.py").write_text(f"VALUE = {index}\\n")
"""
    client = ScriptedLLMClient(tool_turns([
        {"tool": "create_file", "arguments": {"path": "fit.py", "content": script}},
        {"tool": "run_shell", "arguments": {"command": " ".join(map(shlex.quote,
            [sys.executable, "fit.py", str(out / "metrics.json")]))}},
        finish(artifacts=[{"kind": "experiment_result", "path": "metrics.json",
            "media_type": "application/json", "summary": "Actual held-out evaluation", "output_name": "metrics"}]),
    ]))
    agent = NativeExecutorAgent(lambda _: client)
    req = request(root, build_coding_definition(),
        instruction="Implement a linear fitting entry point; fit on the supplied training pairs and evaluate held-out pairs, delivering actual metrics.")
    paused = agent.invoke(req)
    assert paused.status == "needs_user_input" and not (out / "metrics.json").exists()
    completed = agent.invoke(approve(req, paused))
    assert completed.status == "completed", completed.report
    assert json.loads((out / "metrics.json").read_text()) == {"slope": 2.0, "test_mse": 0.0, "prediction": [10.0, 12.0]}
    assert sorted(item.kind for item in completed.artifacts) == ["code_patch", "execution_record", "experiment_result"]
    patch = (out / next(item.path for item in completed.artifacts if item.kind == "code_patch")).read_text()
    assert "fit.py" in patch
    assert "environment/lib/site-packages/local_dependency/module_0.py" in patch
    assert "environment/lib/site-packages/local_dependency/module_127.py" in patch
    assert len(list((root / "environment/lib/site-packages/local_dependency").glob("*.py"))) == 128
    rows = json.loads(next(item.content for item in completed.artifacts if item.kind == "execution_record"))["results"]
    assert rows[0]["exit_code"] == 0 and "fit.py" in rows[0]["command"]


def test_default_catalog_and_definitions_have_one_source():
    registry = default_execution_registry(model_profile="host")
    assert {preset.preset_ref for preset in registry.presets} == {"coding@1", "experiment@1"}
    for preset in registry.presets:
        definition = registry.execution_definitions[preset.execution_ref]
        assert validate_execution_definition(definition) == definition
        assert definition.config.model_profile == "host"


@pytest.mark.parametrize("owner", [AgentOwner.SCIENTIFIC, AgentOwner.EXECUTOR])
def test_parent_catalog_and_executor_guidance_share_capability_boundaries(tmp_path, owner):
    registry = default_execution_registry()
    catalog = [ExecutionCatalogEntry.model_validate(item.model_dump()) for item in registry.presets]
    definition = registry.execution_definitions["coding@1"] if owner == AgentOwner.EXECUTOR else None
    req = request(tmp_path, definition, agent=owner, execution_catalog=catalog,
        permissions=AgentPermissions(execute_commands=True, request_work=True),
        work_depth=1, max_work_depth=3)
    visible = json.loads(runtime_authority_context(req).content)["available_execution_presets"]
    by_ref = {item["preset_ref"]: item for item in visible}
    for preset in registry.presets:
        description = by_ref[preset.preset_ref]["description"]
        assert description == preset.description
        assert description in registry.execution_definitions[preset.execution_ref].config.system_prompt
        assert "Creating or changing temporary, auxiliary or driver scripts is also code implementation." in description

    coding = by_ref["coding@1"]["description"]
    assert "continue with fitting, training, research measurement and evaluation in the same task" in coding
    assert "source-editing tools, Git-baseline patch/change delivery" in coding
    experiment = by_ref["experiment@1"]["description"]
    assert "use available code" in experiment
    assert "no source-editing tools or Git-change delivery" in experiment
    assert "Do not use Shell to bypass missing source-implementation or change-delivery support" in experiment
    assert "trusted host execution, not a read-only sandbox" in experiment



@pytest.mark.parametrize("ending", ["completed", "failed", "exhausted"])
def test_real_finish_freezes_non_utf8_patch_file_and_json_reports(tmp_path, monkeypatch, ending):
    import hashlib
    from urllib.parse import urlparse
    from resagent2_orchestrator.artifacts import ArtifactRegistry
    from resagent2_runtime import PermissionDecision

    root = tmp_path / "repo"
    init_repo(root)
    req = request(root, build_coding_definition())
    (root / "obsolete.py").write_text("OBSOLETE = True\n")
    payload = b"%PDF-1.4\nraw no-NUL content: \xac\xff\r\n%%EOF\n"
    binary_payload = b"\x00\x01\xffbinary\r\n"
    script = (
        "from pathlib import Path; "
        f"Path('payload.pdf').write_bytes(bytes.fromhex('{payload.hex()}')); "
        f"Path('payload.bin').write_bytes(bytes.fromhex('{binary_payload.hex()}')); "
        "Path('value.py').write_text('VALUE = 2\\n'); "
        "Path('obsolete.py').unlink(); "
        "Path('report.json').write_text('{\"ok\": true}'); "
        "Path('measurements.json').write_text('{\"value\": 7}')"
    )
    actions = [{"tool": "run_shell", "arguments": {
        "command": shlex.quote(sys.executable) + " -c " + shlex.quote(script),
    }}]
    if ending != "exhausted":
        actions.append(finish(status=ending, artifacts=[
            {"kind": "experiment_result", "path": path, "media_type": "application/json",
             "summary": "Actual JSON report", "output_name": name}
            for path, name in [("report.json", "report"), ("measurements.json", "measurements")]
        ]))
    monkeypatch.setattr("resagent2_capabilities.permissions.OperationPermissionPolicy.check",
                        lambda *_: PermissionDecision(outcome="allow"))
    client = ScriptedLLMClient(tool_turns(actions))
    store = JsonSessionStore(tmp_path / "sessions")
    result = NativeExecutorAgent(lambda _: client, store=store).invoke(req)
    assert result.status == ("failed" if ending == "exhausted" else ending), result.report
    assert result.llm_calls == 2
    patch = next(item for item in result.artifacts if item.kind == "code_patch")
    assert patch.content is None
    raw = (Path(req.output_dir) / patch.path).read_bytes()
    assert b"GIT binary patch" in raw
    assert b"deleted file mode" in raw and b"-OBSOLETE = True" in raw
    assert b"-VALUE = 1" in raw and b"+VALUE = 2" in raw
    assert not any(item.kind == "code_change" for item in result.artifacts)
    assert sum(item.kind == "code_patch" for item in result.artifacts) == 1
    assert b"+raw no-NUL content: \xac\xff\r\n" in raw
    registry = ArtifactRegistry(tmp_path / "registry")
    refs = [registry.register(item, grant=req.workspace, output_dir=req.output_dir,
            producer="executor", run_id=req.run_id, task_id=req.task_id,
            attempt_number=1, index=index) for index, item in enumerate(result.artifacts)]
    frozen = next(ref for ref in refs if ref.kind == "code_patch")
    assert frozen.metadata["source_root"] == "output_dir"
    assert frozen.sha256 == hashlib.sha256(raw).hexdigest()
    assert Path(urlparse(frozen.uri).path).read_bytes() == raw
    if ending != "exhausted":
        assert result.report == "Original report"
        assert {ref.output_name for ref in refs} >= {"report", "measurements"}
    else:
        assert patch.metadata["diagnostic"] is True
    record = next(item for item in result.artifacts if item.kind == "execution_record")
    assert json.loads(record.content)["results"][0]["exit_code"] == 0
    saved = store.load(result.session.id)
    assert saved.terminal_result.artifacts == result.artifacts
    assert saved.terminal_result.report == result.report
    assert saved.terminal_result.error == result.error
    assert saved.status == result.session.status
    # Apply the registered immutable patch to the exact initial workspace state.
    for path in ["payload.pdf", "payload.bin", "report.json", "measurements.json"]:
        (root / path).unlink()
    (root / "value.py").write_text("VALUE = 1\n")
    (root / "obsolete.py").write_text("OBSOLETE = True\n")
    subprocess.run(["git", "apply", urlparse(frozen.uri).path], cwd=root, check=True)
    assert (root / "payload.pdf").read_bytes() == payload
    assert (root / "payload.bin").read_bytes() == binary_payload
    assert (root / "value.py").read_text() == "VALUE = 2\n"
    assert not (root / "obsolete.py").exists()
    assert json.loads((root / "report.json").read_text()) == {"ok": True}
    assert json.loads((root / "measurements.json").read_text()) == {"value": 7}


def test_completion_and_diagnostic_failures_preserve_original_saved_result_and_usage(tmp_path, monkeypatch):
    from resagent2_components import GitWorkspace
    from resagent2_runtime import PermissionDecision

    root = tmp_path / "repo"
    init_repo(root)
    req = request(root, build_coding_definition())
    calls = []
    def unavailable_patch(self, baseline, path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("primary patch storage failure")
        raise UnicodeDecodeError("utf-8", b"\xac", 0, 1, "diagnostic decoder failure")
    monkeypatch.setattr(GitWorkspace, "write_patch_since", unavailable_patch)
    monkeypatch.setattr("resagent2_capabilities.permissions.OperationPermissionPolicy.check",
                        lambda *_: PermissionDecision(outcome="allow"))
    client = ScriptedLLMClient(tool_turns([
        {"tool": "run_shell", "arguments": {"command": "printf 'VALUE = 2\n' > value.py"}},
        finish(),
    ]))
    store = JsonSessionStore(tmp_path / "sessions")
    result = NativeExecutorAgent(lambda _: client, store=store).invoke(req)
    assert len(calls) == 2
    assert result.status == result.session.status == "failed"
    assert result.error.code == "contract_error"
    assert result.error.message == "completion check failed: primary patch storage failure"
    assert "diagnostic decoder failure" in result.error.details["diagnostic_patch_error"]
    assert result.error.retryable is False
    assert result.llm_calls == len(client.contexts) == 2
    assert (root / "value.py").read_text() == "VALUE = 2\n"
    record = next(item for item in result.artifacts if item.kind == "execution_record")
    assert json.loads(record.content)["results"][0]["exit_code"] == 0
    saved = JsonSessionStore(store.root).load(result.session.id)
    assert saved.status == "failed" and saved.llm_calls_used == 2
    assert saved.terminal_result is not None
    assert saved.terminal_result.error == result.error
    assert saved.terminal_result.report == result.report
    assert saved.terminal_result.artifacts == result.artifacts
    assert saved.terminal_result.llm_calls == 2



def test_git_definition_requires_output_root_before_model_or_workspace_side_effects(tmp_path, monkeypatch):
    root = tmp_path / "not-created"
    agent = NativeExecutorAgent(lambda _: pytest.fail("missing output root must not resolve a model"))
    def forbidden(*args, **kwargs):
        pytest.fail("missing output root must not prepare or observe the workspace")
    monkeypatch.setattr("resagent2_executor.agent.RepoMaterializer.materialize", forbidden)
    monkeypatch.setattr("resagent2_executor.agent.GitWorkspace", forbidden)
    req = request(root, build_coding_definition(), output_dir=None)
    result = agent.invoke(req)
    assert result.status == "blocked"
    assert result.error.code == "invalid_input"
    assert "trusted output_dir" in result.report
    assert not root.exists()
    assert not (tmp_path / "results").exists()
    assert result.llm_calls == 0 and result.session is None and result.artifacts == []


def test_git_completion_requires_explicit_output_root(tmp_path):
    from resagent2_components import GitBaseline, WorkspaceBoundary
    from resagent2_executor.completion import ExecutionCompletionCheck
    boundary = WorkspaceBoundary(WorkspaceGrant(root=str(tmp_path), source="local",
        access=WorkspaceAccess(read_paths=["."], write_paths=["."])))
    with pytest.raises(ValueError, match="trusted output_dir"):
        ExecutionCompletionCheck(boundary, repository=object(), baseline=GitBaseline(tree_hash="baseline"))
