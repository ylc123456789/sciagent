"""Freeze the current model-visible schemas, docstrings and guidance.

The JSON fixture is a reviewed regression baseline, not an automatic snapshot.
Review intended tool changes before updating it; schema history belongs in Git
and docs/history. Behavioral tests separately check execution and boundaries.
"""

import hashlib
import json
from pathlib import Path

from resagent2_capabilities import (
    CreateFileTool,
    DeletePathTool,
    GitDiffTool,
    FetchLiteratureTool,
    ListFilesTool,
    LiteratureSearchTool,
    ReadArtifactTool,
    ReadFileTool,
    ReplaceTextTool,
    SearchTextTool,
    WebFetchTool,
    WebSearchTool,
)
from resagent2_capabilities import RunShellTool
from resagent2_runtime import AskUserTool, FinishTool
from resagent2_scientific.models import ScientificFinishInput
from resagent2_executor.completion import ExecutionFinishInput
from resagent2_runtime.tool_calling import native_tool_schemas
from resagent2_capabilities.delegation import (
    RequestWorkTool as GenericRequestWorkTool, UpdateWorkTool, AbandonWorkTool,
)
from resagent2_scientific.tools import (
    AskUserTool as ScientificAskUserTool,
    FinishTool as ScientificFinishTool,
    RequestWorkTool,
)


def tool_surface_fingerprints() -> dict[str, str]:
    groups = {
        "shared": (CreateFileTool, GitDiffTool, ListFilesTool,
                   LiteratureSearchTool, FetchLiteratureTool, ReadArtifactTool,
                   ReadFileTool, ReplaceTextTool, RunShellTool, SearchTextTool),
        "coding": (DeletePathTool,),
        "delegation": (GenericRequestWorkTool, UpdateWorkTool, AbandonWorkTool),
        "runtime": (AskUserTool, FinishTool),
        "executor": (FinishTool,),
        "scientific": (ScientificAskUserTool, ScientificFinishTool, RequestWorkTool,
                       WebFetchTool, WebSearchTool),
    }
    result = {}
    for group, classes in groups.items():
        for cls in classes:
            # Other tools expose class-level schemas; shared finish binds a role input.
            tool = object.__new__(cls)
            if group == "scientific" and cls is ScientificFinishTool:
                tool = ScientificFinishTool(ScientificFinishInput)
            elif group == "executor":
                tool = FinishTool(ExecutionFinishInput)
            schema = native_tool_schemas((tool,))[0]
            text = json.dumps(schema, sort_keys=True, ensure_ascii=False)
            result[f"{group}/{cls.name}"] = hashlib.sha256(text.encode()).hexdigest()
    return result


def test_model_visible_tool_surface_matches_unified_protocol() -> None:
    baseline = json.loads(Path(__file__).with_name("tool_surface_unified_v2.json").read_text())
    assert tool_surface_fingerprints() == baseline


def test_unified_finish_keeps_one_tool_and_result_shape_with_role_input() -> None:
    assert ScientificFinishTool is FinishTool
    assert set(FinishTool.input_model.model_fields) == set(ScientificFinishInput.model_fields) == set(ExecutionFinishInput.model_fields) == {
        "status", "report", "artifacts",
    }
    assert FinishTool().name == ScientificFinishTool(ScientificFinishInput).name == "finish"
