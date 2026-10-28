"""Check published guidance, not whether a real model will follow it."""

from sciagent_capabilities import ReplaceTextTool
from sciagent_contracts import ScientificVerdict
from sciagent_scientific.context import SCIENTIFIC_PROMPT


# Line wrapping is presentation; these checks cover a few prior failure modes.
SCIENTIFIC_GUIDANCE = " ".join(SCIENTIFIC_PROMPT.split())


def test_scientific_opinion_guidance_publishes_its_content_enum() -> None:
    from sciagent_scientific.models import ScientificFinishInput
    schema = ScientificFinishInput.model_json_schema()
    assert schema["$defs"]["ScientificVerdict"]["enum"] == [value.value for value in ScientificVerdict]
    assert "structured opinion content defined by the finish tool schema" in SCIENTIFIC_GUIDANCE
    assert "independently of finish.status" in SCIENTIFIC_GUIDANCE


def test_scientific_prompt_keeps_own_tool_failures_out_of_request_work() -> None:
    assert "not a substitute for your own tools" in SCIENTIFIC_GUIDANCE
    assert "not a reason to delegate literature work" in SCIENTIFIC_GUIDANCE
    assert "When all available sources fail or necessary evidence remains blocked" in SCIENTIFIC_GUIDANCE
    assert "ask_user: explain the actual error" in SCIENTIFIC_GUIDANCE
    assert "explicit decision on proceeding with limited evidence" in SCIENTIFIC_GUIDANCE


def test_scientific_prompt_distinguishes_previews_from_visible_evidence() -> None:
    assert "short search previews do not establish" in SCIENTIFIC_GUIDANCE
    assert "Reading records are historical logs" in SCIENTIFIC_GUIDANCE
    assert "Read the needed source sections" in SCIENTIFIC_GUIDANCE
    assert "do not guess omitted content" in SCIENTIFIC_GUIDANCE


def test_scientific_prompt_separates_known_prerequisites_from_contingencies() -> None:
    assert "Include already-known prerequisites" in SCIENTIFIC_GUIDANCE
    assert "repair before the dependent measurement" in SCIENTIFIC_GUIDANCE
    assert "within one coherent task" in SCIENTIFIC_GUIDANCE
    assert "Do not run a known-broken experiment" in SCIENTIFIC_GUIDANCE
    assert "Distinguish known prerequisites from hypothetical failures" in SCIENTIFIC_GUIDANCE
    assert "request repair only after a failure is observed" in SCIENTIFIC_GUIDANCE


def test_edit_guidance_allows_multiple_uniquely_matching_edits() -> None:
    guidance = ReplaceTextTool.model_guidance
    assert "old_text must match exactly once" in guidance
    assert "current file per call" in guidance
    assert "Multiple calls are allowed" in guidance


def test_reports_carry_relevant_limits_without_becoming_independent_measurements() -> None:
    assert "carry relevant limitations into your judgment" in SCIENTIFIC_GUIDANCE
    assert "not independent measurements" in SCIENTIFIC_GUIDANCE
    assert "same data do not provide independent confirmation" in SCIENTIFIC_GUIDANCE
    assert "Read historical reports only when needed" in SCIENTIFIC_GUIDANCE



def test_execution_agents_keep_environment_choice_in_general_task_guidance() -> None:
    from sciagent_coding.context import CODING_PROMPT
    from sciagent_experiment.context import EXPERIMENT_PROMPT
    for prompt in (CODING_PROMPT, EXPERIMENT_PROMPT):
        guidance = " ".join(prompt.split())
        assert "discover, select and reuse" in guidance
        assert "user-specified versions and authorization" in guidance
        assert "each call starts a fresh shell" in guidance
        assert "do not persist" in guidance
        assert "prepare_environment" not in guidance
        assert "run_setup" not in guidance
        assert "run_verification" not in guidance
        assert "audit_env" not in guidance


def test_request_work_schema_exposes_stable_configuration_selection() -> None:
    from sciagent_scientific.tools import RequestWorkTool

    schema = RequestWorkTool.input_model.model_json_schema()
    work_schema = schema["$defs"]["WorkRequestDraft"]
    task_ref = work_schema["properties"]["tasks"]["items"]["$ref"]
    task_schema = schema["$defs"][task_ref.rsplit("/", 1)[1]]
    assert set(task_schema["properties"]) == {
        "schema_version", "key", "selection", "instruction", "depends_on", "workspace_id", "input_artifact_ids",
        "input_artifact_bindings", "output_names",
    }
    assert task_schema["additionalProperties"] is False
    assert "WorkflowAgentKind" not in schema["$defs"]

    selection = schema["$defs"]["PresetSelection"]
    assert selection["properties"]["preset_ref"]["type"] == "string"
    assert "enum" not in selection["properties"]["preset_ref"]
    assert "selection" in task_schema["required"]
    assert "NoMatchSelection" in schema["$defs"]
