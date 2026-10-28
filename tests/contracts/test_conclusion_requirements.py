"""Explicit Run delivery requirements stay separate from task semantics."""

import pytest
from pydantic import ValidationError

from resagent2_contracts import (
    ConclusionRequirements, ResearchRequest, RunBudget, RunPermissions,
    TaskProposal, WorkRequestDraft, WorkflowTask,
)


def research_request(**changes):
    return ResearchRequest(**{
        "goal": "Produce metrics.json and compare the models",
        "budget": RunBudget(max_llm_calls=5, timeout_seconds=60),
        "permissions": RunPermissions(),
        **changes,
    })


def test_required_artifacts_default_to_empty_without_inferring_goal():
    assert research_request().required_artifacts == []
    assert ConclusionRequirements().required_artifacts == []


@pytest.mark.parametrize("model", [research_request, ConclusionRequirements])
def test_required_artifacts_round_trip_as_exact_logical_output_names(model):
    names = ["metrics.json", "Metrics.json", "summary_v2", "metrics.json"]
    value = model(required_artifacts=names)
    restored = type(value).model_validate_json(value.model_dump_json())
    assert restored.required_artifacts == names
    assert restored.schema_version == "32.0"


@pytest.mark.parametrize("name", [
    "", " reports", "reports ", "../metrics.json", "reports/metrics.json",
    r"reports\metrics.json", "/metrics.json", r"C:\metrics.json", "metric value",
    "1metrics", "*", "metrics.json?", "结果", "m" * 129,
])
@pytest.mark.parametrize("model", [research_request, ConclusionRequirements])
def test_required_artifacts_reject_paths_and_invalid_logical_names(model, name):
    with pytest.raises(ValidationError, match="required_artifacts"):
        model(required_artifacts=[name])


def test_required_artifacts_do_not_become_task_or_work_request_fields():
    for model in (TaskProposal, WorkflowTask, WorkRequestDraft):
        assert "required_artifacts" not in model.model_fields


@pytest.mark.parametrize("model", [research_request, ConclusionRequirements])
@pytest.mark.parametrize("schema_version", ["16.0", "17.0", "23.0"])
def test_required_artifacts_reject_previous_schema(model, schema_version):
    with pytest.raises(ValidationError, match="schema_version"):
        model(schema_version=schema_version)
