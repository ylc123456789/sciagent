"""Required deliveries use exact names on authorized, verified frozen artifacts."""

import hashlib
from pathlib import Path

import pytest

from resagent2_components import ArtifactReadError, RegisteredArtifactReader
from resagent2_components.artifacts import missing_required_artifacts
from resagent2_contracts import AgentOwner, ArtifactRef


def frozen_ref(tmp_path, *, artifact_id="artifact_metrics", output_name="metrics",
               kind="data", run_id="run_outputs", metadata=None, filename="values.json"):
    path = tmp_path / filename
    path.write_text('{"accuracy": 0.8}', encoding="utf-8")
    return ArtifactRef(
        id=artifact_id, kind=kind, producer=AgentOwner.EXECUTOR,
        run_id=run_id, task_id="task_measure", attempt_number=1,
        uri=path.as_uri(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        media_type="application/json", summary="Measured values",
        output_name=output_name, metadata=metadata or {},
    ), path


def missing(names, refs, *, ids=None):
    reader = RegisteredArtifactReader(refs, run_id="run_outputs")
    return missing_required_artifacts(
        names, artifact_ids=[ref.id for ref in refs] if ids is None else ids, reader=reader,
    )


def test_exact_logical_name_is_independent_of_filename_and_kind(tmp_path):
    ref, _ = frozen_ref(tmp_path)
    assert missing(["metrics"], [ref]) == []
    assert missing(["Metrics", "metrics.json", "values.json", "data"], [ref]) == [
        "Metrics", "metrics.json", "values.json", "data",
    ]


@pytest.mark.parametrize("fields", [
    {"output_name": None, "filename": "metrics"},
    {"output_name": None, "kind": "metrics"},
    {"output_name": None, "metadata": {"output_name": "metrics"}},
    {"output_name": "Metrics"},
])
def test_filename_kind_metadata_and_case_cannot_supply_output_name(tmp_path, fields):
    ref, _ = frozen_ref(tmp_path, **fields)
    assert missing(["metrics"], [ref]) == ["metrics"]


def test_cross_run_and_unauthorized_artifacts_do_not_satisfy_delivery(tmp_path):
    own, _ = frozen_ref(tmp_path)
    foreign, _ = frozen_ref(
        tmp_path, artifact_id="artifact_foreign", run_id="run_other", filename="foreign.json",
    )
    assert missing(["metrics"], [foreign]) == ["metrics"]
    assert missing(["metrics"], [own], ids=["artifact_unknown"]) == ["metrics"]
    assert missing(["metrics"], [], ids=[own.id]) == ["metrics"]


def test_existing_file_without_registered_ref_is_not_a_delivery(tmp_path):
    (tmp_path / "metrics").write_text("exists", encoding="utf-8")
    assert missing(["metrics"], [], ids=["artifact_metrics"]) == ["metrics"]


@pytest.mark.parametrize("fault,error", [
    ("corrupt", "sha256"), ("missing", "file is missing"),
])
def test_matching_output_must_still_verify_its_frozen_bytes(tmp_path, fault, error):
    ref, path = frozen_ref(tmp_path)
    if fault == "corrupt":
        path.write_text("modified after registration", encoding="utf-8")
    else:
        path.unlink()
    with pytest.raises(ArtifactReadError, match=error):
        missing(["metrics"], [ref])


@pytest.mark.parametrize("corrupt_tail", [False, True])
def test_binary_delivery_verifies_all_bytes_without_text_reads(tmp_path, monkeypatch, corrupt_tail):
    ref, path = frozen_ref(tmp_path, output_name="checkpoint", filename="weights.bin")
    payload = bytes(range(256)) * 4096 + b"\x80\xffbinary-tail"
    path.write_bytes(payload)
    ref = ref.model_copy(update={
        "sha256": hashlib.sha256(payload).hexdigest(),
        "media_type": "application/octet-stream",
    })
    if corrupt_tail:
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"\x00")

    def unexpected(*args, **kwargs):
        raise AssertionError("Delivery verification must not load or project full text")

    monkeypatch.setattr(RegisteredArtifactReader, "read_text", unexpected)
    monkeypatch.setattr(Path, "read_bytes", unexpected)
    if corrupt_tail:
        with pytest.raises(ArtifactReadError, match="sha256"):
            missing(["checkpoint"], [ref])
    else:
        assert missing(["checkpoint"], [ref]) == []


def test_unrelated_broken_artifact_is_not_read(tmp_path):
    ref, path = frozen_ref(tmp_path, output_name="unrelated")
    path.unlink()
    assert missing(["metrics"], [ref]) == ["metrics"]


def test_empty_requirements_do_not_resolve_or_read_artifacts(tmp_path, monkeypatch):
    ref, _ = frozen_ref(tmp_path)
    reader = RegisteredArtifactReader([ref], run_id="run_outputs")

    def unexpected(*args, **kwargs):
        raise AssertionError("No artifact lookup is needed for an empty requirement")

    monkeypatch.setattr(reader, "resolve_ref", unexpected)
    monkeypatch.setattr(reader, "verify", unexpected)
    monkeypatch.setattr(reader, "read_text", unexpected)
    assert missing_required_artifacts([], artifact_ids=[ref.id], reader=reader) == []


def test_duplicate_registered_names_are_existential_and_missing_names_are_stable(tmp_path):
    first, _ = frozen_ref(tmp_path)
    second, _ = frozen_ref(tmp_path, artifact_id="artifact_later", filename="later.json")
    assert missing(["metrics", "missing", "metrics", "other", "missing"], [first, second]) == [
        "missing", "other",
    ]


def test_live_resolver_cannot_return_a_different_id_or_run(tmp_path):
    ref, _ = frozen_ref(tmp_path)
    reader = RegisteredArtifactReader([], run_id="run_outputs", resolve=lambda _: ref)
    assert missing_required_artifacts(
        ["metrics"], artifact_ids=["artifact_unregistered"], reader=reader,
    ) == ["metrics"]
    foreign_reader = RegisteredArtifactReader([], run_id="run_other", resolve=lambda _: ref)
    assert missing_required_artifacts(
        ["metrics"], artifact_ids=[ref.id], reader=foreign_reader,
    ) == ["metrics"]
