"""Optional metadata cannot invalidate navigation for a registered artifact."""

import pytest
from pydantic import ValidationError

from resagent2_contracts import ArtifactIndexEntry, ArtifactRef


def ref(metadata):
    return ArtifactRef(id="artifact_import", kind="input", producer="orchestrator",
        run_id="run_test", uri="file:///input.txt", sha256="a" * 64,
        media_type="text/plain", summary="Supplied material",
        metadata={**metadata, "source_type": "import"})


def test_index_preserves_valid_optional_metadata_without_mutating_ref():
    metadata = dict(source_artifact_id="artifact_original", source_type="import",
        agent_started=False, source_url="https://example.org/source",
        final_url="https://example.org/final", paper_id="paper-exact")
    original = ref(metadata)
    before = original.model_dump_json()

    entry = ArtifactIndexEntry.from_ref(original)

    assert entry.model_dump(include=set(metadata)) == metadata
    assert entry.artifact_id == original.id
    assert original.model_dump_json() == before


@pytest.mark.parametrize("metadata", [
    {"source_artifact_id": "not-an-artifact-id"},
    {"source_artifact_id": ["artifact_other"]},
    {"agent_started": "false"}, {"agent_started": 1},
    {"source_url": 12}, {"final_url": {"url": "https://example.org"}},
    {"paper_id": ["paper-other"]},
])
def test_index_omits_only_invalid_optional_metadata(metadata):
    original = ref({**metadata, "unrelated": {"retained": True}})
    before = original.model_dump_json()

    entry = ArtifactIndexEntry.from_ref(original)

    assert entry.artifact_id == original.id
    assert entry.source_type == "import"
    assert all(getattr(entry, name) is None for name in metadata)
    assert original.model_dump_json() == before


def test_index_does_not_suppress_invalid_required_identity():
    original = ref({"source_artifact_id": "invalid"}).model_copy(update={"id": "invalid"})
    with pytest.raises(ValidationError, match="artifact_id"):
        ArtifactIndexEntry.from_ref(original)
