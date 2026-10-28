"""Exact literature fetch preserves real materials and honest partial outcomes."""

from datetime import UTC, datetime

import pytest

from resagent2_capabilities import FetchLiteratureInput, FetchLiteratureTool
from resagent2_components import ArtifactReadError, RegisteredArtifactReader
from resagent2_components.literature import LiteraturePaper, LiteratureSearchError
from resagent2_components.literature.fulltext import PdfFetchError, PdfParseError, PdfText
from resagent2_contracts import AgentOwner, ArtifactCandidate, SessionStatus
from resagent2_orchestrator import ArtifactRegistry
from resagent2_runtime import AgentState


class Registration:
    def __init__(self, root):
        self.registry = ArtifactRegistry(root)
        self.refs = {}
        self.calls = []
    def register_scientific(self, candidate, *, run_id, session_id, operation_key, output_position, content_bytes=None):
        self.calls.append((operation_key, output_position))
        ref = self.registry.register_scientific(candidate, run_id=run_id, session_id=session_id,
                                                content_bytes=content_bytes)
        self.refs[ref.id] = ref
        return ref
    def list_artifacts(self, *, run_id, session_id):
        return [ref for ref in self.refs.values() if ref.run_id == run_id and ref.session_id == session_id]


class Backend:
    def __init__(self, paper=None, error=None):
        self.paper = paper or _paper()
        self.error = error
        self.calls = []
    def resolve(self, source):
        self.calls.append(source)
        if self.error:
            raise self.error
        return self.paper


def _paper(**updates):
    base = LiteraturePaper(paper_id="2401.12345v1", title="A paper", abstract="An abstract",
                           source_url="https://arxiv.org/abs/2401.12345v1",
                           pdf_url="https://arxiv.org/pdf/2401.12345v1")
    return base.model_copy(update=updates)


def _state():
    now = datetime.now(UTC)
    return AgentState(session_id="session_sci", agent_name="scientific", owner=AgentOwner.SCIENTIFIC,
                      run_id="run_pdf", status=SessionStatus.ACTIVE, created_at=now, updated_at=now,
                      current_operation_key="session:session_sci:turn:0:call:fetch")


def download(url, path):
    path.write_bytes(b"%PDF-1.7\ncontrolled bytes")
    return path


def parse(path):
    assert path.read_bytes().startswith(b"%PDF-")
    return PdfText("# PDF text\n\n## Page 1\n\nExact text.", 1, [], "test-parser")


def test_fetch_exact_source_preserves_metadata_pdf_text_and_source_chain(tmp_path):
    register, state, backend = Registration(tmp_path), _state(), Backend()
    result = FetchLiteratureTool(register, backend=backend, download=download, parse=parse).execute(
        state, FetchLiteratureInput(source="arxiv:2401.12345v1"))
    assert backend.calls == ["arxiv:2401.12345v1"]
    paper, pdf, text = result.artifacts
    assert result.ok and [ref.kind for ref in result.artifacts] == ["literature_paper", "literature_pdf", "literature_fulltext"]
    assert pdf.metadata["source_artifact_id"] == paper.id
    assert text.metadata["source_artifact_id"] == pdf.id and text.metadata["paper_artifact_id"] == paper.id
    reader = RegisteredArtifactReader(result.artifacts, run_id=state.run_id)
    assert "Provided abstract" in reader.read_text(paper.id)["content"]
    assert "## Page 1" in reader.read_text(text.id)["content"]
    assert reader.verified_path(pdf.id).read_bytes().startswith(b"%PDF-")
    assert result.memory_updates == {}
    assert register.calls == [(state.current_operation_key, position) for position in (0, 1, 2)]


def test_repeat_provider_fetch_reuses_verified_materials_after_tool_reconstruction(tmp_path):
    register, state, backend = Registration(tmp_path), _state(), Backend()
    args = FetchLiteratureInput(source="arxiv:2401.12345v1")
    first = FetchLiteratureTool(register, backend=backend, download=download, parse=parse).execute(state, args)
    state.authorized_artifact_ids = [ref.id for ref in first.artifacts]
    def unexpected(*args):
        raise AssertionError("unchanged snapshots must not redownload/reparse")
    second = FetchLiteratureTool(register, backend=backend, download=unexpected, parse=unexpected).execute(state, args)
    assert second.value["reused"] == {"paper": True, "pdf": True, "fulltext": True}
    assert second.value["fulltext_artifact_id"] == first.value["fulltext_artifact_id"]
    assert len(register.refs) == 3 and len(backend.calls) == 2


def test_unavailable_pdf_preserves_only_real_metadata(tmp_path):
    register, state = Registration(tmp_path), _state()
    result = FetchLiteratureTool(register, backend=Backend(_paper(pdf_url=None))).execute(
        state, FetchLiteratureInput(source="arxiv:2401.12345v1"))
    assert not result.ok and result.value["status"] == "unavailable"
    assert result.value["error_type"] == "no_pdf_url"
    assert len(result.artifacts) == len(register.refs) == 1
    assert result.value["pdf_artifact_id"] is result.value["fulltext_artifact_id"] is None


def test_resolution_failure_never_creates_fake_paper(tmp_path):
    register = Registration(tmp_path)
    backend = Backend(error=LiteratureSearchError("not found", error_type="not_found"))
    result = FetchLiteratureTool(register, backend=backend).execute(_state(), FetchLiteratureInput(source="doi:10.1234/missing"))
    assert not result.ok and result.value["error_type"] == "not_found"
    assert register.refs == {} and result.artifacts == []


def test_download_failure_retains_metadata_without_claiming_pdf(tmp_path):
    register = Registration(tmp_path)
    def unavailable(*args):
        raise PdfFetchError("HTTP 429")
    result = FetchLiteratureTool(register, backend=Backend(), download=unavailable).execute(
        _state(), FetchLiteratureInput(source="arxiv:2401.12345v1"))
    assert not result.ok and result.value["status"] == "unavailable"
    assert result.value["error_type"] == "download_failed"
    assert len(result.artifacts) == len(register.refs) == 1


def test_parse_failure_preserves_pdf_and_retry_reuses_frozen_original(tmp_path):
    register, state, backend = Registration(tmp_path), _state(), Backend()
    args = FetchLiteratureInput(source="arxiv:2401.12345v1")
    def broken(path):
        raise PdfParseError("no text; OCR disabled")
    first = FetchLiteratureTool(register, backend=backend, download=download, parse=broken).execute(state, args)
    assert not first.ok and first.value["status"] == "partial"
    assert first.value["error_type"] == "parse_failed" and len(first.artifacts) == 2
    pdf_id = first.value["pdf_artifact_id"]
    state.authorized_artifact_ids = [ref.id for ref in first.artifacts]
    def unexpected(*args):
        raise AssertionError("retry must reuse original")
    second = FetchLiteratureTool(register, backend=backend, download=unexpected, parse=parse).execute(state, args)
    assert second.ok and second.value["pdf_artifact_id"] == pdf_id


@pytest.mark.parametrize("failure", ["download", "parse"])
def test_filesystem_error_returns_already_frozen_materials_through_live_port(tmp_path, failure):
    from resagent2_contracts import ResearchRequest
    from resagent2_orchestrator import JsonRunStore, ResearchRun, ScientificArtifactRegistration

    state = _state()
    store = JsonRunStore(tmp_path / "runs")
    now = datetime.now(UTC)
    store.save(ResearchRun(
        run_id=state.run_id, request=ResearchRequest(goal="Preserve acquired materials", permissions={},
                                                   budget={"max_llm_calls": 10, "timeout_seconds": 60}),
        status="running", created_at=now, updated_at=now,
    ))
    registration = ScientificArtifactRegistration(ArtifactRegistry(tmp_path / "artifacts"), store)

    def unavailable(*args):
        raise OSError("material filesystem temporarily unavailable")

    result = FetchLiteratureTool(
        registration, backend=Backend(),
        download=unavailable if failure == "download" else download,
        parse=unavailable if failure == "parse" else parse,
    ).execute(state, FetchLiteratureInput(source="arxiv:2401.12345v1"))
    kinds = ["literature_paper"] + (["literature_pdf"] if failure == "parse" else [])
    assert not result.ok and result.value["error_type"] == failure + "_failed"
    assert result.value["error"] == "material filesystem temporarily unavailable"
    assert [ref.kind for ref in result.artifacts] == kinds
    assert result.value["paper_artifact_id"] == result.artifacts[0].id
    assert result.value["fulltext_artifact_id"] is None
    if failure == "parse":
        assert result.value["status"] == "partial"
        assert result.value["pdf_artifact_id"] == result.artifacts[1].id
    reader = RegisteredArtifactReader(result.artifacts, run_id=state.run_id)
    for ref in result.artifacts:
        reader.verify(ref.id)
    assert set(store.load(state.run_id).artifacts) == {ref.id for ref in result.artifacts}
    registration.acknowledge_registration(run_id=state.run_id, operation_key=state.current_operation_key)
    assert registration.list_artifacts(run_id=state.run_id, session_id=state.session_id) == result.artifacts
    assert store.load(state.run_id).pending_registrations == {}


def test_changed_pdf_location_is_new_metadata_snapshot_with_old_bytes_unchanged(tmp_path):
    register, state, backend = Registration(tmp_path), _state(), Backend(_paper(pdf_url=None))
    args = FetchLiteratureInput(source="arxiv:2401.12345v1")
    first = FetchLiteratureTool(register, backend=backend).execute(state, args)
    state.authorized_artifact_ids = [ref.id for ref in first.artifacts]
    original = first.artifacts[0]
    path = RegisteredArtifactReader([original], run_id=state.run_id).verified_path(original.id)
    frozen = path.read_bytes()
    backend.paper = _paper(pdf_url="https://example.org/new.pdf")
    second = FetchLiteratureTool(register, backend=backend, download=download, parse=parse).execute(state, args)
    assert second.value["paper_artifact_id"] != original.id
    assert path.read_bytes() == frozen and register.refs[original.id] == original


def test_tampered_frozen_pdf_is_rejected_even_when_text_exists(tmp_path):
    register, state = Registration(tmp_path), _state()
    result = FetchLiteratureTool(register, backend=Backend(), download=download, parse=parse).execute(
        state, FetchLiteratureInput(source="arxiv:2401.12345v1"))
    state.authorized_artifact_ids = [ref.id for ref in result.artifacts]
    pdf_id = result.value["pdf_artifact_id"]
    RegisteredArtifactReader(result.artifacts, run_id=state.run_id).verified_path(pdf_id).write_bytes(b"changed")
    with pytest.raises(ArtifactReadError, match="sha256"):
        FetchLiteratureTool(register, backend=Backend()).execute(state, FetchLiteratureInput(source="arxiv:2401.12345v1"))


def test_authorized_imported_pdf_works_offline_and_is_not_relabelled(tmp_path):
    register, state = Registration(tmp_path), _state()
    first = FetchLiteratureTool(register, backend=Backend(), download=download, parse=parse).execute(
        state, FetchLiteratureInput(source="arxiv:2401.12345v1"))
    imported = [ref.model_copy(update={"producer": AgentOwner.ORCHESTRATOR, "session_id": None,
                                      "metadata": {**ref.metadata, "source_type": "import"}})
                for ref in first.artifacts[:2]]
    register.refs.clear()
    def unexpected(*args):
        raise AssertionError("offline imported paper must not access network")
    result = FetchLiteratureTool(register, input_artifacts=imported, download=unexpected, parse=parse).execute(
        state, FetchLiteratureInput(source=f"artifact:{imported[0].id}"))
    assert result.ok and len(result.artifacts) == 1
    assert result.value["paper_artifact_id"] == imported[0].id
    assert result.value["pdf_artifact_id"] == imported[1].id
    assert len(register.refs) == 1 and result.value["reused"]["pdf"]
    assert result.value["fulltext_artifact_id"] != first.value["fulltext_artifact_id"]
    with pytest.raises(ValueError, match="authorized literature_paper"):
        FetchLiteratureTool(register, download=unexpected).execute(
            state, FetchLiteratureInput(source=f"artifact:{imported[0].id}"))


@pytest.mark.parametrize("bad_kind", ["web_page", "literature_pdf", "literature_fulltext"])
def test_artifact_source_accepts_only_authorized_paper(tmp_path, bad_kind):
    register, state = Registration(tmp_path), _state()
    ref = register.registry.register_scientific(
        ArtifactCandidate(kind=bad_kind, path="other.md", media_type="text/markdown", summary="Other", content="other"),
        run_id=state.run_id, session_id=state.session_id)
    with pytest.raises(ValueError, match="authorized literature_paper"):
        FetchLiteratureTool(register, input_artifacts=[ref]).execute(state, FetchLiteratureInput(source=f"artifact:{ref.id}"))


def test_foreign_session_snapshot_is_not_reused_without_explicit_input_grant(tmp_path):
    register, state = Registration(tmp_path), _state()
    first = FetchLiteratureTool(register, backend=Backend(), download=download, parse=parse).execute(
        state, FetchLiteratureInput(source="arxiv:2401.12345v1"))
    other = state.model_copy(update={"session_id": "session_other"})
    second = FetchLiteratureTool(register, backend=Backend(), download=download, parse=parse).execute(
        other, FetchLiteratureInput(source="arxiv:2401.12345v1"))
    assert not second.value["reused"]["paper"]
    assert second.value["paper_artifact_id"] != first.value["paper_artifact_id"]


def test_same_session_unaccepted_snapshot_is_not_implicitly_authorized(tmp_path):
    register, state = Registration(tmp_path), _state()
    first = FetchLiteratureTool(register, backend=Backend(), download=download, parse=parse).execute(
        state, FetchLiteratureInput(source="arxiv:2401.12345v1"))
    second = FetchLiteratureTool(register, backend=Backend(), download=download, parse=parse).execute(
        state, FetchLiteratureInput(source="arxiv:2401.12345v1"))
    assert state.authorized_artifact_ids == []
    assert second.value["paper_artifact_id"] != first.value["paper_artifact_id"]
    assert not second.value["reused"]["paper"]


@pytest.mark.parametrize("source", ["paper title", "https://arxiv.org/pdf/2401.12345", "arxiv:bad",
                                   "doi:not-a-doi", "openalex:X123", "artifact:", "unknown:W1"])
def test_invalid_or_unqualified_source_rejected_before_fetch(source):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FetchLiteratureInput(source=source)


def test_old_fulltext_tool_and_input_are_removed():
    import resagent2_capabilities as capabilities
    from pydantic import ValidationError
    assert not hasattr(capabilities, "FetchLiteratureFulltextTool")
    with pytest.raises(ValidationError):
        FetchLiteratureInput(paper_artifact_id="artifact_old")
