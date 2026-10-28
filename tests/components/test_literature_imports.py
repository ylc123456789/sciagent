"""Local paper manifests preserve supplied facts and never fetch remote files."""

import json

import pytest
from pydantic import ValidationError

from resagent2_components.literature import load_literature_manifest


def manifest(path, papers):
    path.write_text(json.dumps({"papers": papers}), encoding="utf-8")
    return path


def test_manifest_resolves_pdf_relative_to_manifest_and_preserves_metadata(tmp_path, monkeypatch):
    directory = tmp_path / "papers"
    directory.mkdir()
    pdf = directory / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7\\noriginal")
    source = manifest(directory / "papers.json", [{
        "title": "A supplied paper", "doi": "https://doi.org/10.1234/Example",
        "authors": ["Ada"], "published_at": "2026-10-03",
        "abstract": "Supplied abstract", "pdf_path": "source.pdf",
    }])
    monkeypatch.chdir(tmp_path)
    prepared, = load_literature_manifest(source)
    assert prepared.pdf_path == pdf.resolve()
    assert prepared.paper.title == "A supplied paper"
    assert prepared.paper.doi == "10.1234/Example"
    assert prepared.paper.key == "doi:10.1234/example"
    assert prepared.paper.source_url == "https://doi.org/10.1234/Example"
    assert prepared.paper.abstract == "Supplied abstract"
    assert str(prepared.paper.published_at) == "2026-10-03"


def test_metadata_only_has_stable_identity_and_does_not_download(tmp_path):
    source = manifest(tmp_path / "papers.json", [{
        "title": "Metadata only", "pdf_url": "https://example.org/missing.pdf",
    }])
    first, = load_literature_manifest(source)
    second, = load_literature_manifest(source)
    assert first == second and first.pdf_path is None
    assert first.paper.paper_id.startswith("external:")
    assert first.paper.source_url.startswith("urn:resagent2:paper:")
    assert first.paper.pdf_url == "https://example.org/missing.pdf"


@pytest.mark.parametrize("papers", [
    [], [{"title": "Paper", "unknown": "mistyped"}], [{"abstract": "Missing title"}],
    [{"title": " "}], [{"title": "Paper", "authors": [""]}],
])
def test_manifest_rejects_empty_or_malformed_records(tmp_path, papers):
    with pytest.raises(ValidationError):
        load_literature_manifest(manifest(tmp_path / "papers.json", papers))


def test_manifest_rejects_missing_and_non_pdf_local_inputs(tmp_path):
    source = manifest(tmp_path / "papers.json", [{"title": "Paper", "pdf_path": "missing.pdf"}])
    with pytest.raises(FileNotFoundError):
        load_literature_manifest(source)
    bad = tmp_path / "not.pdf"
    bad.write_text("Not a PDF")
    manifest(source, [{"title": "Paper", "pdf_path": str(bad)}])
    with pytest.raises(ValueError, match="PDF signature"):
        load_literature_manifest(source)


def test_existing_arxiv_version_and_source_identity_are_preserved(tmp_path):
    prepared, = load_literature_manifest(manifest(tmp_path / "papers.json", [{
        "title": "Versioned source", "paper_id": "2401.12345v2",
        "source_url": "https://arxiv.org/abs/2401.12345v2", "doi": "10.1234/example",
    }]))
    assert prepared.paper.key == "arxiv:2401.12345v2"


def test_source_url_identity_survives_metadata_snapshot_changes(tmp_path):
    source = manifest(tmp_path / "papers.json", [{
        "title": "A source", "source_url": "https://example.org/paper",
        "abstract": "Original abstract",
    }])
    first, = load_literature_manifest(source)
    manifest(source, [{
        "title": "A source", "source_url": "https://example.org/paper",
        "abstract": "Updated abstract",
    }])
    second, = load_literature_manifest(source)
    assert first.paper.key == second.paper.key
    assert first.paper != second.paper


def test_arxiv_source_url_supplies_versioned_identity_when_id_is_omitted(tmp_path):
    prepared, = load_literature_manifest(manifest(tmp_path / "papers.json", [{
        "title": "Versioned source", "source_url": "https://arxiv.org/abs/2401.12345v2",
    }]))
    assert prepared.paper.paper_id == "2401.12345v2"
    assert prepared.paper.key == "arxiv:2401.12345v2"
