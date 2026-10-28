"""Local, caller-supplied paper records; no registration or network access."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field

from resagent2_runtime.models import NonEmptyStr, RuntimeModel

from .records import LiteraturePaper


class LiteratureImportEntry(RuntimeModel):
    """One paper in a caller-owned JSON manifest."""

    title: NonEmptyStr
    paper_id: NonEmptyStr | None = None
    source_url: NonEmptyStr | None = None
    doi: NonEmptyStr | None = None
    authors: list[NonEmptyStr] = Field(default_factory=list)
    published_at: date | None = None
    abstract: str = ""
    pdf_url: NonEmptyStr | None = None
    pdf_path: NonEmptyStr | None = None


class LiteratureImportManifest(RuntimeModel):
    papers: list[LiteratureImportEntry] = Field(min_length=1)


class PreparedLiteratureImport(RuntimeModel):
    """Normalized source facts and an optional local PDF for the controller."""

    paper: LiteraturePaper
    pdf_path: Path | None = None


def load_literature_manifest(path: str | Path) -> list[PreparedLiteratureImport]:
    """Normalize a local manifest, resolving PDFs relative to its directory.

    The user supplies these paths, not an LLM. No metadata is inferred from the
    PDF, and no remote resource is fetched. Registration freezes the actual bytes.
    """
    source = Path(path).resolve(strict=True)
    manifest = LiteratureImportManifest.model_validate_json(source.read_text(encoding="utf-8"))
    prepared = []
    for entry in manifest.papers:
        pdf = None
        if entry.pdf_path is not None:
            pdf = (source.parent / entry.pdf_path).resolve(strict=True)
            if not pdf.is_file():
                raise ValueError(f"PDF path is not a regular file: {pdf}")
            with pdf.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise ValueError(f"PDF signature is missing: {pdf}")
        doi = entry.doi
        if doi:
            doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/").strip()
            if not doi:
                raise ValueError("DOI must not be empty after normalization")
        supplied = entry.model_dump(mode="json", exclude={"pdf_path"}, exclude_none=True)
        token = hashlib.sha256(json.dumps(supplied, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
        source_url = entry.source_url or (
            f"https://doi.org/{doi}" if doi else f"urn:resagent2:paper:{token}"
        )
        paper_id = entry.paper_id
        if paper_id is None:
            parts = urlsplit(source_url)
            if parts.hostname in {"arxiv.org", "export.arxiv.org"} and "/abs/" in parts.path:
                paper_id = parts.path.split("/abs/", 1)[1]
            elif doi:
                paper_id = "doi:" + doi.lower()
            elif entry.source_url:
                paper_id = "external:" + hashlib.sha256(source_url.encode()).hexdigest()[:24]
            else:
                paper_id = "external:" + token
        paper = LiteraturePaper(
            paper_id=paper_id, title=entry.title, source_url=source_url,
            authors=entry.authors, published_at=entry.published_at,
            abstract=entry.abstract, doi=doi, pdf_url=entry.pdf_url,
        )
        prepared.append(PreparedLiteratureImport(paper=paper, pdf_path=pdf))
    return prepared
