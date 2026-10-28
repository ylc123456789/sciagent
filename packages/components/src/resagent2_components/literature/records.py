"""One bibliographic source and its deterministic, readable presentation."""

from __future__ import annotations

from datetime import date
import re
from urllib.parse import urlsplit

from pydantic import Field

from resagent2_runtime.models import NonEmptyStr, RuntimeModel
from ..text import wrap_text_lines


class LiteraturePaper(RuntimeModel):
    """Metadata and an abstract, never a claim to have retrieved full text."""

    paper_id: NonEmptyStr
    title: NonEmptyStr
    authors: list[NonEmptyStr] = Field(default_factory=list)
    published_at: date | None = None
    abstract: str = ""
    source_url: NonEmptyStr
    doi: str | None = None
    pdf_url: str | None = None

    @property
    def key(self) -> str:
        # An explicit arXiv version must not collapse into another version/DOI.
        if urlsplit(self.source_url).hostname in {"arxiv.org", "export.arxiv.org"}:
            return "arxiv:" + self.paper_id.removeprefix("arxiv:")
        if self.doi:
            return "doi:" + self.doi.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        return self.paper_id

    @property
    def fetch_source(self) -> str:
        """Prefer the precise provider record identity over a DOI alias."""
        if urlsplit(self.source_url).hostname in {"arxiv.org", "export.arxiv.org"}:
            return "arxiv:" + self.paper_id.removeprefix("arxiv:")
        if self.paper_id.startswith("openalex:"):
            return self.paper_id
        return self.key


def parse_literature_source(source: str) -> tuple[str, str]:
    """Validate one explicit provider identity, never guess from a title or URL."""
    if not isinstance(source, str):
        raise ValueError("source must be a namespaced literature identifier")
    namespace, separator, identity = source.strip().partition(":")
    patterns = {
        "arxiv": r"(?:[0-9]{4}\.[0-9]{4,5}|[a-z-]+(?:\.[A-Z]{2})?/[0-9]{7})(?:v[1-9][0-9]*)?",
        "doi": r"10\.[0-9]{4,9}/\S+",
        "openalex": r"W[0-9]+",
        "artifact": r"artifact_[A-Za-z0-9_.-]+",
    }
    if not separator or namespace not in patterns or not re.fullmatch(patterns[namespace], identity):
        raise ValueError("Use arxiv:<ID>, doi:<DOI>, openalex:<W-ID>, or artifact:<authorized paper ID>")
    return namespace, identity.lower() if namespace == "doi" else identity


def render_paper(paper: LiteraturePaper) -> str:
    """Preserve supplied source text; this is not an LLM-generated summary."""
    return wrap_text_lines(
        f"# {paper.title}\n\n"
        "Bibliographic record and provided abstract, not paper full text.\n"
        "Abstracts support only abstract-level claims.\n\n"
        f"Paper ID: {paper.paper_id}\n"
        f"Source: {paper.source_url}\n"
        f"DOI: {paper.doi or '(not supplied)'}\n"
        f"Published: {paper.published_at or '(not supplied)'}\n"
        f"Authors: {', '.join(paper.authors) or '(not supplied)'}\n"
        f"PDF URL: {paper.pdf_url or '(not supplied)'}\n\n"
        f"## Provided abstract\n\n{paper.abstract or '(not supplied)'}"
    ) + "\n"
