"""Explicit literature acquisition from exact provider or authorized identities."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field, field_validator

from resagent2_components.artifacts import ArtifactRegistrationPort, RegisteredArtifactReader
from resagent2_components.literature import (
    LiteraturePaper, LiteratureSearchBackend, LiteratureSearchError,
    parse_literature_source, render_paper,
)
from resagent2_components.literature.fulltext import PdfFetchError, PdfParseError, fetch_pdf, parse_pdf
from resagent2_contracts import AgentOwner, ArtifactCandidate, ArtifactRef
from resagent2_runtime import AgentState, ToolObservation
from resagent2_runtime.models import NonEmptyStr, RuntimeModel


class FetchLiteratureInput(RuntimeModel):
    source: NonEmptyStr = Field(description=(
        "Exact source: arxiv:<ID with optional version>, doi:<DOI>, openalex:<W-ID>, "
        "or artifact:<authorized literature_paper ID>. Copy fetch_source from search."
    ))

    @field_validator("source")
    @classmethod
    def validate_source(cls, value):
        namespace, identity = parse_literature_source(value)
        return f"{namespace}:{identity}"


class FetchLiteratureTool:
    """Preserve real metadata, PDF and extracted text; never guess a paper."""

    name = "fetch_literature"
    input_model = FetchLiteratureInput
    model_guidance = (
        "Preserve a useful literature source using its exact fetch_source, or "
        "artifact:<authorized literature_paper ID> for an imported record/PDF. "
        "This resolves metadata without title guessing, freezes available original "
        "PDF bytes, then extracts page-labelled text with OCR disabled. Metadata, "
        "PDF and full text have separate artifact IDs with source links. Unavailable "
        "PDFs or extraction failures are explicit; actually obtained materials "
        "remain preserved. Unchanged authorized snapshots are reused. Read the "
        "fulltext artifact with read_artifact; acquiring it does not mean it was "
        "read or verified. Figures, formulas and table structure may be incomplete."
    )

    def __init__(self, register: ArtifactRegistrationPort, *,
                 backend: LiteratureSearchBackend | None = None,
                 input_artifacts: list[ArtifactRef] | None = None, download=None, parse=None):
        self.register = register
        self.backend = backend
        self.input_artifacts = list(input_artifacts or [])
        self.download = download or fetch_pdf
        self.parse = parse or parse_pdf

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(FetchLiteratureInput, arguments)
        registered = list({ref.id: ref for ref in [
            *self.input_artifacts, *(ref for ref in self.register.list_artifacts(
                run_id=state.run_id, session_id=state.session_id) if ref.id in state.authorized_artifact_ids),
        ] if ref.run_id == state.run_id}.values())
        reader = RegisteredArtifactReader(registered, run_id=state.run_id)
        namespace, identity = parse_literature_source(args.source)
        if namespace == "artifact":
            paper_ref = reader.resolve_ref(identity)
            if paper_ref is None or paper_ref.kind != "literature_paper":
                raise ValueError("artifact source must name an authorized literature_paper in this Run")
            reader.verify(paper_ref.id)
            paper = LiteraturePaper.model_validate(paper_ref.metadata["paper"])
            reused_paper = True
        else:
            if self.backend is None:
                return ToolObservation(ok=False, summary="No literature provider is configured for source resolution.",
                                       value={"status": "failed", "source": args.source,
                                              "error_type": "provider_unavailable", "error": "No literature provider configured"})
            try:
                paper = self.backend.resolve(args.source)
            except LiteratureSearchError as error:
                return ToolObservation(
                    ok=False, summary=str(error),
                    value={"status": "failed", "source": args.source, "error": str(error),
                           "error_type": error.error_type, "retry_after": error.retry_after,
                           "source_attempts": error.source_attempts},
                )
            data, content = paper.model_dump(mode="json"), render_paper(paper)
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            paper_ref = next((ref for ref in registered
                              if ref.kind == "literature_paper" and ref.sha256 == digest
                              and ref.metadata.get("paper_key") == paper.key
                              and ref.metadata.get("paper") == data), None)
            reused_paper = paper_ref is not None
            if paper_ref is not None:
                reader.verify(paper_ref.id)
            else:
                paper_ref = self.register.register_scientific(
                    ArtifactCandidate(kind="literature_paper", path="paper.md", media_type="text/markdown",
                                      summary=f"{paper.title} — metadata and abstract",
                                      metadata={"paper_key": paper.key, "paper": data}, content=content),
                    run_id=state.run_id, session_id=state.session_id,
                    operation_key=state.current_operation_key, output_position=0,
                )
                registered.append(paper_ref)

        refs = [ref for ref in registered if ref.metadata.get("paper_artifact_id") == paper_ref.id]
        pdf = next((ref for ref in reversed(refs) if ref.kind == "literature_pdf"), None)
        text = next((ref for ref in reversed(refs) if ref.kind == "literature_fulltext"), None)
        if text is not None:
            reader.verify(text.id)
            source = reader.resolve_ref(text.metadata.get("source_artifact_id", ""))
            if (source is None or source.kind != "literature_pdf"
                    or source.metadata.get("paper_artifact_id") != paper_ref.id
                    or source.metadata.get("source_artifact_id") != paper_ref.id):
                raise ValueError("full text has no authorized source PDF for this paper")
            reader.verify(source.id)
            return self._result(state, args.source, paper_ref, source, text,
                                reused_paper=reused_paper, reused_pdf=True, reused_text=True)

        reused_pdf = pdf is not None
        if pdf is not None and pdf.metadata.get("source_artifact_id") != paper_ref.id:
            raise ValueError("PDF has no registered source paper")
        if pdf is None:
            if not paper.pdf_url:
                return self._result(state, args.source, paper_ref, reused_paper=reused_paper,
                                    error="No source-provided PDF URL is available.", error_type="no_pdf_url")
            with tempfile.TemporaryDirectory(prefix="resagent2-paper-") as temporary:
                path = Path(temporary) / "paper.pdf"
                try:
                    self.download(paper.pdf_url, path)
                except (PdfFetchError, OSError) as error:
                    return self._result(state, args.source, paper_ref, reused_paper=reused_paper,
                                        error=str(error), error_type="download_failed")
                # Freeze before extraction so unreadable PDFs remain inspectable.
                pdf = self.register.register_scientific(
                    ArtifactCandidate(kind="literature_pdf", path="paper.pdf", media_type="application/pdf",
                                      summary=f"{paper.title} — original PDF",
                                      metadata={"paper_artifact_id": paper_ref.id,
                                                "source_artifact_id": paper_ref.id, "source_url": paper.pdf_url}),
                    run_id=state.run_id, session_id=state.session_id, content_bytes=path.read_bytes(),
                    operation_key=state.current_operation_key, output_position=1,
                )
        path = RegisteredArtifactReader([pdf], run_id=state.run_id).verified_path(pdf.id)
        try:
            parsed = self.parse(path)
        except (PdfParseError, OSError) as error:
            return self._result(state, args.source, paper_ref, pdf, reused_paper=reused_paper,
                                reused_pdf=reused_pdf, error=str(error), error_type="parse_failed")
        text = self.register.register_scientific(
            ArtifactCandidate(
                kind="literature_fulltext", path="paper_fulltext.md", media_type="text/markdown",
                summary=f"{paper.title} — extracted full text",
                metadata={"paper_artifact_id": paper_ref.id, "source_artifact_id": pdf.id,
                          "parser": "pymupdf4llm", "parser_version": parsed.parser_version,
                          "page_count": parsed.page_count, "warnings": parsed.warnings, "ocr": False},
                content=f"Paper: {paper.title}\nOriginal PDF artifact: {pdf.id}\n\n"
                        "Extracted text; figures, formulas and table structure may be incomplete.\n\n"
                        + parsed.markdown,
            ), run_id=state.run_id, session_id=state.session_id,
            operation_key=state.current_operation_key, output_position=2,
        )
        return self._result(state, args.source, paper_ref, pdf, text,
                            reused_paper=reused_paper, reused_pdf=reused_pdf)

    @staticmethod
    def _result(state, source, paper, pdf=None, text=None, *, reused_paper=False,
                reused_pdf=False, reused_text=False, error=None, error_type=None):
        refs = [ref for ref in (paper, pdf, text)
                if ref is not None and ref.producer == AgentOwner.SCIENTIFIC
                and ref.session_id == state.session_id]
        return ToolObservation(
            ok=error is None, summary=error or "Literature preserved; use read_artifact for exact content.",
            value={"status": "ready" if text else "partial" if pdf else "unavailable",
                   "source": source, "resolved_source": LiteraturePaper.model_validate(paper.metadata["paper"]).fetch_source,
                   "paper_artifact_id": paper.id, "pdf_artifact_id": pdf.id if pdf else None,
                   "fulltext_artifact_id": text.id if text else None,
                   "reused": {"paper": reused_paper, "pdf": reused_pdf, "fulltext": reused_text},
                   "error": error, "error_type": error_type,
                   "warnings": text.metadata.get("warnings", []) if text else []},
            artifacts=refs,
        )
