"""Bibliographic leads returned directly; only fetch preserves materials."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, Field, model_validator

from resagent2_runtime import AgentState, ToolObservation
from resagent2_runtime.models import NonEmptyStr, RuntimeModel
from resagent2_components.literature import LiteratureSearchBackend, LiteratureSearchError


class LiteratureSearchToolInput(RuntimeModel):
    query: NonEmptyStr = Field(description=(
        "Plain keywords or double-quoted phrases; do not use provider-specific "
        "field prefixes or Boolean operators. Escape literal double quotes and backslashes."
    ))
    source: Literal["auto", "arxiv", "openalex"] = "auto"
    scope: Literal["topic", "title"] = Field(
        default="topic",
        description=("topic searches title/abstract keywords or phrases; title treats "
                     "the complete query as one title phrase, with or without outer "
                     "double quotes. Check returned titles, authors and identifiers; "
                     "matching is not guaranteed exact or unique."),
    )
    page: int = Field(default=1, ge=1)
    max_results: int = Field(default=5, ge=1, le=20)
    start_year: int | None = Field(default=None, ge=1900, le=2100)
    end_year: int | None = Field(default=None, ge=1900, le=2100)

    @model_validator(mode="after")
    def validate_search_bounds(self):
        if self.start_year is not None and self.end_year is not None and self.start_year > self.end_year:
            raise ValueError("start_year must not exceed end_year")
        if self.source == "auto" and self.page > 1:
            raise ValueError("pagination requires an explicit source")
        return self


class LiteratureSearchTool:
    """Return one provider page without registering papers or search receipts."""

    name = "literature_search"
    input_model = LiteratureSearchToolInput
    model_guidance = (
        "Use short keywords or quoted phrases for scope='topic'; scope='title' "
        "searches one title phrase. Verify returned titles, authors and identifiers. "
        "Results are metadata/abstract leads, not full text or saved artifacts. "
        "They may disappear from model context after history compaction. Use "
        "fetch_literature with the returned fetch_source to preserve useful material, "
        "then read_artifact for exact content. Start with a small max_results. "
        "source='auto' chooses an available provider on page 1; use an explicit "
        "source to switch or paginate. next_request preserves provider and filters. "
        "OpenAlex numbered pages cover at most 10000 records; a missing next_request "
        "does not establish exhaustion when the total is unknown or the page limit "
        "is reached. Empty or failed searches do not prove absence."
    )

    def __init__(self, backend: LiteratureSearchBackend) -> None:
        self.backend = backend

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(LiteratureSearchToolInput, arguments)
        try:
            result = self.backend.search(
                args.query, max_results=args.max_results, start_year=args.start_year,
                end_year=args.end_year, source=args.source, scope=args.scope, page=args.page,
            )
        except LiteratureSearchError as error:
            return ToolObservation(
                ok=False, summary=str(error),
                value={"query": args.query, "requested_source": args.source,
                       "source": None, "executed_query": None, "page": args.page,
                       "papers": [], "status": "failed", "total_results": None,
                       "source_attempts": error.source_attempts, "next_request": None,
                       "error": str(error), "error_type": error.error_type,
                       "retry_after": error.retry_after},
            )
        seen, papers = set(), []
        for paper in result.papers:
            if paper.key in seen:
                continue
            seen.add(paper.key)
            papers.append({**paper.model_dump(mode="json"), "fetch_source": paper.fetch_source})
        next_request = ({**args.model_dump(mode="json"), "source": result.source,
                         "page": result.next_page} if result.next_page is not None else None)
        return ToolObservation(
            summary=f"Found {len(papers)} literature leads for {args.query!r}",
            value={"query": args.query, "requested_source": args.source,
                   "source": result.source, "executed_query": result.executed_query,
                   "page": result.page, "papers": papers,
                   "status": "results" if papers else "empty", "total_results": result.total_results,
                   "source_attempts": result.source_attempts, "next_request": next_request,
                   "error": None, "error_type": None, "retry_after": None},
        )
