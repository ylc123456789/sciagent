"""Bibliographic search operations; no model tool entry points."""

from ._http import (
    LiteratureSearchError,
    LiteratureUnavailableError,
)
from .backends import (
    ArxivLiteratureBackend,
    OpenAlexLiteratureBackend,
    MultiSourceLiteratureBackend,
    LiteraturePaper,
    LiteratureSearchBackend,
    LiteratureSearchResult,
)

from .records import parse_literature_source, render_paper

from .imports import PreparedLiteratureImport, load_literature_manifest
