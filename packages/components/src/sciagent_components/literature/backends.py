"""Normalized bibliographic records, peer backends and artifact presentation."""

from __future__ import annotations

import json
import logging
import math
import re
import shlex
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Protocol
from urllib.parse import quote, urlencode
import httpx

from defusedxml import ElementTree

from sciagent_runtime.http import send_request
from .records import LiteraturePaper, parse_literature_source
from ._http import (
    METADATA_RESPONSE_MAX_BYTES, USER_AGENT, LiteratureHTTP, LiteratureSearchError,
    LiteratureUnavailableError,
)

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_HTTP = LiteratureHTTP("arXiv", interval_seconds=3.0)
_OPENALEX_HTTP = LiteratureHTTP("OpenAlex", interval_seconds=1.0)


@dataclass
class LiteratureSearchResult:
    """One provider page; an absent total does not prove exhaustion."""

    papers: list[LiteraturePaper]
    source: str
    executed_query: str
    page: int
    next_page: int | None
    source_attempts: list[dict] = field(default_factory=list)
    total_results: int | None = None


def _validate_search(query, max_results, source, scope, page, name=None, start_year=None, end_year=None):
    if (not isinstance(query, str) or not query.strip() or
            type(max_results) is not int or max_results < 1 or
            type(page) is not int or page < 1 or scope not in ('topic', 'title') or
            (source == 'auto' and page != 1) or
            (name is not None and source not in ('auto', name)) or
            any(value is not None and (type(value) is not int or not 1 <= value <= 9999)
                for value in (start_year, end_year)) or
            (start_year is not None and end_year is not None and start_year > end_year)):
        raise LiteratureSearchError('Invalid literature search query, scope, source or page',
                                    error_type='invalid_query')
    _keywords(query)


def _keywords(query):
    lexer = shlex.shlex(query, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.quotes = '"'
    try:
        keywords = list(lexer)
    except ValueError:
        raise LiteratureSearchError(
            "Use keywords and balanced double-quoted phrases for literature search", error_type="invalid_query"
        ) from None
    if not keywords or any(not word.strip() for word in keywords):
        raise LiteratureSearchError("Literature query must contain non-empty keywords", error_type="invalid_query")
    return keywords


def _title_phrase(query: str) -> str:
    """Treat parsed title words as one phrase instead of separate AND terms."""
    title = " ".join(" ".join(_keywords(query)).split())
    escaped = title.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'


def _attempt(source, error=None):
    attempt = dict(source=source, status='failed' if error else 'success',
                   error_type=error.error_type if error else None,
                   error=str(error) if error else None)
    if error is not None and error.retry_after is not None:
        attempt['retry_after'] = error.retry_after
    return attempt


def _result(papers, source, executed_query, page, max_results, total, raw_count):
    next_page = (page + 1 if (total is not None and page * max_results < total)
                 or (total is None and raw_count >= max_results) else None)
    return LiteratureSearchResult(papers, source, executed_query, page, next_page,
                                  [_attempt(source)], total)


class LiteratureSearchBackend(Protocol):
    """Provider-neutral literature search returning normalized papers."""

    name: str

    def search(
        self,
        query: str,
        *,
        max_results: int,
        source: str = 'auto',
        scope: str = 'topic',
        page: int = 1,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> LiteratureSearchResult:
        """Return one normalized provider page and execution facts."""

    def resolve(self, source: str) -> LiteraturePaper:
        """Resolve exactly one namespaced provider identity for fetch."""


class ArxivLiteratureBackend:
    """Query the arXiv API and normalize Atom entries into LiteraturePaper.

    Requests are serialized and spaced by at least three seconds in-process.
    Rate limits trigger cooldown immediately; transient errors have bounded
    retries. Failures never become empty search results.
    """

    name = "arxiv"
    _endpoint = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries  # Historical name: total HTTP attempts.

    def search(
        self,
        query: str,
        *,
        max_results: int,
        source: str = 'auto',
        scope: str = 'topic',
        page: int = 1,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> LiteratureSearchResult:
        _validate_search(query, max_results, source, scope, page, self.name, start_year, end_year)
        executed_query = self._search_query(query, start_year, end_year, scope)
        params = {'search_query': executed_query, 'start': (page - 1) * max_results,
                  'max_results': max_results, 'sortBy': 'relevance', 'sortOrder': 'descending'}
        try:
            body = self._fetch(f"{self._endpoint}?{urlencode(params)}")
            papers = self._parse(body)
            root = ElementTree.fromstring(body)
            raw_total = root.findtext('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
            total = int(raw_total) if raw_total is not None else None
            if total is not None and total < 0:
                raise ValueError('negative total')
            return _result(papers, self.name, executed_query, page, max_results, total,
                           len(root.findall(f"{_ATOM}entry")))
        except LiteratureSearchError as error:
            error.source_attempts = [_attempt(self.name, error)]
            raise
        except ValueError as error:
            failure = LiteratureSearchError('arXiv returned an invalid result count')
            failure.source_attempts = [_attempt(self.name, failure)]
            raise failure from error

    @staticmethod
    def _search_query(
        query: str, start_year: int | None, end_year: int | None, scope: str = 'topic'
    ) -> str:
        """Translate plain keywords and double-quoted phrases to arXiv syntax."""
        keywords = _keywords(query)
        escaped = [word.replace("\\", "\\\\").replace('"', '\\"') for word in keywords]
        terms = (['ti:' + _title_phrase(query)] if scope == 'title' else
                 ['(ti:"' + word + '" OR abs:"' + word + '")' for word in escaped])
        if start_year is not None and end_year is not None:
            terms.append(
                f"submittedDate:[{start_year}01010000 TO {end_year}12312359]"
            )
        elif start_year is not None:
            terms.append(f"submittedDate:[{start_year}01010000 TO 999912312359]")
        elif end_year is not None:
            terms.append(f"submittedDate:[000001010000 TO {end_year}12312359]")
        return " AND ".join(terms)

    def _request(self, url: str) -> httpx.Response:
        """One raw HTTP request; overridable in tests to avoid the network."""
        return send_request(httpx.Request("GET", url, headers={"User-Agent": USER_AGENT}),
                            timeout=self.timeout_seconds,
                            max_response_bytes=METADATA_RESPONSE_MAX_BYTES)

    def resolve(self, source: str) -> LiteraturePaper:
        namespace, identity = parse_literature_source(source)
        if namespace != self.name:
            raise LiteratureSearchError("arXiv resolution requires arxiv:<ID>", error_type="invalid_query")
        try:
            papers = self._parse(self._fetch(f"{self._endpoint}?{urlencode({'id_list': identity})}"))
            if len(papers) != 1:
                raise LiteratureSearchError("arXiv did not return one exact paper", error_type="not_found")
            paper = papers[0]
            returned = paper.paper_id
            # An unversioned identity must resolve to an explicit actual version.
            valid = returned == identity if re.search(r"v[1-9][0-9]*$", identity) else (
                re.fullmatch(re.escape(identity) + r"v[1-9][0-9]*", returned) is not None)
            if not valid:
                raise LiteratureSearchError("arXiv returned a different paper/version", error_type="identity_mismatch")
            return paper
        except LiteratureSearchError as error:
            error.source_attempts = [_attempt(self.name, error)]
            raise

    def _fetch(self, url: str) -> bytes:
        return _ARXIV_HTTP.fetch(
            lambda: self._request(url), max_attempts=self.max_retries
        )

    def _parse(self, body: bytes) -> list[LiteraturePaper]:
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as error:
            raise LiteratureSearchError(f"arXiv returned invalid XML: {error}") from error
        if root.tag != f"{_ATOM}feed":
            raise LiteratureSearchError("arXiv returned a non-Atom response")

        papers: list[LiteraturePaper] = []
        seen: set[str] = set()
        for entry in root.findall(f"{_ATOM}entry"):
            if "/abs/" not in self._text(entry, "id"):
                raise LiteratureSearchError("arXiv returned an error or invalid entry")
            try:
                paper = self._paper(entry)
            except ValueError as error:
                raise LiteratureSearchError("arXiv returned an invalid paper") from error
            if paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            papers.append(paper)
        return papers

    def _paper(self, entry: ElementTree.Element) -> LiteraturePaper:
        id_url = self._text(entry, "id")
        paper_id = self._paper_id(id_url)
        parse_literature_source("arxiv:" + paper_id)
        abstract = self._text(entry, "summary")
        abstract = " ".join(abstract.split())
        authors = [
            name.text.strip()
            for author in entry.findall(f"{_ATOM}author")
            for name in author.findall(f"{_ATOM}name")
            if name.text and name.text.strip()
        ]
        published = self._text(entry, "published")
        published_at = None
        if published:
            try:
                published_at = date.fromisoformat(published[:10])
            except ValueError:
                published_at = None
        return LiteraturePaper(
            paper_id=paper_id,
            title=" ".join(self._text(entry, "title").split()),
            authors=authors,
            published_at=published_at,
            abstract=abstract,
            source_url=f"https://arxiv.org/abs/{paper_id}",
            pdf_url=f"https://arxiv.org/pdf/{paper_id}",
            doi=entry.findtext("{http://arxiv.org/schemas/atom}doi"),
        )

    @staticmethod
    def _text(entry: ElementTree.Element, tag: str) -> str:
        node = entry.find(f"{_ATOM}{tag}")
        return node.text or "" if node is not None else ""

    @staticmethod
    def _paper_id(id_url: str) -> str:
        """Preserve the provider-supplied arXiv identifier, including its version."""
        return id_url.split("/abs/", 1)[-1]


class MultiSourceLiteratureBackend:
    """Peer providers with sticky bounded failover only for auto first pages."""

    name = "auto"

    def __init__(self, *backends: LiteratureSearchBackend) -> None:
        if not backends:
            raise ValueError('at least one literature backend is required')
        self.backends = backends
        self._sources = {backend.name: backend for backend in backends}
        if len(self._sources) != len(backends) or 'auto' in self._sources:
            raise ValueError('literature backend names must be unique and not auto')
        self._current_index = 0

    def search(self, query: str, *, max_results: int, source: str = 'auto',
               scope: str = 'topic', page: int = 1, start_year: int | None = None,
               end_year: int | None = None) -> LiteratureSearchResult:
        _validate_search(query, max_results, source, scope, page, start_year=start_year, end_year=end_year)
        if source != 'auto' and source not in self._sources:
            raise LiteratureSearchError('Unknown literature source', error_type='invalid_query')
        bounds = dict(max_results=max_results, scope=scope, page=page,
                      start_year=start_year, end_year=end_year)
        attempts = []
        indices = ([list(self._sources).index(source)] if source != 'auto' else
                   [(self._current_index + offset) % len(self.backends)
                    for offset in range(len(self.backends))])
        for index in indices:
            backend = self.backends[index]
            try:
                result = backend.search(query, source=backend.name, **bounds)
            except LiteratureSearchError as error:
                attempts.extend(error.source_attempts or [_attempt(backend.name, error)])
                error.source_attempts = list(attempts)
                if source != 'auto' or not isinstance(error, LiteratureUnavailableError):
                    raise
                logging.getLogger(__name__).warning('Literature source unavailable (%s): %s', backend.name, error)
                continue
            attempts.extend(result.source_attempts or [_attempt(backend.name)])
            result.source_attempts = attempts
            if source == 'auto':
                self._current_index = index
            return result
        raise LiteratureUnavailableError(
            'All literature sources unavailable: ' + '; '.join(item['error'] for item in attempts),
            source_attempts=attempts)

    def resolve(self, source: str) -> LiteraturePaper:
        namespace, _ = parse_literature_source(source)
        provider = "openalex" if namespace == "doi" else namespace
        backend = self._sources.get(provider)
        if backend is None:
            raise LiteratureSearchError("No configured provider resolves this source", error_type="invalid_query")
        return backend.resolve(source)


class OpenAlexLiteratureBackend:
    """Normalize indexed records/abstracts, without fetching or inventing full text."""

    name = "openalex"
    _endpoint = "https://api.openalex.org/works"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries  # Total HTTP attempts, as in arXiv backend.

    def search(
        self,
        query: str,
        *,
        max_results: int,
        source: str = 'auto',
        scope: str = 'topic',
        page: int = 1,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> LiteratureSearchResult:
        _validate_search(query, max_results, source, scope, page, self.name, start_year, end_year)
        if max_results > 200 or page * max_results > 10_000:
            raise LiteratureSearchError('OpenAlex supports at most 200 records per page and 10000 records by page', error_type='invalid_query')
        params = {'per_page': max_results, 'page': page,
                  'select': 'id,doi,display_name,publication_date,authorships,abstract_inverted_index,best_oa_location'}
        filters = []
        executed_query = _title_phrase(query) if scope == 'title' else query
        params['search.title' if scope == 'title' else 'search.title_and_abstract'] = executed_query
        if start_year is not None:
            filters.append(f'from_publication_date:{start_year}-01-01')
        if end_year is not None:
            filters.append(f'to_publication_date:{end_year}-12-31')
        if filters:
            params['filter'] = ','.join(filters)
        try:
            body = _OPENALEX_HTTP.fetch(lambda: self._request(f"{self._endpoint}?{urlencode(params)}"),
                                      max_attempts=self.max_retries, quota_cooldown=self._quota_cooldown)
            papers = self._parse(body)
            response = json.loads(body)
            total = response.get('meta', {}).get('count')
            if total is not None and (type(total) is not int or total < 0):
                raise LiteratureSearchError('OpenAlex returned an invalid result count')
            result = _result(papers, self.name, executed_query, page, max_results, total,
                             len(response['results']))
            if result.next_page is not None and result.next_page * max_results > 10_000:
                result.next_page = None
            return result
        except LiteratureSearchError as error:
            error.source_attempts = [_attempt(self.name, error)]
            raise
        except (AttributeError, TypeError) as error:
            failure = LiteratureSearchError('OpenAlex returned invalid result metadata')
            failure.source_attempts = [_attempt(self.name, failure)]
            raise failure from error

    def _request(self, url: str) -> httpx.Response:
        headers = {"User-Agent": USER_AGENT}
        if self._api_key:
            # Never place credentials in URLs, artifacts, or model context.
            headers["Authorization"] = f"Bearer {self._api_key}"
        return send_request(httpx.Request("GET", url, headers=headers),
                            timeout=self.timeout_seconds,
                            max_response_bytes=METADATA_RESPONSE_MAX_BYTES)

    def resolve(self, source: str) -> LiteraturePaper:
        namespace, identity = parse_literature_source(source)
        if namespace not in {"doi", "openalex"}:
            raise LiteratureSearchError("OpenAlex resolution requires doi:<DOI> or openalex:<W-ID>",
                                        error_type="invalid_query")
        lookup = "https://doi.org/" + identity if namespace == "doi" else identity
        try:
            body = _OPENALEX_HTTP.fetch(
                lambda: self._request(f"{self._endpoint}/{quote(lookup, safe=':/')}"),
                max_attempts=self.max_retries, quota_cooldown=self._quota_cooldown,
            )
            try:
                paper = self._paper(json.loads(body))
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise LiteratureSearchError("OpenAlex returned invalid bibliographic data") from error
            returned = (paper.doi.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
                        if namespace == "doi" and paper.doi else paper.paper_id.removeprefix("openalex:"))
            if returned != identity:
                raise LiteratureSearchError("OpenAlex returned a different paper identity", error_type="identity_mismatch")
            return paper
        except LiteratureSearchError as error:
            error.source_attempts = [_attempt(self.name, error)]
            raise

    @staticmethod
    def _quota_cooldown(response: httpx.Response) -> float:
        """Only an explicit zero remaining balance proves quota exhaustion."""
        remaining = response.headers.get("X-RateLimit-Remaining", "")
        reset = response.headers.get("X-RateLimit-Reset", "")
        if len(remaining) > 64 or len(reset) > 64:
            return 0.0
        try:
            balance, seconds = Decimal(remaining), float(reset)
        except (InvalidOperation, ValueError):
            return 0.0
        if balance.is_zero() and math.isfinite(seconds) and seconds > 0:
            return seconds  # OpenAlex documents seconds until midnight UTC.
        return 0.0

    def _parse(self, body: bytes) -> list[LiteraturePaper]:
        try:
            response = json.loads(body)
            if not isinstance(response, dict) or not isinstance(
                response.get("results"), list
            ):
                raise ValueError("missing results list")
            papers = {}
            for record in response["results"]:
                paper = self._paper(record)
                papers.setdefault(paper.paper_id, paper)
            return list(papers.values())
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise LiteratureSearchError(
                "OpenAlex returned invalid bibliographic data"
            ) from error

    def _paper(self, record: dict) -> LiteraturePaper:
        source_url = record["id"]
        prefix = "https://openalex.org/W"
        if not isinstance(source_url, str) or not source_url.startswith(prefix):
            raise ValueError("invalid OpenAlex work id")
        if not source_url[len(prefix):].isdigit():
            raise ValueError("invalid OpenAlex work id")
        published = record.get("publication_date")
        return LiteraturePaper(
            paper_id=f"openalex:{source_url.rsplit('/', 1)[-1]}",
            title=record["display_name"],
            authors=[
                entry["author"]["display_name"]
                for entry in (record.get("authorships") or [])
                if entry["author"].get("display_name")
            ],
            published_at=date.fromisoformat(published) if published else None,
            abstract=self._abstract(record.get("abstract_inverted_index")),
            source_url=source_url,
            doi=record.get("doi"),
            pdf_url=(record.get("best_oa_location") or {}).get("pdf_url"),
        )

    def _abstract(self, index: dict[str, list[int]] | None) -> str:
        if index is None:
            return ""
        words = {}
        for word, positions in index.items():
            if not isinstance(word, str) or not isinstance(positions, list):
                raise ValueError("invalid abstract index")
            for position in positions:
                if type(position) is not int or position < 0 or position in words:
                    raise ValueError("invalid abstract position")
                words[position] = word
        abstract = " ".join(words[position] for position in sorted(words))
        return abstract
