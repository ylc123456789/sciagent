"""DeepSeek's independent Search API behind the neutral search interface."""

from __future__ import annotations

import math
from urllib.parse import urlparse

from resagent2_runtime.service_request import (
    ServiceRequestClient, ServiceRequestError, ServiceRequestHTTPError,
)

from .web import (
    MAX_WEB_SNIPPET_CHARS, WebSearchError, WebSearchItem, WebSearchResult, _retry_after,
)


class DeepSeekWebSearchBackend:
    """Perform one bounded Search request, without an intermediate model call."""

    name = "deepseek"

    def __init__(self, client: ServiceRequestClient, *, count: int = 10) -> None:
        if type(count) is not int or not 1 <= count <= 50:
            raise ValueError("search count must be an integer between 1 and 50")
        self.client = client
        self.count = count

    def set_trace_context(self, **kwargs) -> None:
        self.client.set_trace_context(**{"provider": self.name, **kwargs})

    def search(self, query: str) -> WebSearchResult:
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 128:
            raise WebSearchError(
                "web query must contain 1 to 128 characters after trimming",
                error_type="invalid_query",
            )
        query = query.strip()
        try:
            payload = self.client.request({
                "query": query, "count": self.count, "need_content": False,
            })
        except ServiceRequestHTTPError as error:
            raise WebSearchError(
                f"DeepSeek search HTTP {error.status_code}; request rejected",
                error_type=("rate_limited" if error.status_code == 429
                            else "provider_error" if 500 <= error.status_code <= 599
                            else "request_rejected"),
                retry_after=_retry_after(error.retry_after) or None,
            ) from None
        except ServiceRequestError as error:
            raise WebSearchError(str(error), error_type=error.error_type) from None
        return WebSearchResult(
            provider=self.name, query=query, results=_search_items(payload),
        )


def _invalid_response() -> WebSearchError:
    # Provider-supplied prose and URLs are not reflected into failure messages.
    return WebSearchError("DeepSeek returned invalid search results", error_type="invalid_response")


def _valid_url(value: object) -> bool:
    if (not isinstance(value, str) or not value
            or any(character.isspace() or ord(character) < 32 or ord(character) == 127
                   for character in value)):
        return False
    try:
        parsed = urlparse(value)
        # Accessing port also rejects malformed and out-of-range port numbers.
        parsed.port
        return bool(parsed.scheme in {"http", "https"} and parsed.hostname
                    and parsed.username is None and parsed.password is None)
    except ValueError:
        return False


def _search_items(payload: dict) -> list[WebSearchItem]:
    """Validate native results, preserving provider order and score semantics."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise _invalid_response()

    required = {
        "rerank_score", "title", "url", "snippet", "summary", "content",
        "site_name", "site_icon", "date", "authority",
    }
    seen: set[str] = set()
    results = []
    for item in payload["results"]:
        if not isinstance(item, dict) or not required.issubset(item):
            raise _invalid_response()
        title, url = item["title"], item["url"]
        score, authority = item["rerank_score"], item["authority"]
        if (not isinstance(title, str) or not _valid_url(url)
                or any(not isinstance(item[key], str) for key in ("snippet", "summary", "content"))
                or any(item[key] is not None and not isinstance(item[key], str)
                       for key in ("date", "site_name", "site_icon"))
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not 0 <= score <= 3 or not math.isfinite(score)
                or type(authority) is not int or not -1 <= authority <= 4):
            raise _invalid_response()
        # Check every item before deduplication so a malformed duplicate cannot
        # conceal a provider contract violation.
        if url in seen:
            continue
        seen.add(url)
        snippet, summary = item["snippet"], item["summary"]
        results.append(WebSearchItem(
            title=title.strip(), url=url,
            snippet=snippet[:MAX_WEB_SNIPPET_CHARS],
            score=float(score), published_at=item["date"],
            snippet_truncated=len(snippet) > MAX_WEB_SNIPPET_CHARS,
            summary=summary[:MAX_WEB_SNIPPET_CHARS],
            summary_truncated=len(summary) > MAX_WEB_SNIPPET_CHARS,
            authority=authority, site_name=item["site_name"],
        ))
    return results
