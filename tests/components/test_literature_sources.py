"""Source selection has no arXiv/OpenAlex special cases or network access."""

import pytest

from tests.components.test_literature_http import http_response

from resagent2_components import (
    LiteratureSearchError,
    LiteratureSearchResult,
    LiteratureUnavailableError,
    MultiSourceLiteratureBackend,
)
from tests.capabilities.test_literature import paper


class ScriptedSource:
    def __init__(self, name, replies, calls):
        self.name = name
        self.replies = iter(replies)
        self.calls = calls

    def search(self, query, **bounds):
        self.calls.append((self.name, query, bounds))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return LiteratureSearchResult(reply, self.name, query, bounds.get("page", 1), None)


@pytest.mark.parametrize("names", [("arxiv", "openalex"), ("openalex", "arxiv")])
def test_peers_can_switch_both_ways_and_reuse_success(names):
    calls = []
    first_result, second_result = [paper("1")], [paper("2")]
    first = ScriptedSource(names[0], [
        LiteratureUnavailableError("rate limited"), first_result,
    ], calls)
    second = ScriptedSource(names[1], [
        second_result, second_result, LiteratureUnavailableError("rate limited"),
    ], calls)
    backend = MultiSourceLiteratureBackend(first, second)
    bounds = dict(max_results=4, start_year=2020, end_year=2024)

    assert backend.search("first", **bounds).papers == second_result
    assert backend.search("next", **bounds).papers == second_result
    # The previous source can recover and be used when the current one fails.
    assert backend.search("return", **bounds).papers == first_result
    assert [(name, query) for name, query, _ in calls] == [
        (names[0], "first"), (names[1], "first"),
        (names[1], "next"),
        (names[1], "return"), (names[0], "return"),
    ]
    assert all(received == dict(bounds, source=name, scope="topic", page=1) for name, _, received in calls)


def test_all_unavailable_visits_each_source_once_and_keeps_all_errors():
    calls = []
    sources = [
        ScriptedSource(name, [LiteratureUnavailableError(name)], calls)
        for name in ("first", "second", "third")
    ]
    with pytest.raises(LiteratureUnavailableError, match="All literature sources") as caught:
        MultiSourceLiteratureBackend(*sources).search("x", max_results=1)
    assert [name for name, _, _ in calls] == ["first", "second", "third"]
    assert all(name in str(caught.value) for name in ("first", "second", "third"))


def test_initial_order_is_configurable_and_success_does_not_query_other_sources():
    calls = []
    expected = [paper("1")]
    selected = ScriptedSource("selected", [expected, expected], calls)
    untouched = ScriptedSource("untouched", [], calls)
    backend = MultiSourceLiteratureBackend(selected, untouched)
    assert backend.search("x", max_results=1).papers == expected
    assert backend.search("y", max_results=1).papers == expected
    assert [name for name, _, _ in calls] == ["selected", "selected"]


def test_valid_empty_result_after_switch_is_success_not_another_failure():
    calls = []
    first = ScriptedSource("first", [LiteratureUnavailableError("offline")], calls)
    second = ScriptedSource("second", [[], []], calls)
    backend = MultiSourceLiteratureBackend(first, second)
    assert backend.search("x", max_results=1).papers == []
    assert backend.search("y", max_results=1).papers == []
    assert [name for name, _, _ in calls] == ["first", "second", "second"]


def test_request_error_does_not_loop_back_or_try_more_sources():
    calls = []
    first = ScriptedSource("first", [LiteratureUnavailableError("offline")], calls)
    second = ScriptedSource("second", [LiteratureSearchError("HTTP 400")], calls)
    third = ScriptedSource("third", [], calls)
    with pytest.raises(LiteratureSearchError, match="HTTP 400"):
        MultiSourceLiteratureBackend(first, second, third).search("x", max_results=1)
    assert [name for name, _, _ in calls] == ["first", "second"]


def test_empty_configuration_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        MultiSourceLiteratureBackend()


def test_peer_switching_respects_each_real_backends_http_cooldown(monkeypatch):
    import json
    from tests.components.test_literature_http import http_error
    from resagent2_components.literature import (
        _http as _literature_http,
        backends as literature,
        backends as openalex,
    )
    from tests.capabilities.test_literature import ARXIV_ATOM
    from tests.components.test_openalex import work

    clock, calls = [0.0], []
    monkeypatch.setattr(_literature_http.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        _literature_http.time, "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    for module, attribute, name, interval in (
        (literature, "_ARXIV_HTTP", "arXiv", 3),
        (openalex, "_OPENALEX_HTTP", "OpenAlex", 1),
    ):
        monkeypatch.setattr(
            module, attribute,
            _literature_http.LiteratureHTTP(name, interval_seconds=interval),
        )

    def arxiv_request(self, url):
        calls.append(("arxiv", clock[0]))
        if clock[0] < 60:
            raise http_error(429)
        return http_response(ARXIV_ATOM.encode())

    def openalex_request(self, url):
        calls.append(("openalex", clock[0]))
        if clock[0] >= 10:
            raise http_error(429)
        return http_response(json.dumps({"results": [work()]}).encode())

    monkeypatch.setattr(literature.ArxivLiteratureBackend, "_request", arxiv_request)
    monkeypatch.setattr(openalex.OpenAlexLiteratureBackend, "_request", openalex_request)
    backend = MultiSourceLiteratureBackend(
        literature.ArxivLiteratureBackend(), openalex.OpenAlexLiteratureBackend(),
    )
    assert backend.search("x", max_results=3).papers[0].paper_id == "openalex:W123"
    clock[0] = 10
    with pytest.raises(LiteratureUnavailableError, match="All literature sources"):
        backend.search("x", max_results=3)
    # arXiv remains in cooldown: do not make another HTTP request to it yet.
    assert calls == [("arxiv", 0), ("openalex", 0), ("openalex", 10)]
    clock[0] = 61
    assert backend.search("x", max_results=3).papers[0].paper_id == "2301.00001v2"
    # OpenAlex remains in cooldown until t=70; arXiv has recovered.
    assert calls[-1] == ("arxiv", 61)
    assert len(calls) == 4


@pytest.fixture
def real_backends(monkeypatch):
    from resagent2_components.literature import _http, backends

    for attribute, name in (("_ARXIV_HTTP", "arXiv"), ("_OPENALEX_HTTP", "OpenAlex")):
        monkeypatch.setattr(
            backends, attribute, _http.LiteratureHTTP(name, interval_seconds=0),
        )
    return backends


def test_406_tries_real_peer_backend_and_keeps_successful_source(real_backends, monkeypatch):
    import json
    from tests.components.test_literature_http import http_error
    from tests.components.test_openalex import work

    calls = []
    def arxiv_request(self, url):
        calls.append("arxiv")
        raise http_error(406)
    def openalex_request(self, url):
        calls.append("openalex")
        return http_response(json.dumps({"results": [work()]}).encode())
    monkeypatch.setattr(real_backends.ArxivLiteratureBackend, "_request", arxiv_request)
    monkeypatch.setattr(real_backends.OpenAlexLiteratureBackend, "_request", openalex_request)
    backend = MultiSourceLiteratureBackend(
        real_backends.ArxivLiteratureBackend(), real_backends.OpenAlexLiteratureBackend(),
    )

    first = backend.search("calibration", max_results=3)
    assert first.papers[0].paper_id == "openalex:W123"
    assert first.papers[0].source_url == "https://openalex.org/W123"
    assert backend.search("next query", max_results=3).papers == first.papers
    assert calls == ["arxiv", "openalex", "openalex"]


@pytest.mark.parametrize("peer_status", [406, 429])
def test_406_and_unavailable_peer_preserve_both_causes(real_backends, monkeypatch, peer_status):
    from tests.components.test_literature_http import http_error

    calls = []
    def arxiv_request(self, url):
        calls.append("arxiv")
        raise http_error(406)
    def openalex_request(self, url):
        calls.append("openalex")
        raise http_error(peer_status)
    monkeypatch.setattr(real_backends.ArxivLiteratureBackend, "_request", arxiv_request)
    monkeypatch.setattr(real_backends.OpenAlexLiteratureBackend, "_request", openalex_request)
    backend = MultiSourceLiteratureBackend(
        real_backends.ArxivLiteratureBackend(), real_backends.OpenAlexLiteratureBackend(),
    )

    with pytest.raises(LiteratureUnavailableError, match="All literature sources") as caught:
        backend.search("calibration", max_results=3)
    assert calls == ["arxiv", "openalex"]
    assert "arXiv HTTP 406" in str(caught.value)
    assert f"OpenAlex HTTP {peer_status}" in str(caught.value)


@pytest.mark.parametrize("failure", [400, 401, 403, 404, "invalid_xml", "deadline", "bug"])
def test_other_backend_failures_do_not_switch_sources(real_backends, monkeypatch, failure):
    from resagent2_runtime.budget import DeadlineExceededError
    from tests.components.test_literature_http import http_error

    calls = []
    def arxiv_request(self, url):
        calls.append("arxiv")
        if failure == "invalid_xml":
            return http_response(b"<broken")
        if failure == "deadline":
            raise DeadlineExceededError("deadline")
        if failure == "bug":
            raise RuntimeError("bug")
        raise http_error(failure)
    def openalex_request(self, url):
        calls.append("openalex")
        return http_response(b'{"results": []}')
    monkeypatch.setattr(real_backends.ArxivLiteratureBackend, "_request", arxiv_request)
    monkeypatch.setattr(real_backends.OpenAlexLiteratureBackend, "_request", openalex_request)
    backend = MultiSourceLiteratureBackend(
        real_backends.ArxivLiteratureBackend(), real_backends.OpenAlexLiteratureBackend(),
    )

    expected = (DeadlineExceededError if failure == "deadline" else
                RuntimeError if failure == "bug" else LiteratureSearchError)
    with pytest.raises(expected) as caught:
        backend.search("calibration", max_results=3)
    assert not isinstance(caught.value, LiteratureUnavailableError)
    assert calls == ["arxiv"]


@pytest.mark.parametrize('source,page,scope', [('auto', 2, 'topic'), ('missing', 1, 'topic'), ('arxiv', 0, 'topic'), ('arxiv', 1, 'fulltext')])
def test_invalid_selection_has_no_provider_calls(source, page, scope):
    calls = []
    backend = MultiSourceLiteratureBackend(ScriptedSource('arxiv', [], calls))
    with pytest.raises(LiteratureSearchError) as caught:
        backend.search('query', max_results=2, source=source, page=page, scope=scope)
    assert caught.value.error_type == 'invalid_query'
    assert calls == []


def test_explicit_source_failure_never_switches():
    calls = []
    backend = MultiSourceLiteratureBackend(
        ScriptedSource('arxiv', [LiteratureUnavailableError('offline')], calls),
        ScriptedSource('openalex', [], calls))
    with pytest.raises(LiteratureUnavailableError) as caught:
        backend.search('q', max_results=2, source='arxiv', page=2)
    assert [call[0] for call in calls] == ['arxiv']
    assert caught.value.source_attempts[0]['source'] == 'arxiv'


def test_duplicate_source_names_rejected():
    with pytest.raises(ValueError, match='unique'):
        MultiSourceLiteratureBackend(ScriptedSource('arxiv', [], []), ScriptedSource('arxiv', [], []))


def test_fallback_keeps_rate_limit_cooldown_in_attempt():
    calls = []
    backend = MultiSourceLiteratureBackend(
        ScriptedSource('arxiv', [LiteratureUnavailableError('limited', error_type='rate_limited', retry_after=120)], calls),
        ScriptedSource('openalex', [[]], calls))
    result = backend.search('q', max_results=2)
    assert backend.name == 'auto'
    assert result.source_attempts[0]['retry_after'] == 120
    assert result.source_attempts[0]['error_type'] == 'rate_limited'
    assert result.source_attempts[1]['status'] == 'success'
