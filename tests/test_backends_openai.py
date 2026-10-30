"""Tests for the OpenAI-compatible backend, using a fake server (httpx.MockTransport)."""

import json
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from llm_literature_screener import Record
from llm_literature_screener.backends.base import BackendError
from llm_literature_screener.backends.openai_compat import (
    OpenAICompatScorer,
    score_from_answer,
    score_from_logprobs,
)
from llm_literature_screener.criteria import Criteria

CRITERIA = Criteria.from_dict({"question": "Is this about exercise for back pain?"})
RECORDS = [Record(id=str(i), title=f"Record {i}", abstract="Abstract.") for i in range(5)]

Handler = Callable[[httpx.Request], httpx.Response]


def chat_response(content: str, top: list[tuple[str, float]] | None) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": content}}
    if top is not None:
        choice["logprobs"] = {
            "content": [
                {
                    "token": top[0][0],
                    "logprob": top[0][1],
                    "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top],
                }
            ]
        }
    return {"choices": [choice]}


def make_scorer(handler: Handler, **options: Any) -> OpenAICompatScorer:
    return OpenAICompatScorer(
        "test-model",
        base_url="http://server.test/v1",
        transport=httpx.MockTransport(handler),
        retry_backoff=0.0,
        **options,
    )


def test_score_is_the_difference_of_log_probabilities() -> None:
    assert score_from_logprobs([("Yes", -0.2), ("No", -1.9)]) == pytest.approx(1.7)


def test_answer_variants_are_pooled() -> None:
    top = [("Yes", -1.0), (" yes", -2.0), ("No", -0.5)]
    expected = math.log(math.exp(-1.0) + math.exp(-2.0)) + 0.5
    assert score_from_logprobs(top) == pytest.approx(expected)


def test_missing_answer_is_bounded_by_the_least_likely_token() -> None:
    # "No" is not among the top tokens, so log P(No) <= -6.0 and the score is at least 5.9.
    assert score_from_logprobs([("Yes", -0.1), ("Maybe", -3.0), ("The", -6.0)]) == pytest.approx(
        5.9
    )
    # With a single returned token the leftover probability mass gives the bound.
    only_yes = score_from_logprobs([("Yes", math.log(0.99))])
    assert only_yes == pytest.approx(math.log(0.99) - math.log(0.01))


def test_neither_answer_gives_none() -> None:
    assert score_from_logprobs([("The", -0.1), ("A", -2.0)]) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Yes", 1.0), ("no.", -1.0), (" YES, it fits", 1.0), ("Maybe", None), ("", None)],
)
def test_score_from_generated_answer(text: str, expected: float | None) -> None:
    assert score_from_answer(text) == expected


def test_requests_one_token_with_logprobs_and_keeps_input_order() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://server.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        payload = json.loads(request.content)
        requests.append(payload)
        number = int(payload["messages"][1]["content"].split("Title: Record ")[1][0])
        return httpx.Response(
            200, json=chat_response("Yes", [("Yes", -0.1 * number), ("No", -1.0)])
        )

    progress: list[int] = []
    scorer = make_scorer(handler, api_key="secret", concurrency=3)
    scores = scorer.score(RECORDS, CRITERIA, progress=lambda done, total: progress.append(done))

    assert scores == pytest.approx([1.0 - 0.1 * i for i in range(5)])
    assert sorted(progress) == [1, 2, 3, 4, 5]
    payload = requests[0]
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 1
    assert payload["temperature"] == 0.0
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 20
    assert payload["messages"][0]["role"] == "system"
    assert scorer.warnings == []
    assert "log-probabilities" in scorer.info.score_definition


def test_falls_back_to_generated_answers_with_a_warning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        answer = "Yes" if "Record 1" in payload["messages"][1]["content"] else "No"
        return httpx.Response(200, json=chat_response(answer, None))

    scorer = make_scorer(handler)
    assert scorer.score(RECORDS[:3], CRITERIA) == [-1.0, 1.0, -1.0]
    (warning,) = scorer.warnings
    assert "no log-probabilities for 3 of 3 records" in warning
    assert "ranking quality" in warning
    assert "generated answer" in scorer.info.score_definition


def test_retries_without_logprobs_when_the_server_rejects_them() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "logprobs" in json.loads(request.content):
            return httpx.Response(400, json={"error": "logprobs is not supported"})
        return httpx.Response(200, json=chat_response("Yes", None))

    scorer = make_scorer(handler, concurrency=1)
    assert scorer.score(RECORDS[:2], CRITERIA) == [1.0, 1.0]
    assert "no log-probabilities" in scorer.warnings[0]


def test_unclear_answers_get_a_neutral_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response("The", [("The", -0.1), ("A", -2.5)]))

    scorer = make_scorer(handler)
    assert scorer.score(RECORDS[:2], CRITERIA) == [0.0, 0.0]
    assert "neither Yes nor No" in scorer.warnings[0]


def test_transient_errors_are_retried() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=chat_response("No", [("No", -0.1), ("Yes", -2.4)]))

    assert make_scorer(handler).score(RECORDS[:1], CRITERIA) == [pytest.approx(-2.3)]
    assert calls["count"] == 3


def test_permanent_errors_raise_a_backend_error() -> None:
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    with pytest.raises(BackendError, match="401: invalid api key"):
        make_scorer(unauthorized).score(RECORDS[:1], CRITERIA)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(BackendError, match="could not reach"):
        make_scorer(down, max_retries=1).score(RECORDS[:1], CRITERIA)

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(BackendError, match="no choices"):
        make_scorer(empty).score(RECORDS[:1], CRITERIA)


def test_explain_returns_generated_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 60
        assert "logprobs" not in payload
        return httpx.Response(200, json=chat_response(" Adults with chronic pain; RCT. ", None))

    assert make_scorer(handler).explain(RECORDS[:2], CRITERIA) == [
        "Adults with chronic pain; RCT.",
        "Adults with chronic pain; RCT.",
    ]
