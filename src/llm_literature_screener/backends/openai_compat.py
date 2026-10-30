"""Backend for any OpenAI-compatible server: vLLM, llama.cpp server, Ollama and others.

Each record is sent as a chat completion that may produce one token, with ``logprobs``
and ``top_logprobs`` requested. The score is log P(Yes) - log P(No) of the first answer
token, which equals logit(Yes) - logit(No), the same quantity as the transformers backend.
Tokens such as "Yes", " yes" and "YES" are pooled. When only one of the two answers is
among the returned top tokens, the missing one gets an upper bound on its log-probability
(the smallest returned log-probability, or the log of the probability mass left over),
which can only move a record's score towards inclusion when the missing answer is "Yes".

When the server returns no log-probabilities, the scorer falls back to the generated answer
(Yes = +1, No = -1) and records a warning: records with the same answer are then tied, so
ranking and calibration lose most of their resolution.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

import httpx

from llm_literature_screener.backends.base import BackendError, ProgressCallback, ScorerInfo
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.prompt import (
    MAX_ABSTRACT_WORDS,
    ChatMessage,
    reason_messages,
    screening_messages,
)
from llm_literature_screener.record import Record

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1"
FALLBACK_SCORES = {"yes": 1.0, "no": -1.0}
"""Scores used when the server returns no log-probabilities."""
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

T = TypeVar("T")
I = TypeVar("I")  # noqa: E741


def _logsumexp(values: Sequence[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def score_from_logprobs(top: Sequence[tuple[str, float]]) -> float | None:
    """Compute log P(Yes) - log P(No) from the most likely first tokens.

    Args:
        top: (token, log-probability) pairs for the first generated token.

    Returns:
        The score, or None when neither answer is among the tokens.
    """
    yes = [logprob for token, logprob in top if token.strip().lower() == "yes"]
    no = [logprob for token, logprob in top if token.strip().lower() == "no"]
    if not yes and not no:
        return None
    logprobs = [logprob for _, logprob in top]
    bound = min(logprobs)
    remaining = 1.0 - sum(math.exp(logprob) for logprob in logprobs)
    if remaining > 0:
        bound = min(bound, math.log(remaining))
    log_yes = _logsumexp(yes) if yes else bound
    log_no = _logsumexp(no) if no else bound
    return log_yes - log_no


def score_from_answer(text: str) -> float | None:
    """Score a generated answer: +1 for Yes, -1 for No, None for anything else."""
    words = text.strip().split()
    if not words:
        return None
    return FALLBACK_SCORES.get(words[0].strip(".,:;!\"'").lower())


def _top_logprobs(choice: dict[str, Any]) -> list[tuple[str, float]] | None:
    """Extract the first token's top log-probabilities from a chat or completion choice."""
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        pairs = [
            (str(item["token"]), float(item["logprob"]))
            for item in first.get("top_logprobs") or []
            if "token" in item and item.get("logprob") is not None
        ]
        if not pairs and "token" in first and first.get("logprob") is not None:
            pairs = [(str(first["token"]), float(first["logprob"]))]
        return pairs or None
    legacy = logprobs.get("top_logprobs")
    if isinstance(legacy, list) and legacy and isinstance(legacy[0], dict):
        return [(str(token), float(value)) for token, value in legacy[0].items()]
    return None


class OpenAICompatScorer:
    """Score records through an OpenAI-compatible chat completions endpoint.

    Args:
        model: Model name as the server knows it.
        base_url: Base URL of the API, for example ``http://localhost:8000/v1`` (vLLM),
            ``http://localhost:8080/v1`` (llama.cpp) or ``http://localhost:11434/v1`` (Ollama).
        api_key: Sent as a bearer token when given.
        top_logprobs: Number of most likely first tokens to request (most servers allow 20).
        concurrency: Number of requests in flight at once.
        timeout: Seconds to wait for one response.
        max_retries: Retries after a connection error or a 429/5xx response.
        retry_backoff: Seconds before the first retry; doubled for each further retry.
        max_abstract_words: Abstracts are cut to this many words.
        max_new_tokens: Length limit of generated reasons.
        transport: Custom httpx transport, used by the tests.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        top_logprobs: int = 20,
        concurrency: int = 8,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        max_abstract_words: int = MAX_ABSTRACT_WORDS,
        max_new_tokens: int = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not model:
            raise BackendError("The openai backend needs a model name.")
        if concurrency < 1:
            raise BackendError("concurrency must be at least 1.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.top_logprobs = top_logprobs
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_abstract_words = max_abstract_words
        self.max_new_tokens = max_new_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url + "/", headers=headers, timeout=timeout, transport=transport
        )
        self._request_logprobs = True
        self._lock = threading.Lock()
        self._scored = 0
        self._fallback = 0
        self._unclear = 0

    @property
    def info(self) -> ScorerInfo:
        """Describe the server and how scores were computed."""
        if self._scored and self._fallback == self._scored:
            definition = (
                "the model's generated answer (Yes = +1, No = -1), because the server returned "
                "no log-probabilities"
            )
        else:
            definition = (
                'the difference between the log-probabilities of "Yes" and "No" for the first '
                "answer token"
            )
            if self._fallback:
                definition += (
                    f" (for {self._fallback} records without log-probabilities, the generated "
                    "answer was used instead)"
                )
        return ScorerInfo(
            backend="openai",
            model=self.model,
            description=f"OpenAI-compatible server at {self.base_url}",
            score_definition=definition,
        )

    @property
    def warnings(self) -> list[str]:
        """Warnings about records scored from generated answers or with unclear answers."""
        messages = []
        if self._fallback:
            messages.append(
                f"The server returned no log-probabilities for {self._fallback} of "
                f"{self._scored} records, so their scores come only from the generated answer "
                "(Yes = +1, No = -1). Records with the same answer are tied, which lowers "
                "ranking quality and makes the threshold coarse. Use a server that returns "
                "logprobs (for example vLLM or llama.cpp) for proper ranking."
            )
        if self._unclear:
            messages.append(
                f"For {self._unclear} records the model's first token was neither Yes nor No; "
                "they got a neutral score of 0 (P(include) = 0.5) and stay in human screening."
            )
        return messages

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._client.close()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a chat completion request, retrying transient failures."""
        attempt = 0
        while True:
            try:
                response = self._client.post("chat/completions", json=payload)
            except httpx.TransportError as error:
                failure = f"could not reach {self.base_url}: {error}"
            else:
                if response.status_code < 400:
                    data: dict[str, Any] = response.json()
                    return data
                if response.status_code in (400, 422) and "logprobs" in payload:
                    # Some servers reject the logprobs parameters instead of ignoring them.
                    self._request_logprobs = False
                    payload = {
                        k: v for k, v in payload.items() if k not in ("logprobs", "top_logprobs")
                    }
                    continue
                failure = f"{self.base_url} answered {response.status_code}: {response.text[:300]}"
                if response.status_code not in RETRY_STATUS:
                    raise BackendError(failure)
            if attempt >= self.max_retries:
                raise BackendError(failure)
            time.sleep(self.retry_backoff * 2**attempt)
            attempt += 1

    def _score_one(self, messages: list[ChatMessage]) -> float:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
        }
        if self._request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs
        choice = self._first_choice(self._post(payload))
        top = _top_logprobs(choice)
        if top:
            score = score_from_logprobs(top)
            with self._lock:
                self._scored += 1
                if score is None:
                    self._unclear += 1
            return 0.0 if score is None else score
        answer = score_from_answer(self._text(choice))
        with self._lock:
            self._scored += 1
            self._fallback += 1
            first_fallback = self._fallback == 1
            if answer is None:
                self._unclear += 1
        if first_fallback:
            logger.warning(
                "The server returned no log-probabilities; scoring from generated answers. "
                "Ranking quality drops."
            )
        return 0.0 if answer is None else answer

    def _explain_one(self, messages: list[ChatMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
        }
        return self._text(self._first_choice(self._post(payload))).strip()

    @staticmethod
    def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise BackendError(f"The server's answer has no choices: {str(data)[:300]}")
        choice: dict[str, Any] = choices[0]
        return choice

    @staticmethod
    def _text(choice: dict[str, Any]) -> str:
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return str(message["content"])
        return str(choice.get("text") or "")

    def _map(
        self,
        function: Callable[[I], T],
        items: Sequence[I],
        progress: ProgressCallback | None,
    ) -> list[T]:
        """Apply ``function`` to every item with a thread pool, keeping the input order."""
        results: dict[int, T] = {}
        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        try:
            futures: dict[Future[T], int] = {
                pool.submit(function, item): index for index, item in enumerate(items)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                if progress is not None:
                    progress(len(results), len(items))
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
        return [results[index] for index in range(len(items))]

    def score(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[float]:
        """Return log P(Yes) - log P(No) for each record (see the module docstring)."""
        prompts = [screening_messages(criteria, r, self.max_abstract_words) for r in records]
        return self._map(self._score_one, prompts, progress)

    def explain(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        """Generate a short reason for each record."""
        prompts = [reason_messages(criteria, r, self.max_abstract_words) for r in records]
        return self._map(self._explain_one, prompts, progress)
