"""The Scorer protocol that every model backend implements."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llm_literature_screener.criteria import Criteria
from llm_literature_screener.record import Record

ProgressCallback = Callable[[int, int], None]
"""Called with (items done, items in total) while a backend works through a list."""


class BackendError(RuntimeError):
    """Raised when a backend cannot be created, cannot load its model, or gets no answer."""


@dataclass(frozen=True)
class ScorerInfo:
    """Description of a scorer for reports and the methods paragraph.

    Attributes:
        backend: Backend name: ``hf``, ``openai`` or ``mock``.
        model: Model identifier, for example ``Qwen/Qwen2.5-7B-Instruct``.
        description: How the model is run, for example "Hugging Face transformers on cuda".
        score_definition: What the score is, as a phrase that completes "The score was ...".
    """

    backend: str
    model: str
    description: str
    score_definition: str


@runtime_checkable
class Scorer(Protocol):
    """A model that scores records for relevance and can give a short reason.

    Scores are on the logit scale: logit(Yes) - logit(No) for the first answer token, so
    P(include) = sigmoid(score). Only the ranking and the calibrated threshold matter, which
    is why scores are kept unbounded instead of being squashed into probabilities.
    """

    @property
    def info(self) -> ScorerInfo:
        """Describe the backend and model."""
        ...

    @property
    def warnings(self) -> list[str]:
        """Problems noticed while scoring that affect how far the scores can be trusted."""
        ...

    def score(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[float]:
        """Return one score per record, in input order."""
        ...

    def explain(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        """Return one short free-text reason per record, in input order."""
        ...
