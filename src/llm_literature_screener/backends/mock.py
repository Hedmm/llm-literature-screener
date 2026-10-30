"""A deterministic keyword-based scorer for tests and demonstrations.

The mock does not use a language model. It measures which share of the content words of
the review question and inclusion criteria appear in a record's title and abstract,
subtracts a penalty for each word that occurs only in the exclusion criteria, and adds a
small fixed jitter derived from the record so that ties are rare. Like the validated prompt,
which asks the model to answer Yes when the abstract is missing, it does not hold a missing
abstract against a record. The scores are on a logit-like scale so that the rest of the
pipeline (calibration, decisions, reports) behaves as it does with a real model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

from llm_literature_screener.backends.base import ProgressCallback, ScorerInfo
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.record import Record

INCLUDE_WEIGHT = 16.0
"""Score added for a record that contains every inclusion term (scaled by the share found)."""
EXCLUDE_WEIGHT = 3.0
"""Score subtracted for each exclusion term found."""
MISSING_ABSTRACT_BONUS = 3.0
"""Score added when the abstract is missing, so that such records go to people."""
OFFSET = -5.5
JITTER = 0.5

_WORD = re.compile(r"[a-z][a-z-]+")
_STOPWORD_TEXT = """
about above after again against also among analysis another anything around because been
before being below between both cannot could does doing down during each either else every
from further have having here into itself just least less many more most much must neither
none only other over same should since some such than that their them then there these they
this those through under until upon very were what when where whether which while with
within without would your record records study studies report reports published paper
papers article articles using used based include exclude including excluding effect effects
background objective objectives method methods result results conclusion conclusions
intervention interventions participant participants outcome outcomes
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())
"""Common words that say nothing about a record's topic."""


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def terms(*texts: str) -> set[str]:
    """Return the stemmed content words (four letters or more, no stop words) of ``texts``."""
    words: set[str] = set()
    for text in texts:
        for word in _WORD.findall(text.lower()):
            if len(word) >= 4 and word not in STOPWORDS:
                words.add(_stem(word.strip("-")))
    return words


def _criteria_terms(criteria: Criteria) -> tuple[set[str], set[str]]:
    include = terms(
        criteria.question,
        criteria.population,
        criteria.intervention,
        criteria.comparator,
        *criteria.outcomes,
        *criteria.study_designs,
        *criteria.include,
    )
    exclude = terms(*criteria.exclude) - include
    return include, exclude


def _jitter(record: Record) -> float:
    digest = hashlib.sha256(f"{record.id}\n{record.title}".encode()).digest()
    return (int.from_bytes(digest[:4], "big") / 2**32 - 0.5) * 2 * JITTER


def _report(progress: ProgressCallback | None, total: int) -> None:
    if progress is not None:
        progress(total, total)


def _listing(words: Iterable[str], limit: int = 5) -> str:
    chosen = sorted(words)[:limit]
    return ", ".join(chosen)


class MockScorer:
    """Keyword-based stand-in for a language model, for tests and demos.

    The scores are reproducible across runs and machines. They only measure word overlap
    with the protocol, so they say nothing about the quality of a real model.
    """

    @property
    def info(self) -> ScorerInfo:
        """Describe the mock backend."""
        return ScorerInfo(
            backend="mock",
            model="mock-keyword-scorer",
            description="deterministic keyword overlap with the protocol (no language model)",
            score_definition=(
                "a keyword-overlap score from a mock backend used for testing and "
                "demonstration, not a language model"
            ),
        )

    @property
    def warnings(self) -> list[str]:
        """Remind every report that these scores do not come from a language model."""
        return [
            "Scores come from the mock backend (keyword overlap), not from a language model. "
            "Use it to try the tool, never to screen a real review."
        ]

    def score(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[float]:
        """Score each record by its overlap with the inclusion and exclusion terms."""
        include, exclude = _criteria_terms(criteria)
        scores = []
        for record in records:
            words = terms(record.title, record.abstract)
            score = (
                INCLUDE_WEIGHT * len(words & include) / max(1, len(include))
                - EXCLUDE_WEIGHT * len(words & exclude)
                + (0.0 if record.abstract.strip() else MISSING_ABSTRACT_BONUS)
                + OFFSET
                + _jitter(record)
            )
            scores.append(round(score, 4))
        _report(progress, len(records))
        return scores

    def explain(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        """Name the protocol terms found in each record."""
        include, exclude = _criteria_terms(criteria)
        reasons = []
        for record in records:
            words = terms(record.title, record.abstract)
            found, against = words & include, words & exclude
            parts = []
            if found:
                parts.append(f"Mentions {len(found)} protocol terms ({_listing(found)})")
            if against:
                parts.append(f"also exclusion terms ({_listing(against)})")
            reasons.append("; ".join(parts) + "." if parts else "Mentions no protocol terms.")
        _report(progress, len(records))
        return reasons
