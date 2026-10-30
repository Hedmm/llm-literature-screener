"""Tests for the optional one-line reasons."""

from collections.abc import Sequence

import pytest

from llm_literature_screener import Record
from llm_literature_screener.backends.base import ProgressCallback, ScorerInfo
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.reasons import clean_reason, generate_reasons, records_needing_reasons

CRITERIA = Criteria.from_dict({"question": "Q?"})


class RecordingScorer:
    """A scorer that records which records it was asked to explain."""

    def __init__(self) -> None:
        self.explained: list[str] = []

    @property
    def info(self) -> ScorerInfo:
        return ScorerInfo("test", "test", "test", "test")

    @property
    def warnings(self) -> list[str]:
        return []

    def score(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[float]:
        return [0.0] * len(records)

    def explain(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        self.explained.extend(record.id for record in records)
        return [f"Reason: Record {record.id} fits.\nSecond line." for record in records]


def test_reasons_are_generated_only_at_or_above_the_threshold() -> None:
    records = [Record(id=str(i)) for i in range(4)]
    scorer = RecordingScorer()
    reasons = generate_reasons(records, CRITERIA, scorer, [2.0, -9.0, -3.0, -5.5], threshold=-5.5)
    assert scorer.explained == ["0", "2", "3"]
    assert reasons == ["Record 0 fits.", None, "Record 2 fits.", "Record 3 fits."]


def test_no_model_call_when_nothing_is_above_the_threshold() -> None:
    scorer = RecordingScorer()
    assert generate_reasons([Record(id="1")], CRITERIA, scorer, [-10.0], threshold=0.0) == [None]
    assert scorer.explained == []


def test_records_needing_reasons() -> None:
    assert records_needing_reasons([1.0, -1.0, 0.0], threshold=0.0) == [0, 2]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reason: Adults with chronic pain.", "Adults with chronic pain."),
        ('  "RCT in adults."  ', "RCT in adults."),
        ("\n\n- Meets the population criterion.\nMore text.", "Meets the population criterion."),
        ("Answer:   two   spaces", "two spaces"),
        ("", ""),
    ],
)
def test_clean_reason(text: str, expected: str) -> None:
    assert clean_reason(text) == expected


def test_long_reasons_are_cut_at_a_word_boundary() -> None:
    text = "word " * 100
    cleaned = clean_reason(text, max_chars=22)
    assert cleaned == "word word word word …"
