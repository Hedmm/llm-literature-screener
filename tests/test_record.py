"""Tests for the shared Record model."""

import math

import pytest

from llm_literature_screener import Assessment, Decision, Record, sigmoid


@pytest.mark.parametrize(
    ("x", "expected"),
    [(0.0, 0.5), (2.0, 1 / (1 + math.exp(-2.0))), (-2.0, 1 / (1 + math.exp(2.0)))],
)
def test_sigmoid_matches_the_logistic_function(x: float, expected: float) -> None:
    assert sigmoid(x) == pytest.approx(expected)


def test_sigmoid_does_not_overflow_for_large_magnitudes() -> None:
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0


def test_assessment_p_include_is_the_sigmoid_of_the_score() -> None:
    assessment = Assessment(score=-11.25, decision=Decision.EXCLUDE)
    assert assessment.p_include == pytest.approx(sigmoid(-11.25))


def test_decision_string_value() -> None:
    assert str(Decision.UNCERTAIN) == "uncertain"
    assert Decision("include") is Decision.INCLUDE


def test_raw_value_lookup_keeps_file_order() -> None:
    record = Record(id="1", raw=[("AU", "Doe, A."), ("TI", "A title"), ("AU", "Roe, B.")])
    assert record.raw_values("AU") == ["Doe, A.", "Roe, B."]
    assert record.raw_value("TI") == "A title"
    assert record.raw_value("AB", default="none") == "none"
