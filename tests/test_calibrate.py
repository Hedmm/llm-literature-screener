"""Tests for threshold calibration and the Wilson interval."""

import json

import pytest

from llm_literature_screener import Decision
from llm_literature_screener.calibrate import (
    CalibrationError,
    calibrate,
    fixed_threshold,
    recall_ladder,
    required_hits,
    threshold_for_recall,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("successes", "total", "low", "high"),
    [
        # Newcombe (1998), Statistics in Medicine 17:857-872, Wilson score interval.
        (81, 263, 0.2553, 0.3662),
        (15, 148, 0.0624, 0.1605),
        (0, 20, 0.0000, 0.1611),
        (1, 29, 0.0061, 0.1718),
        # Recall intervals published in the case study (rounded to three decimals there).
        (28, 29, 0.8282, 0.9939),
        (19, 19, 0.8318, 1.0000),
        (69, 73, 0.8674, 0.9785),
        (88, 92, 0.8935, 0.9830),
    ],
)
def test_wilson_interval_matches_published_values(
    successes: int, total: int, low: float, high: float
) -> None:
    assert wilson_interval(successes, total) == pytest.approx((low, high), abs=5e-5)


def test_wilson_interval_widens_with_confidence_and_rejects_bad_input() -> None:
    low95, high95 = wilson_interval(8, 10, 0.95)
    low99, high99 = wilson_interval(8, 10, 0.99)
    assert low99 < low95 < 0.8 < high95 < high99
    with pytest.raises(ValueError, match="at least one trial"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="between 0 and total"):
        wilson_interval(5, 4)
    with pytest.raises(ValueError, match="confidence"):
        wilson_interval(1, 2, confidence=1.0)


@pytest.mark.parametrize(
    ("target", "inclusions", "expected"),
    [(0.95, 20, 19), (0.95, 29, 28), (0.95, 19, 19), (0.9, 10, 9), (1.0, 7, 7), (0.01, 5, 1)],
)
def test_required_hits_is_robust_to_float_error(
    target: float, inclusions: int, expected: int
) -> None:
    assert required_hits(target, inclusions) == expected


def test_threshold_is_the_highest_score_that_keeps_the_target() -> None:
    inclusion_scores = [5.0, 3.0, 1.0, -2.0, -4.0]
    assert threshold_for_recall(inclusion_scores, 0.8) == -2.0
    assert threshold_for_recall(inclusion_scores, 0.6) == 1.0
    assert threshold_for_recall(inclusion_scores, 1.0) == -4.0
    with pytest.raises(CalibrationError, match="target recall"):
        threshold_for_recall(inclusion_scores, 1.5)
    with pytest.raises(CalibrationError, match="at least one"):
        threshold_for_recall([], 0.95)


def test_calibrate_uses_only_labeled_inclusions_and_reports_recall() -> None:
    scores = [6.0, 4.0, 2.0, 1.0, -1.0, -3.0, -5.0, -8.0, 0.5, -6.0]
    labels = [True, False, True, None, True, False, True, None, None, False]
    calibration = calibrate(scores, labels, target_recall=0.75, upper_threshold=0.0)
    assert calibration.threshold == -1.0
    assert (calibration.known_inclusions, calibration.found, calibration.labeled) == (4, 3, 7)
    assert calibration.recall == 0.75
    assert calibration.recall_ci == pytest.approx(wilson_interval(3, 4))
    assert calibration.method == "target_recall"
    assert [calibration.decide(score) for score in (6.0, 0.0, -0.5, -1.0, -1.01)] == [
        Decision.INCLUDE,
        Decision.INCLUDE,
        Decision.UNCERTAIN,
        Decision.UNCERTAIN,
        Decision.EXCLUDE,
    ]


def test_tied_inclusions_at_the_threshold_are_all_kept() -> None:
    calibration = calibrate([3.0, 1.0, 1.0, 1.0], [True, True, True, True], target_recall=0.5)
    assert calibration.threshold == 1.0
    assert calibration.found == 4
    assert calibration.recall == 1.0


def test_upper_threshold_never_falls_below_the_threshold() -> None:
    calibration = calibrate([5.0, 4.0, 3.0], [True, True, False], target_recall=1.0)
    assert calibration.threshold == 4.0
    assert calibration.upper_threshold == 4.0
    assert calibration.decide(4.0) is Decision.INCLUDE
    assert calibration.decide(3.9) is Decision.EXCLUDE


def test_few_known_inclusions_give_a_warning() -> None:
    few = calibrate([1.0] * 9 + [-1.0], [True] * 9 + [False])
    (warning,) = few.warnings
    assert "Only 9 known inclusions" in warning
    assert "fewer than 10" in warning
    enough = calibrate([1.0] * 10, [True] * 10)
    assert enough.warnings == ()


def test_calibration_needs_a_known_inclusion() -> None:
    with pytest.raises(CalibrationError, match="None of the labeled records is an inclusion"):
        calibrate([1.0, 2.0], [False, None])
    with pytest.raises(ValueError, match="2 scores but 1 labels"):
        calibrate([1.0, 2.0], [True])


def test_fixed_threshold_reports_recall_when_labels_exist() -> None:
    calibration = fixed_threshold([2.0, -1.0, -4.0], [True, True, None], threshold=-2.0)
    assert calibration.method == "fixed"
    assert calibration.target_recall is None
    assert (calibration.found, calibration.known_inclusions, calibration.recall) == (2, 2, 1.0)
    unlabeled = fixed_threshold([2.0, -1.0], [None, None], threshold=0.0)
    assert unlabeled.recall is None
    assert unlabeled.recall_ci is None
    assert any("cannot be estimated" in warning for warning in unlabeled.warnings)


def test_recall_ladder_trades_recall_for_workload() -> None:
    scores = [float(score) for score in range(20, 0, -1)]
    labels: list[bool | None] = [index % 2 == 0 for index in range(20)]
    rows = recall_ladder(scores, labels, targets=(0.5, 0.8, 1.0))
    assert [row.target_recall for row in rows] == [0.5, 0.8, 1.0]
    assert [row.threshold for row in rows] == [12.0, 6.0, 2.0]
    assert [row.flagged for row in rows] == [9, 15, 19]
    assert rows[-1].workload_saved == pytest.approx(0.05)
    assert all(row.recall >= row.target_recall for row in rows)


def test_calibration_serializes_to_json() -> None:
    calibration = calibrate([1.0, -1.0], [True, False])
    data = json.loads(json.dumps(calibration.to_dict()))
    assert data["threshold"] == 1.0
    assert data["recall_ci"] == pytest.approx(list(wilson_interval(1, 1)))
