"""Tests for the screening metrics, against hand-computed values."""

import json

import pytest

from llm_literature_screener.calibrate import wilson_interval
from llm_literature_screener.metrics import (
    cohen_kappa,
    confusion,
    evaluate,
    rank_order,
    recall_curve,
    roc_auc,
    screened_to_reach,
    wss,
)


def test_confusion_and_derived_rates() -> None:
    flags = [True, True, True, False, False, False]
    labels = [True, False, False, True, False, False]
    counts = confusion(flags, labels)
    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (1, 2, 1, 2)
    assert counts.recall == 0.5
    assert counts.precision == pytest.approx(1 / 3)
    assert counts.workload_saved == 0.5
    nothing_flagged = confusion([False, False], [True, False])
    assert nothing_flagged.precision is None
    assert confusion([True], [False]).recall is None


def test_roc_auc_with_ties_counts_one_half() -> None:
    # Pairs: 0.9 beats all three exclusions; 0.8 beats two and ties one: (3 + 2.5) / 6.
    assert roc_auc([0.9, 0.8, 0.8, 0.3, 0.1], [True, False, True, False, False]) == pytest.approx(
        5.5 / 6
    )
    assert roc_auc([3.0, 2.0, 1.0], [True, True, False]) == 1.0
    assert roc_auc([1.0, 2.0, 3.0], [True, True, False]) == 0.0
    assert roc_auc([1.0, 1.0, 1.0, 1.0], [True, False, True, False]) == 0.5
    with pytest.raises(ValueError, match="one inclusion and one exclusion"):
        roc_auc([1.0, 2.0], [True, True])


def test_screened_to_reach_and_wss() -> None:
    # Ranked order puts the four inclusions at positions 1, 2, 5 and 9 of 10.
    scores = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    labels = [True, True, False, False, True, False, False, False, True, False]
    assert screened_to_reach(scores, labels, 0.5) == 2
    assert screened_to_reach(scores, labels, 0.95) == 9
    assert screened_to_reach(scores, labels, 1.0) == 9
    assert wss(scores, labels, 0.95) == pytest.approx((10 - 9) / 10 - 0.05)
    with pytest.raises(ValueError, match="no inclusions"):
        screened_to_reach(scores, [False] * 10, 0.95)
    with pytest.raises(ValueError, match="recall must be"):
        screened_to_reach(scores, labels, 0.0)


def test_ties_are_broken_against_the_model() -> None:
    # The inclusion shares its score with two exclusions, so it is read last of the three.
    assert screened_to_reach([1.0, 1.0, 1.0, 0.0], [True, False, False, False], 1.0) == 3
    # The output ranking itself keeps the input order for ties.
    assert rank_order([1.0, 2.0, 1.0]) == [1, 0, 2]


def test_recall_curve_is_the_exact_step_curve() -> None:
    curve = recall_curve([3.0, 2.0, 1.0, 0.0], [False, True, False, True])
    assert curve == [(0.0, 0.0), (0.25, 0.0), (0.5, 0.5), (0.75, 0.5), (1.0, 1.0)]


def test_cohen_kappa_textbook_example() -> None:
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no: observed 0.7, expected 0.5, kappa 0.4.
    first = [True] * 25 + [False] * 25
    second = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert cohen_kappa(first, second) == pytest.approx(0.4)
    assert cohen_kappa([True, False], [True, False]) == 1.0
    with pytest.raises(ValueError, match="undefined"):
        cohen_kappa([True, True], [True, True])
    with pytest.raises(ValueError, match="equally long"):
        cohen_kappa([True], [True, False])


def test_evaluate_combines_ranking_and_decision_metrics() -> None:
    scores = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    labels = [True, True, False, False, True, False, False, False, True, False]
    human = [True, True, True, False, True, False, False, False, False, False]
    result = evaluate(scores, labels, threshold=5.5, human=human)
    assert (result.records, result.inclusions, result.flagged, result.found) == (10, 4, 5, 3)
    assert result.recall == 0.75
    assert result.recall_ci == pytest.approx(wilson_interval(3, 4))
    assert result.precision == pytest.approx(0.6)
    assert result.workload_saved == 0.5
    assert result.screened_for_95 == 9
    assert result.screened_for_100 == 9
    assert result.auc == pytest.approx(roc_auc(scores, labels))
    assert result.kappa == pytest.approx(cohen_kappa([s >= 5.5 for s in scores], human))
    assert result.kappa_reference == "human"
    assert json.loads(json.dumps(result.to_dict()))["recall_ci"] == pytest.approx(
        list(result.recall_ci)
    )


def test_evaluate_without_flags_or_exclusions() -> None:
    result = evaluate([1.0, 2.0], [True, True], threshold=5.0)
    assert result.flagged == 0
    assert result.precision is None
    assert result.auc is None
    # Expected agreement is zero here, so kappa is defined: no agreement beyond chance.
    assert result.kappa == 0.0
    constant = evaluate([1.0, 2.0], [True, True], threshold=0.0)
    assert constant.kappa is None
    with pytest.raises(ValueError, match="no inclusions"):
        evaluate([1.0], [False], threshold=0.0)
