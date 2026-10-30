"""Screening metrics: recall, precision, workload saved, WSS@95, ROC AUC, recall curve, kappa.

Ranking metrics follow the order of the scores, highest first. Equal scores are common
(bf16 logits have a coarse resolution), so ties are broken against the model: among records
with the same score, inclusions are read last. Records read to reach a recall level, WSS@95
and the recall curve therefore never look better than a reviewer reading in score order would
find them. ROC AUC counts ties as one half, as usual.

Decision metrics compare the records a threshold flags for human screening (score at or
above the threshold) with the human labels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from llm_literature_screener.calibrate import wilson_interval


def rank_order(scores: Sequence[float]) -> list[int]:
    """Return record indices sorted by score, highest first; ties keep their input order."""
    return sorted(range(len(scores)), key=lambda index: -scores[index])


def evaluation_order(scores: Sequence[float], labels: Sequence[bool]) -> list[int]:
    """Return record indices by score, highest first, with inclusions last among ties."""
    return sorted(range(len(scores)), key=lambda index: (-scores[index], labels[index]))


def _check(scores: Sequence[float], labels: Sequence[bool]) -> None:
    if len(scores) != len(labels):
        raise ValueError(f"Got {len(scores)} scores but {len(labels)} labels.")
    if not labels:
        raise ValueError("Metrics need at least one record.")


@dataclass(frozen=True)
class Confusion:
    """Counts of flagged and not flagged records against the human labels."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total(self) -> int:
        """Number of records."""
        return self.tp + self.fp + self.fn + self.tn

    @property
    def flagged(self) -> int:
        """Records flagged for human screening."""
        return self.tp + self.fp

    @property
    def recall(self) -> float | None:
        """Share of inclusions that were flagged, or None without inclusions."""
        positives = self.tp + self.fn
        return self.tp / positives if positives else None

    @property
    def precision(self) -> float | None:
        """Share of flagged records that are inclusions, or None when nothing is flagged."""
        return self.tp / self.flagged if self.flagged else None

    @property
    def workload_saved(self) -> float:
        """Share of records that nobody has to read: those not flagged."""
        return 1.0 - self.flagged / self.total if self.total else 0.0


def confusion(flags: Sequence[bool], labels: Sequence[bool]) -> Confusion:
    """Cross-tabulate flags (True = flagged for screening) against labels (True = included)."""
    if len(flags) != len(labels):
        raise ValueError(f"Got {len(flags)} flags but {len(labels)} labels.")
    tp = sum(1 for flag, label in zip(flags, labels, strict=True) if flag and label)
    fp = sum(1 for flag, label in zip(flags, labels, strict=True) if flag and not label)
    fn = sum(1 for flag, label in zip(flags, labels, strict=True) if not flag and label)
    return Confusion(tp=tp, fp=fp, fn=fn, tn=len(flags) - tp - fp - fn)


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve: the chance that an inclusion outscores an exclusion.

    Computed from the Mann-Whitney U statistic, with tied scores counting one half.

    Raises:
        ValueError: If the labels do not contain both inclusions and exclusions.
    """
    _check(scores, labels)
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("ROC AUC needs at least one inclusion and one exclusion.")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        average_rank = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = average_rank
        start = end + 1
    rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def screened_to_reach(scores: Sequence[float], labels: Sequence[bool], recall: float) -> int:
    """Number of records read in score order until ``recall`` of the inclusions are found.

    Raises:
        ValueError: If there are no inclusions or ``recall`` is not in (0, 1].
    """
    _check(scores, labels)
    if not 0 < recall <= 1:
        raise ValueError("recall must be in (0, 1].")
    positives = sum(1 for label in labels if label)
    if not positives:
        raise ValueError("The labels contain no inclusions.")
    needed = max(1, math.ceil(recall * positives - 1e-9))
    found = 0
    for position, index in enumerate(evaluation_order(scores, labels), start=1):
        if labels[index]:
            found += 1
            if found >= needed:
                return position
    raise AssertionError("unreachable: every inclusion is found by the end")  # pragma: no cover


def wss(scores: Sequence[float], labels: Sequence[bool], recall: float = 0.95) -> float:
    """Work saved over sampling at a recall level, for example WSS@95.

    WSS@r = (N - n_r) / N - (1 - r), where n_r is the number of records read in score order
    to find r of the inclusions. Reading in random order saves nothing on average.
    """
    screened = screened_to_reach(scores, labels, recall)
    return (len(scores) - screened) / len(scores) - (1 - recall)


def recall_curve(scores: Sequence[float], labels: Sequence[bool]) -> list[tuple[float, float]]:
    """Return the exact recall-vs-screened step curve as (share read, share found) points.

    Records are read in score order. The curve rises by one inclusion at each inclusion's
    rank and is flat in between; it starts at (0, 0) and ends at (1, 1).

    Raises:
        ValueError: If there are no inclusions.
    """
    _check(scores, labels)
    positives = sum(1 for label in labels if label)
    if not positives:
        raise ValueError("The labels contain no inclusions.")
    total = len(scores)
    points = [(0.0, 0.0)]
    found = 0
    for position, index in enumerate(evaluation_order(scores, labels), start=1):
        if labels[index]:
            points.append(((position - 1) / total, found / positives))
            found += 1
            points.append((position / total, found / positives))
    points.append((1.0, 1.0))
    unique = [points[0]]
    for point in points[1:]:
        if point != unique[-1]:
            unique.append(point)
    return unique


def cohen_kappa(first: Sequence[bool], second: Sequence[bool]) -> float:
    """Cohen's kappa: agreement between two binary raters beyond chance.

    Raises:
        ValueError: If the inputs differ in length, are empty, or kappa is undefined because
            both raters give one and the same answer to every record.
    """
    if len(first) != len(second) or not first:
        raise ValueError("Kappa needs two equally long, non-empty sequences.")
    total = len(first)
    observed = sum(1 for a, b in zip(first, second, strict=True) if a == b) / total
    rate_first = sum(1 for a in first if a) / total
    rate_second = sum(1 for b in second if b) / total
    expected = rate_first * rate_second + (1 - rate_first) * (1 - rate_second)
    if expected == 1.0:
        raise ValueError("Kappa is undefined when both raters always give the same answer.")
    return (observed - expected) / (1 - expected)


@dataclass(frozen=True)
class Evaluation:
    """Metrics for one set of scores against complete human labels.

    Attributes:
        records: Number of records.
        inclusions: Number of human inclusions.
        threshold: Score threshold; records at or above it are flagged.
        flagged: Records flagged for human screening.
        found: Inclusions among the flagged records.
        recall: found / inclusions.
        recall_ci: Wilson confidence interval of the recall.
        precision: found / flagged (None when nothing is flagged).
        workload_saved: Share of records not flagged.
        auc: ROC AUC of the scores (None without exclusions).
        wss95: Work saved over sampling at 95% recall.
        screened_for_95: Records read in score order to find 95% of the inclusions.
        screened_for_100: Records read in score order to find every inclusion.
        kappa: Cohen's kappa between the flags and the reference human decisions.
        kappa_reference: Which human decisions the kappa uses: ``labels`` or ``human``.
        confidence: Confidence level of ``recall_ci``.
    """

    records: int
    inclusions: int
    threshold: float
    flagged: int
    found: int
    recall: float
    recall_ci: tuple[float, float]
    precision: float | None
    workload_saved: float
    auc: float | None
    wss95: float
    screened_for_95: int
    screened_for_100: int
    kappa: float | None
    kappa_reference: str
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as plain data for JSON."""
        data = asdict(self)
        data["recall_ci"] = list(self.recall_ci)
        return data


def evaluate(
    scores: Sequence[float],
    labels: Sequence[bool],
    threshold: float,
    human: Sequence[bool] | None = None,
    confidence: float = 0.95,
) -> Evaluation:
    """Compute ranking and decision metrics against complete human labels.

    Args:
        scores: One score per record.
        labels: Reference decision per record (True = included), used for recall,
            precision and the ranking metrics.
        threshold: Records at or above it are flagged for human screening.
        human: Optional second set of human decisions for Cohen's kappa, for example the
            title and abstract decisions when ``labels`` are final inclusions.
        confidence: Confidence level of the recall interval.

    Raises:
        ValueError: If lengths differ or the labels contain no inclusions.
    """
    _check(scores, labels)
    flags = [score >= threshold for score in scores]
    counts = confusion(flags, labels)
    positives = counts.tp + counts.fn
    if not positives:
        raise ValueError("The labels contain no inclusions, so recall is undefined.")
    try:
        auc: float | None = roc_auc(scores, labels)
    except ValueError:
        auc = None
    reference = list(human) if human is not None else list(labels)
    try:
        kappa: float | None = cohen_kappa(flags, reference)
    except ValueError:
        kappa = None
    return Evaluation(
        records=len(scores),
        inclusions=positives,
        threshold=threshold,
        flagged=counts.flagged,
        found=counts.tp,
        recall=counts.tp / positives,
        recall_ci=wilson_interval(counts.tp, positives, confidence),
        precision=counts.precision,
        workload_saved=counts.workload_saved,
        auc=auc,
        wss95=wss(scores, labels, 0.95),
        screened_for_95=screened_to_reach(scores, labels, 0.95),
        screened_for_100=screened_to_reach(scores, labels, 1.0),
        kappa=kappa,
        kappa_reference="human" if human is not None else "labels",
        confidence=confidence,
    )
