"""Threshold calibration on records the review team has already screened.

The threshold is the highest score that keeps at least the target recall (for example 95%)
of the known inclusions. Records below it are excluded; records at or above it go to human
screening. Among those, records below an upper threshold (by default a score of 0, where
the model leans towards "No") are marked uncertain, the rest include.

The recall reached on the known inclusions comes with a Wilson confidence interval. With
fewer than 10 known inclusions the threshold depends on very few records and the interval
is wide, so the calibration carries a warning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from llm_literature_screener.record import Decision

DEFAULT_TARGET_RECALL = 0.95
DEFAULT_UPPER_THRESHOLD = 0.0
"""Score at or above which a flagged record is marked include: P(include) >= 0.5."""
MIN_KNOWN_INCLUSIONS = 10
"""Below this number of known inclusions the calibration carries a warning."""
LADDER_TARGETS = (0.8, 0.9, 0.95, 0.98, 1.0)


class CalibrationError(ValueError):
    """Raised when a threshold cannot be calibrated, for example without known inclusions."""


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the Wilson score interval for the proportion ``successes / total``.

    Unlike the normal approximation, the interval stays within [0, 1] and behaves well for
    proportions near 0 or 1 and for small samples, which is the usual case for recall.

    Raises:
        ValueError: If ``total`` is not positive or ``successes`` is out of range.
    """
    if total <= 0:
        raise ValueError("The Wilson interval needs at least one trial.")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    ) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def required_hits(target_recall: float, inclusions: int) -> int:
    """Smallest number of inclusions that reaches ``target_recall`` (at least one)."""
    # The small tolerance stops float error from asking for one more hit, e.g. 0.95 * 20.
    return max(1, math.ceil(target_recall * inclusions - 1e-9))


def threshold_for_recall(inclusion_scores: Sequence[float], target_recall: float) -> float:
    """Return the highest threshold that keeps ``target_recall`` of the inclusions.

    Raises:
        CalibrationError: If there are no inclusion scores or the target is not in (0, 1].
    """
    if not 0 < target_recall <= 1:
        raise CalibrationError("The target recall must be greater than 0 and at most 1.")
    if not inclusion_scores:
        raise CalibrationError("Calibration needs at least one known inclusion.")
    ranked = sorted(inclusion_scores, reverse=True)
    return ranked[required_hits(target_recall, len(ranked)) - 1]


@dataclass(frozen=True)
class Calibration:
    """Decision thresholds and the recall they reach on the known inclusions.

    Attributes:
        threshold: Records scoring below it are excluded.
        upper_threshold: Records at or above it are marked include; flagged records below it
            are marked uncertain. Never lower than ``threshold``.
        method: ``target_recall`` when calibrated, ``fixed`` when set by the user.
        target_recall: The recall target used for calibration.
        confidence: Confidence level of ``recall_ci``.
        labeled: Number of records with a known human decision.
        known_inclusions: Number of known inclusions.
        found: Known inclusions at or above the threshold.
        recall: found / known_inclusions, or None without known inclusions.
        recall_ci: Wilson interval for the recall, or None without known inclusions.
        warnings: Reasons to treat the threshold with care.
    """

    threshold: float
    upper_threshold: float
    method: str
    target_recall: float | None
    confidence: float
    labeled: int
    known_inclusions: int
    found: int
    recall: float | None
    recall_ci: tuple[float, float] | None
    warnings: tuple[str, ...] = field(default=())

    def decide(self, score: float) -> Decision:
        """Return the decision for one score."""
        if score < self.threshold:
            return Decision.EXCLUDE
        if score >= self.upper_threshold:
            return Decision.INCLUDE
        return Decision.UNCERTAIN

    def to_dict(self) -> dict[str, Any]:
        """Return the calibration as plain data for JSON."""
        return {
            "method": self.method,
            "threshold": self.threshold,
            "upper_threshold": self.upper_threshold,
            "target_recall": self.target_recall,
            "confidence": self.confidence,
            "labeled": self.labeled,
            "known_inclusions": self.known_inclusions,
            "found": self.found,
            "recall": self.recall,
            "recall_ci": list(self.recall_ci) if self.recall_ci else None,
            "warnings": list(self.warnings),
        }


def _check_lengths(scores: Sequence[float], labels: Sequence[bool | None]) -> None:
    if len(scores) != len(labels):
        raise ValueError(f"Got {len(scores)} scores but {len(labels)} labels.")


def _summarize(
    scores: Sequence[float],
    labels: Sequence[bool | None],
    threshold: float,
    upper_threshold: float,
    method: str,
    target_recall: float | None,
    confidence: float,
) -> Calibration:
    inclusion_scores = [s for s, label in zip(scores, labels, strict=True) if label is True]
    labeled = sum(1 for label in labels if label is not None)
    found = sum(1 for score in inclusion_scores if score >= threshold)
    known = len(inclusion_scores)
    recall = found / known if known else None
    interval = wilson_interval(found, known, confidence) if known else None
    warnings: list[str] = []
    if interval is not None and known < MIN_KNOWN_INCLUSIONS:
        warnings.append(
            f"Only {known} known inclusion{'s' if known != 1 else ''}: with fewer than "
            f"{MIN_KNOWN_INCLUSIONS}, the threshold rests on very few records and the recall "
            f"estimate is uncertain ({confidence:.0%} CI {interval[0]:.0%} to "
            f"{interval[1]:.0%}). Label more records before relying on the exclusions."
        )
    if method == "fixed" and not known:
        warnings.append("No known inclusions, so the recall of this threshold cannot be estimated.")
    return Calibration(
        threshold=threshold,
        upper_threshold=max(upper_threshold, threshold),
        method=method,
        target_recall=target_recall,
        confidence=confidence,
        labeled=labeled,
        known_inclusions=known,
        found=found,
        recall=recall,
        recall_ci=interval,
        warnings=tuple(warnings),
    )


def calibrate(
    scores: Sequence[float],
    labels: Sequence[bool | None],
    target_recall: float = DEFAULT_TARGET_RECALL,
    upper_threshold: float = DEFAULT_UPPER_THRESHOLD,
    confidence: float = 0.95,
) -> Calibration:
    """Calibrate the exclusion threshold on the records that have a human label.

    Args:
        scores: One score per record (labeled and unlabeled).
        labels: True for a known inclusion, False for a known exclusion, None if unknown.
        target_recall: Share of the known inclusions the threshold must keep.
        upper_threshold: Score at or above which flagged records are marked include.
        confidence: Confidence level of the recall interval.

    Raises:
        CalibrationError: If there are no known inclusions or the target is invalid.
    """
    _check_lengths(scores, labels)
    inclusion_scores = [s for s, label in zip(scores, labels, strict=True) if label is True]
    if not inclusion_scores:
        raise CalibrationError(
            "None of the labeled records is an inclusion, so no threshold can be calibrated. "
            "Label more records (including some known inclusions) or set a threshold yourself."
        )
    threshold = threshold_for_recall(inclusion_scores, target_recall)
    return _summarize(
        scores, labels, threshold, upper_threshold, "target_recall", target_recall, confidence
    )


def fixed_threshold(
    scores: Sequence[float],
    labels: Sequence[bool | None],
    threshold: float,
    upper_threshold: float = DEFAULT_UPPER_THRESHOLD,
    confidence: float = 0.95,
) -> Calibration:
    """Use a threshold chosen by the user and report its recall on any known inclusions."""
    _check_lengths(scores, labels)
    return _summarize(scores, labels, threshold, upper_threshold, "fixed", None, confidence)


@dataclass(frozen=True)
class LadderRow:
    """The threshold and its consequences for one recall target."""

    target_recall: float
    threshold: float
    found: int
    recall: float
    recall_ci: tuple[float, float]
    flagged: int
    workload_saved: float


def recall_ladder(
    scores: Sequence[float],
    labels: Sequence[bool | None],
    targets: Sequence[float] = LADDER_TARGETS,
    confidence: float = 0.95,
) -> list[LadderRow]:
    """Show what each recall target would cost, to help choose one.

    ``flagged`` and ``workload_saved`` count all scored records, labeled or not.
    """
    rows = []
    for target in targets:
        calibration = calibrate(scores, labels, target, confidence=confidence)
        assert calibration.recall is not None and calibration.recall_ci is not None
        flagged = sum(1 for score in scores if score >= calibration.threshold)
        rows.append(
            LadderRow(
                target_recall=target,
                threshold=calibration.threshold,
                found=calibration.found,
                recall=calibration.recall,
                recall_ci=calibration.recall_ci,
                flagged=flagged,
                workload_saved=1 - flagged / len(scores),
            )
        )
    return rows
