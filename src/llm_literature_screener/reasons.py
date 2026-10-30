"""Optional one-line reasons for the records that people will read.

Generating text costs far more than scoring, so reasons are generated only for records at
or above the exclusion threshold: those marked include or uncertain. Excluded records get
no reason.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from llm_literature_screener.backends.base import ProgressCallback, Scorer
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.record import Record

MAX_REASON_CHARS = 300
_PREFIX = re.compile(r"^(?:[-*•>\s]+|reason\s*:\s*|answer\s*:\s*)+", re.IGNORECASE)


def clean_reason(text: str, max_chars: int = MAX_REASON_CHARS) -> str:
    """Reduce generated text to one tidy line.

    Keeps the first non-empty line, removes list markers, a "Reason:" prefix and wrapping
    quotes, and cuts the text at a word boundary if it is longer than ``max_chars``.
    """
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    line = _PREFIX.sub("", line).strip().strip("\"'“”").strip()
    line = " ".join(line.split())
    if len(line) > max_chars:
        line = line[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + " …"
    return line


def records_needing_reasons(scores: Sequence[float], threshold: float) -> list[int]:
    """Return the indices of records at or above the exclusion threshold."""
    return [index for index, score in enumerate(scores) if score >= threshold]


def generate_reasons(
    records: Sequence[Record],
    criteria: Criteria,
    scorer: Scorer,
    scores: Sequence[float],
    threshold: float,
    progress: ProgressCallback | None = None,
) -> list[str | None]:
    """Generate a one-line reason for every record at or above ``threshold``.

    Returns:
        One entry per record: the reason, or None for records below the threshold.
    """
    selected = records_needing_reasons(scores, threshold)
    reasons: list[str | None] = [None] * len(records)
    if not selected:
        return reasons
    texts = scorer.explain([records[index] for index in selected], criteria, progress)
    for index, text in zip(selected, texts, strict=True):
        reasons[index] = clean_reason(text) or None
    return reasons
