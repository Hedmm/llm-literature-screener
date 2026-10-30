"""PRISMA 2020 counts for the screening stage and a methods paragraph.

PRISMA 2020 has a box for "records marked as ineligible by automation tools" among the
records removed before screening. Records this tool excludes belong there; the records it
flags (include or uncertain) are the ones the review team screens. The later boxes (reports
sought, assessed and included) depend on human decisions and are left for the team.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from llm_literature_screener.backends.base import ScorerInfo
from llm_literature_screener.calibrate import Calibration
from llm_literature_screener.record import Decision, sigmoid


@dataclass(frozen=True)
class PrismaCounts:
    """Counts for the identification and screening boxes of a PRISMA 2020 flow diagram.

    Attributes:
        identified: Records read from the input files.
        duplicates_removed: Duplicate records removed before screening.
        assessed_by_tool: Unique records the tool scored.
        excluded_by_tool: Records marked as ineligible by the automation tool.
        to_screen: Records passed on to title and abstract screening by people.
        include: Flagged records the model leans towards including.
        uncertain: Flagged records the model leans against but that are above the threshold.
        sources: Records identified per input file, as (file name, count) pairs.
    """

    identified: int
    duplicates_removed: int
    assessed_by_tool: int
    excluded_by_tool: int
    to_screen: int
    include: int
    uncertain: int
    sources: tuple[tuple[str, int], ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """Return the counts as plain data for JSON."""
        data = asdict(self)
        data["sources"] = dict(self.sources)
        return data


def prisma_counts(
    identified: int,
    duplicates_removed: int,
    decisions: Sequence[Decision],
    sources: Sequence[tuple[str, int]] = (),
) -> PrismaCounts:
    """Count the screening-stage boxes from the tool's decisions."""
    include = sum(1 for decision in decisions if decision is Decision.INCLUDE)
    uncertain = sum(1 for decision in decisions if decision is Decision.UNCERTAIN)
    excluded = sum(1 for decision in decisions if decision is Decision.EXCLUDE)
    return PrismaCounts(
        identified=identified,
        duplicates_removed=duplicates_removed,
        assessed_by_tool=len(decisions),
        excluded_by_tool=excluded,
        to_screen=include + uncertain,
        include=include,
        uncertain=uncertain,
        sources=tuple(sources),
    )


def _model_phrase(info: ScorerInfo) -> str:
    if info.backend == "hf":
        return (
            f"the open large language model {info.model}, run locally with Hugging Face "
            "transformers and without fine-tuning"
        )
    if info.backend == "openai":
        return f"the large language model {info.model}, served through an OpenAI-compatible API"
    if info.backend == "mock":
        return "a keyword-based mock scorer intended for demonstration only (not a language model)"
    return f"the model {info.model} ({info.description})"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def methods_paragraph(
    info: ScorerInfo,
    calibration: Calibration,
    counts: PrismaCounts,
    tool_version: str,
    *,
    deduplicated: bool = True,
    reasons: bool = False,
) -> str:
    """Write a methods paragraph for the screening stage of a systematic review.

    It states the model, how the score was computed, the threshold rule with its recall
    target, the recall reached on the known inclusions, and that people made every final
    eligibility decision.
    """
    sentences = [
        f"Titles and abstracts were pre-screened with llm-literature-screener {tool_version} "
        f"using {_model_phrase(info)}."
    ]
    if deduplicated:
        sentences.append(
            f"Before screening, {counts.duplicates_removed} duplicate records were removed by "
            "matching DOIs, PubMed IDs, and normalized titles with publication years."
        )
    sentences.append(
        "For each record, the model was given the review question, the eligibility criteria, "
        "and the record's title and abstract, and was asked whether the record should go to "
        f"full-text review; the screening score was {info.score_definition}."
    )
    threshold = calibration.threshold
    if calibration.method == "target_recall" and calibration.target_recall is not None:
        sentences.append(
            f"The exclusion threshold was calibrated on {calibration.labeled} records that the "
            f"review team had already screened, {calibration.known_inclusions} of them known "
            "inclusions: it was set to the highest score that kept at least "
            f"{calibration.target_recall:.0%} of the known inclusions "
            f"(a score of {threshold:.2f}, P(include) = {sigmoid(threshold):.3g})."
        )
    else:
        sentences.append(
            f"The exclusion threshold was set by the review team at a score of {threshold:.2f} "
            f"(P(include) = {sigmoid(threshold):.3g})."
        )
    if calibration.recall is not None and calibration.recall_ci is not None:
        low, high = calibration.recall_ci
        sentences.append(
            f"At this threshold, {calibration.found} of {calibration.known_inclusions} known "
            f"inclusions were kept (recall {_percent(calibration.recall)}; "
            f"{calibration.confidence:.0%} Wilson confidence interval {_percent(low)} to "
            f"{_percent(high)})."
        )
    sentences.append(
        f"Records scoring below the threshold (n = {counts.excluded_by_tool}) were marked as "
        f"ineligible by an automation tool, and the remaining {counts.to_screen} records were "
        "screened by the review team."
    )
    if reasons:
        sentences.append(
            "For these records the model also wrote a one-line reason to support the reviewers."
        )
    sentences.append(
        "The tool made no final eligibility decision: every inclusion and exclusion was decided "
        "by the reviewers."
    )
    return " ".join(sentences)


def prisma_markdown(counts: PrismaCounts, methods: str) -> str:
    """Return the PRISMA counts and the methods paragraph as a Markdown document."""
    rows = [("Records identified", f"{counts.identified:,}")]
    rows.extend((f"Records identified from {name}", f"{count:,}") for name, count in counts.sources)
    rows.extend(
        [
            ("Duplicate records removed", f"{counts.duplicates_removed:,}"),
            ("Records marked as ineligible by automation tools", f"{counts.excluded_by_tool:,}"),
            ("Records screened by the review team", f"{counts.to_screen:,}"),
            ("Records excluded by the review team", "to be completed by the review team"),
        ]
    )
    lines = [
        "# PRISMA 2020: identification and screening",
        "",
        "| PRISMA 2020 box | Count |",
        "|---|---|",
        *(f"| {label} | {value} |" for label, value in rows),
        "",
        f"Of the {counts.to_screen:,} records passed to the review team, the model leaned "
        f"towards inclusion for {counts.include:,} and was uncertain about {counts.uncertain:,}.",
        "",
        "## Methods paragraph",
        "",
        methods,
        "",
    ]
    return "\n".join(lines)
