"""The screening pipeline shared by the command line and the web app.

1. Read one or more records files and combine them.
2. Attach human labels from a labels file, if given.
3. Remove duplicates (by DOI, PMID, and normalized title plus year).
4. Score every unique record with the chosen backend.
5. Calibrate the exclusion threshold on the labeled records, or use a fixed threshold.
6. Decide include, uncertain or exclude, and optionally generate one-line reasons.
7. Write the results, the records in their original formats, PRISMA counts, a methods
   paragraph and the HTML report.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_literature_screener import __version__
from llm_literature_screener.backends.base import ProgressCallback, Scorer, ScorerInfo
from llm_literature_screener.calibrate import (
    DEFAULT_TARGET_RECALL,
    DEFAULT_UPPER_THRESHOLD,
    Calibration,
    CalibrationError,
    calibrate,
    fixed_threshold,
)
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.dedup import DedupResult, find_duplicates, write_dedup_report
from llm_literature_screener.io import (
    SUFFIXES,
    apply_labels,
    detect_format,
    read_labels,
    read_records,
    read_results,
    read_scores,
    write_records,
    write_results,
)
from llm_literature_screener.prisma import (
    PrismaCounts,
    methods_paragraph,
    prisma_counts,
    prisma_markdown,
)
from llm_literature_screener.reasons import generate_reasons
from llm_literature_screener.record import Assessment, Record
from llm_literature_screener.report import ReportData, write_report

StageProgress = Callable[[str, int, int], None]
"""Called with (stage, done, total). The stages are ``scoring`` and ``reasons``."""

RESULTS_FILE = "screening_results.csv"
REPORT_FILE = "report.html"
PRISMA_FILE = "prisma.md"
SUMMARY_FILE = "run.json"
CALIBRATION_FILE = "calibration.json"
DUPLICATES_FILE = "duplicates.csv"
CRITERIA_FILE = "criteria.yaml"


@dataclass(frozen=True)
class SourceFile:
    """One records file read for screening.

    Attributes:
        path: The file.
        format: Its format name (see :data:`llm_literature_screener.io.FORMATS`).
        records: The records read from it.
    """

    path: Path
    format: str
    records: list[Record]


def load_source(
    path: Path,
    fmt: str | None = None,
    columns: Mapping[str, str] | None = None,
    sheet: str | None = None,
) -> SourceFile:
    """Read a records file, detecting its format when ``fmt`` is not given."""
    detected = fmt or detect_format(path)
    return SourceFile(path, detected, read_records(path, detected, columns, sheet))


def combine_sources(sources: Sequence[SourceFile]) -> tuple[list[Record], dict[str, int]]:
    """Combine the records of several files into one list with unique ids.

    With one file the ids are kept. With several, each id is prefixed with its file name
    (``pubmed.nbib:12345``), so that ids from different files never collide.

    Returns:
        The combined records and, for each record id, the index of its source file.
    """
    if len(sources) == 1:
        records = list(sources[0].records)
        return records, {record.id: 0 for record in records}
    names = [source.path.name for source in sources]
    if len(set(names)) < len(names):
        names = [f"{index + 1}-{name}" for index, name in enumerate(names)]
    combined: list[Record] = []
    source_of: dict[str, int] = {}
    for index, (name, source) in enumerate(zip(names, sources, strict=True)):
        for record in source.records:
            prefixed = replace(record, id=f"{name}:{record.id}")
            combined.append(prefixed)
            source_of[prefixed.id] = index
    return combined, source_of


@dataclass
class ScreeningOptions:
    """Settings of a screening run.

    Attributes:
        target_recall: Share of the known inclusions the threshold must keep.
        threshold: A fixed exclusion threshold; calibration is skipped when it is set.
        upper_threshold: Score at or above which flagged records are marked include.
        deduplicate: Remove duplicate records before scoring.
        reasons: Generate a one-line reason for every record at or above the threshold.
        confidence: Confidence level of the recall interval.
    """

    target_recall: float = DEFAULT_TARGET_RECALL
    threshold: float | None = None
    upper_threshold: float = DEFAULT_UPPER_THRESHOLD
    deduplicate: bool = True
    reasons: bool = False
    confidence: float = 0.95


@dataclass
class ScreeningRun:
    """The outcome of :func:`run_screening`.

    Attributes:
        records: Unique records in input order, each with an assessment.
        input_records: All records before duplicate removal.
        dedup: The duplicate report, or None when deduplication was off.
        calibration: The thresholds used for the decisions.
        prisma: PRISMA 2020 counts for the screening stage.
        methods: Methods paragraph describing the run.
        info: The scorer that produced the scores.
        criteria: The review protocol.
        options: The settings of the run.
        title: Title used in the report.
        warnings: Warnings from the scorer and the pipeline (calibration warnings are
            kept in ``calibration.warnings``).
        source_of: Index of the source file of each unique record, by record id.
        started_at: When the run started.
        seconds: Time spent scoring.
    """

    records: list[Record]
    input_records: list[Record]
    dedup: DedupResult | None
    calibration: Calibration
    prisma: PrismaCounts
    methods: str
    info: ScorerInfo
    criteria: Criteria
    options: ScreeningOptions
    title: str
    warnings: list[str] = field(default_factory=list)
    source_of: dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    seconds: float = 0.0

    def report_data(self) -> ReportData:
        """Return the data for the HTML report of this run."""
        return ReportData(
            title=self.title,
            records=self.records,
            calibration=self.calibration,
            info=self.info,
            tool_version=__version__,
            generated_at=datetime.now(),
            criteria=self.criteria,
            prisma=self.prisma,
            methods=self.methods,
            warnings=self.warnings,
            seconds=self.seconds,
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready summary of the run: settings, counts and calibration."""
        return {
            "tool": "llm-literature-screener",
            "version": __version__,
            "title": self.title,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "scoring_seconds": round(self.seconds, 2),
            "scorer": asdict(self.info),
            "options": asdict(self.options),
            "prisma": self.prisma.to_dict(),
            "calibration": self.calibration.to_dict(),
            "methods": self.methods,
            "warnings": self.warnings,
        }


def _stage(progress: StageProgress | None, name: str) -> ProgressCallback | None:
    if progress is None:
        return None

    def report(done: int, total: int) -> None:
        progress(name, done, total)

    return report


def run_screening(
    records: Sequence[Record],
    criteria: Criteria,
    scorer: Scorer,
    options: ScreeningOptions | None = None,
    *,
    progress: StageProgress | None = None,
    sources: Sequence[tuple[str, int]] = (),
    source_of: Mapping[str, int] | None = None,
    title: str | None = None,
) -> ScreeningRun:
    """Deduplicate, score, calibrate and decide.

    Args:
        records: Records to screen, with human labels where known.
        criteria: The review protocol.
        scorer: The model backend.
        options: Settings of the run.
        progress: Optional callback for progress bars.
        sources: (file name, records read) pairs for the PRISMA counts.
        source_of: Source file index of each record id, kept for writing outputs.
        title: Title for the report; defaults to the protocol's title.

    Raises:
        ValueError: If there are no records.
        CalibrationError: If no threshold is given and no record is a known inclusion.
            This is checked before scoring, so no model time is wasted.
    """
    options = options or ScreeningOptions()
    if not records:
        raise ValueError("There are no records to screen.")
    started_at = datetime.now()
    dedup = find_duplicates(records) if options.deduplicate else None
    unique = dedup.records if dedup else list(records)
    if options.threshold is None and not any(record.label is True for record in unique):
        raise CalibrationError(
            "No record is labeled as a known inclusion, so the threshold cannot be calibrated. "
            "Give decisions your team has already made (--labels), or set --threshold."
        )

    clock = time.perf_counter()
    scores = scorer.score(unique, criteria, progress=_stage(progress, "scoring"))
    seconds = time.perf_counter() - clock

    labels = [record.label for record in unique]
    if options.threshold is None:
        calibration = calibrate(
            scores, labels, options.target_recall, options.upper_threshold, options.confidence
        )
    else:
        calibration = fixed_threshold(
            scores, labels, options.threshold, options.upper_threshold, options.confidence
        )
    reasons: list[str | None] = [None] * len(unique)
    if options.reasons:
        reasons = generate_reasons(
            unique,
            criteria,
            scorer,
            scores,
            calibration.threshold,
            progress=_stage(progress, "reasons"),
        )
    assessed = [
        replace(record, assessment=Assessment(score, calibration.decide(score), reason))
        for record, score, reason in zip(unique, scores, reasons, strict=True)
    ]
    decisions = [record.assessment.decision for record in assessed if record.assessment]
    counts = prisma_counts(len(records), dedup.removed if dedup else 0, decisions, sources)
    methods = methods_paragraph(
        scorer.info,
        calibration,
        counts,
        __version__,
        deduplicated=options.deduplicate,
        reasons=options.reasons,
    )
    return ScreeningRun(
        records=assessed,
        input_records=list(records),
        dedup=dedup,
        calibration=calibration,
        prisma=counts,
        methods=methods,
        info=scorer.info,
        criteria=criteria,
        options=options,
        title=title or criteria.title or "Title and abstract screening",
        warnings=list(scorer.warnings),
        source_of=dict(source_of or {}),
        started_at=started_at,
        seconds=seconds,
    )


def _screened_name(source: SourceFile, taken: set[str]) -> str:
    suffix = source.path.suffix if source.path.suffix else SUFFIXES[source.format]
    name = f"{source.path.stem}_screened{suffix}"
    counter = 2
    while name in taken:
        name = f"{source.path.stem}_screened_{counter}{suffix}"
        counter += 1
    taken.add(name)
    return name


def write_outputs(
    run: ScreeningRun,
    out_dir: Path,
    sources: Sequence[SourceFile] = (),
    *,
    report: bool = True,
) -> dict[str, Path]:
    """Write every output of a run to ``out_dir`` and return the paths by kind.

    The outputs are the ranked results CSV, each source file in its own format with the
    tool's output added, the PRISMA counts with the methods paragraph, the calibration and a
    run summary as JSON, the duplicate report, a copy of the protocol, and the HTML report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["results"] = out_dir / RESULTS_FILE
    write_results(run.records, paths["results"])

    taken: set[str] = set()
    for index, source in enumerate(sources):
        own = [record for record in run.records if run.source_of.get(record.id, 0) == index]
        path = out_dir / _screened_name(source, taken)
        write_records(own, path, source.format)
        paths[f"records:{source.path.name}"] = path

    paths["prisma"] = out_dir / PRISMA_FILE
    paths["prisma"].write_text(prisma_markdown(run.prisma, run.methods), encoding="utf-8")

    paths["calibration"] = out_dir / CALIBRATION_FILE
    paths["calibration"].write_text(
        json.dumps(run.calibration.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    if run.dedup is not None:
        paths["duplicates"] = out_dir / DUPLICATES_FILE
        write_dedup_report(run.dedup, run.input_records, paths["duplicates"])

    paths["criteria"] = out_dir / CRITERIA_FILE
    paths["criteria"].write_text(run.criteria.to_yaml(), encoding="utf-8")

    if report:
        paths["report"] = out_dir / REPORT_FILE
        write_report(run.report_data(), paths["report"])

    paths["summary"] = out_dir / SUMMARY_FILE
    summary = run.summary()
    summary["files"] = {kind: path.name for kind, path in paths.items()}
    summary["sources"] = [
        {"file": str(source.path), "format": source.format, "records": len(source.records)}
        for source in sources
    ]
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return paths


@dataclass
class SavedRun:
    """What ``lls screen`` saved in a results directory.

    Attributes:
        records: The ranked records from the results CSV, with assessments.
        calibration: The thresholds used, if ``calibration.json`` exists.
        info: The scorer, if ``run.json`` exists.
        prisma: The PRISMA counts, if ``run.json`` exists.
        title: The report title, if ``run.json`` exists.
        warnings: Warnings recorded during the run.
        criteria: The protocol, if ``criteria.yaml`` exists.
        reasons: Whether reasons were generated.
        deduplicated: Whether duplicates were removed.
        seconds: Scoring time, if known.
    """

    records: list[Record]
    calibration: Calibration | None = None
    info: ScorerInfo | None = None
    prisma: PrismaCounts | None = None
    title: str | None = None
    warnings: list[str] = field(default_factory=list)
    criteria: Criteria | None = None
    reasons: bool = False
    deduplicated: bool = True
    seconds: float | None = None


def results_path(path: Path) -> Path:
    """Return the results CSV for a results directory, or ``path`` itself for a file."""
    return path / RESULTS_FILE if path.is_dir() else path


def _calibration_from_dict(data: Mapping[str, Any]) -> Calibration:
    interval = data.get("recall_ci")
    return Calibration(
        threshold=float(data["threshold"]),
        upper_threshold=float(data["upper_threshold"]),
        method=str(data["method"]),
        target_recall=data.get("target_recall"),
        confidence=float(data.get("confidence", 0.95)),
        labeled=int(data.get("labeled", 0)),
        known_inclusions=int(data.get("known_inclusions", 0)),
        found=int(data.get("found", 0)),
        recall=data.get("recall"),
        recall_ci=(float(interval[0]), float(interval[1])) if interval else None,
        warnings=tuple(data.get("warnings", ())),
    )


def load_saved_run(path: Path) -> SavedRun:
    """Load a results directory (or a results CSV with its sibling files).

    Raises:
        FileNotFoundError: If the results CSV does not exist.
    """
    csv_path = results_path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No results file at {csv_path}.")
    folder = csv_path.parent
    saved = SavedRun(records=read_results(csv_path))
    calibration_path = folder / CALIBRATION_FILE
    if calibration_path.is_file():
        saved.calibration = _calibration_from_dict(json.loads(calibration_path.read_text()))
    summary_path = folder / SUMMARY_FILE
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved.info = ScorerInfo(**summary["scorer"])
        prisma = dict(summary["prisma"])
        prisma["sources"] = tuple(prisma.get("sources", {}).items())
        saved.prisma = PrismaCounts(**prisma)
        saved.title = summary.get("title")
        saved.warnings = list(summary.get("warnings", []))
        saved.reasons = bool(summary.get("options", {}).get("reasons", False))
        saved.deduplicated = bool(summary.get("options", {}).get("deduplicate", True))
        saved.seconds = summary.get("scoring_seconds")
    criteria_path = folder / CRITERIA_FILE
    if criteria_path.is_file():
        saved.criteria = Criteria.from_yaml(criteria_path)
    return saved


def unknown_scorer(source: Path) -> ScorerInfo:
    """Describe scores that were read from a file without information about the model."""
    return ScorerInfo(
        backend="unknown",
        model="unknown model",
        description=f"scores read from {source.name}",
        score_definition=f"the score given in {source.name}",
    )


def report_from_results(
    path: Path,
    *,
    labels: Path | None = None,
    threshold: float | None = None,
    target_recall: float | None = None,
    upper_threshold: float | None = None,
    title: str | None = None,
) -> ReportData:
    """Build report data from saved results, optionally with new labels or thresholds.

    Without new labels or thresholds the saved decisions and calibration are used. With
    them, the threshold is recalibrated (or fixed) on the saved scores, the decisions are
    recomputed, and the PRISMA counts and methods paragraph are updated to match. Reasons
    are kept from the saved results.

    Raises:
        FileNotFoundError: If the results file does not exist.
        CalibrationError: If recalibration is needed but impossible.
    """
    csv_path = results_path(path)
    saved = load_saved_run(path) if (csv_path.parent / SUMMARY_FILE).is_file() else None
    recalibrate = any(
        option is not None for option in (labels, threshold, target_recall, upper_threshold)
    )
    if saved is not None and saved.calibration is not None and not recalibrate:
        records, calibration = saved.records, saved.calibration
    else:
        records, scores = read_scores(csv_path)
        reasons = {}
        if saved is not None:
            reasons = {r.id: r.assessment.reason for r in saved.records if r.assessment}
        else:
            try:
                reasons = {
                    r.id: r.assessment.reason for r in read_results(csv_path) if r.assessment
                }
            except ValueError:
                reasons = {}
        if labels is not None:
            apply_labels(records, read_labels(labels))
        known = [record.label for record in records]
        upper = DEFAULT_UPPER_THRESHOLD if upper_threshold is None else upper_threshold
        if threshold is not None:
            calibration = fixed_threshold(scores, known, threshold, upper)
        else:
            calibration = calibrate(scores, known, target_recall or DEFAULT_TARGET_RECALL, upper)
        for record, score in zip(records, scores, strict=True):
            record.assessment = Assessment(score, calibration.decide(score), reasons.get(record.id))
    info = saved.info if saved and saved.info else unknown_scorer(csv_path)
    prisma: PrismaCounts | None = None
    methods: str | None = None
    if saved is not None and saved.prisma is not None:
        decisions = [record.assessment.decision for record in records if record.assessment]
        prisma = prisma_counts(
            saved.prisma.identified,
            saved.prisma.duplicates_removed,
            decisions,
            saved.prisma.sources,
        )
        methods = methods_paragraph(
            info,
            calibration,
            prisma,
            __version__,
            deduplicated=saved.deduplicated,
            reasons=saved.reasons,
        )
    return ReportData(
        title=title or (saved.title if saved and saved.title else "Title and abstract screening"),
        records=records,
        calibration=calibration,
        info=info,
        tool_version=__version__,
        generated_at=datetime.now(),
        criteria=saved.criteria if saved else None,
        prisma=prisma,
        methods=methods,
        warnings=saved.warnings if saved else [],
        seconds=saved.seconds if saved else None,
    )
