"""Tests for the self-contained HTML report."""

import re
from datetime import datetime
from pathlib import Path

import pytest

from llm_literature_screener import Assessment, Record
from llm_literature_screener.backends.base import ScorerInfo
from llm_literature_screener.calibrate import calibrate, fixed_threshold
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.metrics import evaluate
from llm_literature_screener.prisma import prisma_counts
from llm_literature_screener.report import REPO_URL, ReportData, build_report, write_report

INFO = ScorerInfo("mock", "mock-keyword-scorer", "keyword overlap", "a keyword score")
CRITERIA = Criteria.from_dict({"question": "Does exercise help back pain?"})


def scored_records(labels: list[bool | None]) -> tuple[list[Record], list[float]]:
    scores = [6.0 - index for index in range(len(labels))]
    records = [
        Record(
            id=f"r{index}",
            title=f"Record number {index}",
            year=2020,
            journal="J Synth",
            label=label,
        )
        for index, label in enumerate(labels)
    ]
    return records, scores


def make_report(labels: list[bool | None], **options: object) -> tuple[ReportData, str]:
    records, scores = scored_records(labels)
    calibration = calibrate(scores, labels, target_recall=0.75, upper_threshold=3.5)
    for record, score in zip(records, scores, strict=True):
        record.assessment = Assessment(
            score, calibration.decide(score), reason=f"Reason {record.id}"
        )
    decisions = [record.assessment.decision for record in records if record.assessment]
    counts = prisma_counts(len(records) + 1, 1, decisions, sources=[("refs.ris", len(records) + 1)])
    data = ReportData(
        title="Synthetic review",
        records=records,
        calibration=calibration,
        info=INFO,
        tool_version="0.1.0",
        generated_at=datetime(2026, 1, 2, 3, 4),
        criteria=CRITERIA,
        prisma=counts,
        methods="Methods text.",
        warnings=["A backend warning."],
        seconds=1.5,
        **options,  # type: ignore[arg-type]
    )
    return data, build_report(data)


LABELS: list[bool | None] = [True, False, True, None, True, None, False, True, None, None]


def test_report_is_self_contained() -> None:
    _, page = make_report(LABELS)
    assert page.startswith("<!DOCTYPE html>")
    assert not re.search(r"<script[^>]+src=|<link[^>]+href=|<img|@import|url\(", page)
    assert re.findall(r'href=["\']([^"\']+)', page) == [REPO_URL]
    assert "<svg" in page and 'class="curve"' in page


def test_report_shows_summary_decisions_and_uncertain_records() -> None:
    data, page = make_report(LABELS)
    # Known inclusions score 6, 4, 2 and -1; keeping 75% of them needs the third, 2.0.
    assert data.calibration.threshold == 2.0
    assert "Left for people to screen" in page
    assert "3 of 4" in page
    uncertain = page.split("Uncertain records (2)")[1].split("</table>")[0]
    assert "Record number 3" in uncertain and "Reason r3" in uncertain
    assert "Record number 4" in uncertain
    assert "Record number 2" not in uncertain and "Record number 5" not in uncertain
    assert "A backend warning." in page
    assert "Records marked as ineligible by automation tools" in page
    assert "Methods text." in page
    assert "Does exercise help back pain?" in page
    assert "scored in 1.5 s" in page


def test_record_text_is_escaped() -> None:
    labels: list[bool | None] = [True, None, None]
    records, scores = scored_records(labels)
    records[1].title = "<script>alert('x')</script> & more"
    calibration = fixed_threshold(scores, labels, threshold=0.0, upper_threshold=10.0)
    for record, score in zip(records, scores, strict=True):
        record.assessment = Assessment(score, calibration.decide(score), reason="<b>bold</b>")
    page = build_report(ReportData("T", records, calibration, INFO, "0.1.0", datetime(2026, 1, 1)))
    assert "<script>alert" not in page
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt; &amp; more" in page
    assert "&lt;b&gt;bold&lt;/b&gt;" in page


def test_report_without_known_inclusions_explains_the_missing_curve() -> None:
    labels: list[bool | None] = [None, None, False]
    records, scores = scored_records(labels)
    calibration = fixed_threshold(scores, labels, threshold=4.5)
    for record, score in zip(records, scores, strict=True):
        record.assessment = Assessment(score, calibration.decide(score))
    page = build_report(ReportData("T", records, calibration, INFO, "0.1.0", datetime(2026, 1, 1)))
    assert "curve cannot be drawn" in page
    assert 'id="curve-data"' not in page
    assert "Known inclusions kept" in page and "unknown" in page
    assert "cannot be estimated" in page


def test_evaluation_section_and_long_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    _, scores = scored_records(LABELS)
    labels = [label is True for label in LABELS]
    evaluation = evaluate(scores, labels, threshold=0.0)
    monkeypatch.setattr("llm_literature_screener.report.MAX_LISTED", 1)
    _, page = make_report(LABELS, evaluation=evaluation)
    assert "Evaluation against complete labels" in page
    assert "WSS@95" in page
    assert "Showing the first 1 of 2" in page


def test_unscored_records_are_rejected(tmp_path: Path) -> None:
    data, _ = make_report(LABELS)
    data.records = [Record(id="x")]
    with pytest.raises(ValueError, match="no assessment"):
        write_report(data, tmp_path / "report.html")


def test_write_report(tmp_path: Path) -> None:
    data, page = make_report(LABELS)
    path = tmp_path / "report.html"
    write_report(data, path)
    assert path.read_text(encoding="utf-8") == page
