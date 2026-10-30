"""Tests for the screening pipeline on the bundled synthetic example."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from llm_literature_screener import Decision, Record
from llm_literature_screener.backends import MockScorer
from llm_literature_screener.backends.base import ProgressCallback, ScorerInfo
from llm_literature_screener.calibrate import CalibrationError
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.io import apply_labels, read_labels, read_records, read_results
from llm_literature_screener.pipeline import (
    ScreeningOptions,
    SourceFile,
    combine_sources,
    load_saved_run,
    load_source,
    report_from_results,
    run_screening,
    write_outputs,
)
from tests.conftest import EXAMPLES


def example_records() -> list[Record]:
    records = read_records(EXAMPLES / "records.ris")
    apply_labels(records, read_labels(EXAMPLES / "labels.csv"))
    return records


CRITERIA = Criteria.from_yaml(EXAMPLES / "protocol.yaml")


def test_run_screening_deduplicates_scores_calibrates_and_decides() -> None:
    stages: list[tuple[str, int, int]] = []
    run = run_screening(
        example_records(),
        CRITERIA,
        MockScorer(),
        ScreeningOptions(reasons=True),
        progress=lambda stage, done, total: stages.append((stage, done, total)),
        sources=[("records.ris", 26)],
    )
    assert run.prisma.identified == 26
    assert run.prisma.duplicates_removed == 1
    assert len(run.records) == run.prisma.assessed_by_tool == 25
    assert run.dedup is not None and run.dedup.matches[0].removed_id == "EX23"
    calibration = run.calibration
    assert calibration.method == "target_recall"
    assert calibration.known_inclusions == 7
    assert calibration.recall == 1.0
    assert calibration.warnings  # fewer than 10 known inclusions
    flagged = 0
    for record in run.records:
        assert record.assessment is not None
        assert record.assessment.decision is calibration.decide(record.assessment.score)
        has_reason = record.assessment.reason is not None
        assert has_reason == (record.assessment.decision is not Decision.EXCLUDE)
        flagged += has_reason
    assert run.prisma.to_screen == flagged
    assert ("scoring", 25, 25) in stages
    assert ("reasons", flagged, flagged) in stages
    assert run.title == CRITERIA.title
    assert "mock" in run.warnings[0]
    assert "highest score that kept at least 95%" in run.methods


class ExplodingScorer(MockScorer):
    """A scorer that must not be called."""

    def score(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[float]:
        raise AssertionError("the model should not run")


def test_missing_known_inclusions_stop_the_run_before_scoring() -> None:
    records = read_records(EXAMPLES / "records.ris")
    with pytest.raises(CalibrationError, match="--labels"):
        run_screening(records, CRITERIA, ExplodingScorer())
    with pytest.raises(ValueError, match="no records"):
        run_screening([], CRITERIA, MockScorer())


def test_fixed_threshold_without_labels_and_without_dedup() -> None:
    records = read_records(EXAMPLES / "records.ris")
    run = run_screening(
        records, CRITERIA, MockScorer(), ScreeningOptions(threshold=0.0, deduplicate=False)
    )
    assert run.dedup is None
    assert len(run.records) == 26
    assert run.calibration.method == "fixed"
    assert run.calibration.recall is None
    assert "set by the review team" in run.methods
    assert all(
        record.assessment is not None and record.assessment.decision is not Decision.UNCERTAIN
        for record in run.records
    )


def test_write_outputs_writes_every_file(tmp_path: Path) -> None:
    source = load_source(EXAMPLES / "records.ris")
    records, source_of = combine_sources([source])
    apply_labels(records, read_labels(EXAMPLES / "labels.csv"))
    run = run_screening(records, CRITERIA, MockScorer(), source_of=source_of)
    paths = write_outputs(run, tmp_path, [source])
    assert {path.name for path in tmp_path.iterdir()} == {
        "screening_results.csv",
        "records_screened.ris",
        "prisma.md",
        "calibration.json",
        "duplicates.csv",
        "criteria.yaml",
        "report.html",
        "run.json",
    }
    assert paths["records:records.ris"].read_text().count("N1  - lls_decision:") == 25
    assert "EX23" in paths["duplicates"].read_text()
    summary = json.loads(paths["summary"].read_text())
    assert summary["prisma"]["identified"] == 26
    assert summary["scorer"]["backend"] == "mock"
    assert summary["files"]["report"] == "report.html"
    assert Criteria.from_yaml(paths["criteria"]) == CRITERIA
    assert len(read_results(paths["results"])) == 25


def test_several_source_files_get_prefixed_ids_and_separate_outputs(tmp_path: Path) -> None:
    first = load_source(EXAMPLES / "records.ris")
    copy = tmp_path / "second.ris"
    copy.write_text((EXAMPLES / "records.ris").read_text())
    second = load_source(copy)
    records, source_of = combine_sources([first, second])
    assert records[0].id == "records.ris:EX01"
    assert source_of["second.ris:EX01"] == 1
    run = run_screening(
        records, CRITERIA, MockScorer(), ScreeningOptions(threshold=0.0), source_of=source_of
    )
    # Every record of the second file duplicates one of the first.
    assert run.prisma.duplicates_removed == 27
    paths = write_outputs(run, tmp_path / "out", [first, second])
    assert paths["records:second.ris"].read_text() == ""
    assert paths["records:records.ris"].read_text().count("ER  - ") == 25


def test_sources_with_the_same_file_name_are_numbered(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for folder in ("a", "b"):
        (tmp_path / folder / "refs.ris").write_text("TY  - JOUR\nTI  - Title\nER  - \n")
    sources = [load_source(tmp_path / folder / "refs.ris") for folder in ("a", "b")]
    records, _ = combine_sources(sources)
    assert [record.id for record in records] == ["1-refs.ris:1", "2-refs.ris:1"]
    assert isinstance(sources[0], SourceFile)


def test_saved_run_reloads_and_recalibrates(tmp_path: Path) -> None:
    run = run_screening(example_records(), CRITERIA, MockScorer(), ScreeningOptions(reasons=True))
    write_outputs(run, tmp_path)
    saved = load_saved_run(tmp_path)
    assert len(saved.records) == 25
    assert saved.calibration == run.calibration
    assert saved.info == run.info
    assert saved.criteria == CRITERIA
    assert saved.reasons is True

    same = report_from_results(tmp_path)
    assert same.calibration == run.calibration
    assert same.prisma == run.prisma

    stricter = report_from_results(tmp_path, threshold=1.0, title="Stricter")
    assert stricter.title == "Stricter"
    assert stricter.calibration.method == "fixed"
    assert stricter.prisma is not None
    assert stricter.prisma.identified == 26
    assert stricter.prisma.excluded_by_tool > run.prisma.excluded_by_tool
    assert stricter.methods is not None and "set by the review team" in stricter.methods
    reasons = [r.assessment.reason for r in stricter.records if r.assessment]
    assert any(reasons)


def test_report_from_a_plain_scores_file(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "id,title,score,label\na,First,2.0,include\nb,Second,-1.0,exclude\nc,Third,0.5,\n"
    )
    data = report_from_results(scores)
    assert data.info == ScorerInfo(
        "unknown", "unknown model", "scores read from scores.csv", "the score given in scores.csv"
    )
    assert data.calibration.threshold == 2.0
    assert data.prisma is None
    with pytest.raises(FileNotFoundError):
        load_saved_run(tmp_path / "missing")
