"""Tests for the labels file reader and the flat results table."""

import logging
from pathlib import Path

import pytest

from llm_literature_screener import Assessment, Decision, Record
from llm_literature_screener.io import (
    apply_labels,
    read_labels,
    read_records,
    read_results,
    read_scores,
    write_results,
)
from llm_literature_screener.io.labels import LabelFileError
from llm_literature_screener.io.results import ResultsFileError


def test_read_labels_skips_unrecognized_values(
    fixtures: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        entries = read_labels(fixtures / "labels.csv")
    assert [entry.label for entry in entries] == [True, False, True, False]
    assert entries[1].doi == "10.5555/synth.0002"
    assert entries[2].title == "a synthetic record without an abstract"
    assert "unrecognized label" in caplog.text


def test_apply_labels_matches_by_id_doi_and_title(fixtures: Path) -> None:
    records = read_records(fixtures / "sample.ris")
    for record in records:
        record.label = None
    report = apply_labels(records, read_labels(fixtures / "labels.csv"))
    assert [record.label for record in records] == [True, False, True]
    assert (report.entries, report.matched, report.changed) == (4, 3, 0)
    assert [entry.id for entry in report.unmatched] == ["X99"]


def test_apply_labels_counts_replaced_labels(fixtures: Path) -> None:
    records = [Record(id="R1", label=False)]
    report = apply_labels(records, read_labels(fixtures / "labels.csv"))
    assert records[0].label is True
    assert report.changed == 1


def test_labels_file_needs_label_and_identifier_columns(tmp_path: Path) -> None:
    no_label = tmp_path / "no_label.csv"
    no_label.write_text("id,score\n1,0.5\n")
    with pytest.raises(LabelFileError, match="no label column"):
        read_labels(no_label)
    no_id = tmp_path / "no_id.csv"
    no_id.write_text("label,comment\ninclude,x\n")
    with pytest.raises(LabelFileError, match="id, doi, pmid or title"):
        read_labels(no_id)
    custom = tmp_path / "custom.csv"
    custom.write_text("id,verdict\n1,include\n")
    assert read_labels(custom, label_column="verdict")[0].label is True


def _assessed(record_id: str, score: float, decision: Decision, label: bool | None) -> Record:
    return Record(
        id=record_id,
        title=f"Title {record_id}",
        authors=["Example, Ann", "Sample, Ben"],
        year=2020,
        label=label,
        assessment=Assessment(score=score, decision=decision, reason=f"Reason {record_id}"),
    )


def test_results_table_is_ranked_and_reads_back(tmp_path: Path) -> None:
    records = [
        _assessed("a", -4.0, Decision.EXCLUDE, False),
        _assessed("b", 2.5, Decision.INCLUDE, True),
        _assessed("c", -1.0, Decision.UNCERTAIN, None),
    ]
    path = tmp_path / "screening_results.csv"
    write_results(records, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "rank,id,title,authors,year,journal,doi,pmid,label,score,p_include,decision,reason"
    )
    assert lines[1].startswith('1,b,Title b,"Example, Ann; Sample, Ben",2020,')

    parsed = read_results(path)
    assert [record.id for record in parsed] == ["b", "c", "a"]
    assert [record.label for record in parsed] == [True, None, False]
    assert [record.assessment.decision for record in parsed if record.assessment] == [
        Decision.INCLUDE,
        Decision.UNCERTAIN,
        Decision.EXCLUDE,
    ]
    assert parsed[0].authors == ["Example, Ann", "Sample, Ben"]
    assert parsed[0].assessment is not None
    assert parsed[0].assessment.reason == "Reason b"

    scored, scores = read_scores(path)
    assert [record.id for record in scored] == ["b", "c", "a"]
    assert scores == [2.5, -1.0, -4.0]


def test_write_results_requires_assessments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no assessment"):
        write_results([Record(id="x")], tmp_path / "results.csv")


def test_results_reader_reports_bad_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ResultsFileError, match="empty"):
        read_scores(empty)
    missing = tmp_path / "missing.csv"
    missing.write_text("id,title\n1,x\n")
    with pytest.raises(ResultsFileError, match="score"):
        read_scores(missing)
    bad_score = tmp_path / "bad.csv"
    bad_score.write_text("id,score\n1,high\n")
    with pytest.raises(ResultsFileError, match="not a number"):
        read_scores(bad_score)
    bad_decision = tmp_path / "decision.csv"
    bad_decision.write_text("id,score,decision\n1,0.5,keep\n")
    with pytest.raises(ResultsFileError, match="unknown decision"):
        read_results(bad_decision)
