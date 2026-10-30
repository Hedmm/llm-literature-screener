"""Tests for Rayyan and Covidence exports and for format detection."""

import csv
from pathlib import Path

import pytest

from llm_literature_screener import Assessment, Decision
from llm_literature_screener.io import (
    UnknownFormatError,
    detect_format,
    read_records,
    write_records,
)
from llm_literature_screener.io.rayyan import label_from_notes, reviewer_decisions


def test_rayyan_reviewer_decisions_are_parsed_from_notes() -> None:
    notes = (
        'Imported | RAYYAN-INCLUSION: {"Alice"=>"Excluded", "Bob"=>"Included"} | RAYYAN-LABELS: x'
    )
    assert reviewer_decisions(notes) == {"Alice": "excluded", "Bob": "included"}
    assert reviewer_decisions("no decisions here") == {}


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        ('RAYYAN-INCLUSION: {"A"=>"Included"}', True),
        ('RAYYAN-INCLUSION: {"A"=>"Excluded", "B"=>"Excluded"}', False),
        ('RAYYAN-INCLUSION: {"A"=>"Excluded", "B"=>"Included"}', True),
        ('RAYYAN-INCLUSION: {"A"=>"Maybe"}', None),
        ('RAYYAN-INCLUSION: {"A"=>"Excluded", "B"=>"Maybe"}', None),
        ("", None),
    ],
)
def test_rayyan_label_rules(notes: str, expected: bool | None) -> None:
    assert label_from_notes(notes) is expected


def test_read_rayyan_export(fixtures: Path) -> None:
    records = read_records(fixtures / "rayyan.csv")
    assert [record.id for record in records] == [
        "rayyan-1001",
        "rayyan-1002",
        "rayyan-1003",
        "rayyan-1004",
        "rayyan-1005",
    ]
    assert [record.label for record in records] == [True, False, None, True, None]
    first = records[0]
    assert first.authors == ["Example, Ann", "Sample, Ben"]
    assert first.pmid == "90000501"
    assert first.doi == "10.5555/synth.0501"
    assert first.year == 2021


def test_write_rayyan_keeps_notes_and_adds_output(fixtures: Path, tmp_path: Path) -> None:
    records = read_records(fixtures / "rayyan.csv")
    for record in records:
        record.assessment = Assessment(score=0.5, decision=Decision.INCLUDE)
    out = tmp_path / "rayyan_screened.csv"
    write_records(records, out, "rayyan")
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["notes"].startswith("RAYYAN-INCLUSION:")
    assert rows[0]["lls_decision"] == "include"
    assert detect_format(out) == "rayyan"
    assert [record.label for record in read_records(out)] == [True, False, None, True, None]


def test_read_covidence_export(fixtures: Path) -> None:
    first, second = read_records(fixtures / "covidence.csv")
    assert (first.id, second.id) == ("#101", "#102")
    assert first.year == 2020
    assert first.pmid == "90000601"
    assert second.pmid == ""  # A Web of Science accession number is not a PMID.
    assert first.doi == "10.5555/synth.0601"
    assert first.label is None


def test_write_covidence(fixtures: Path, tmp_path: Path) -> None:
    records = read_records(fixtures / "covidence.csv")
    for record in records:
        record.assessment = Assessment(score=-2.0, decision=Decision.EXCLUDE)
    out = tmp_path / "covidence_screened.csv"
    write_records(records, out, "covidence")
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["Covidence #"] == "#102"
    assert rows[1]["lls_decision"] == "exclude"
    assert detect_format(out) == "covidence"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sample.ris", "ris"),
        ("sample.nbib", "medline"),
        ("sample.csv", "csv"),
        ("sample.tsv", "tsv"),
        ("rayyan.csv", "rayyan"),
        ("covidence.csv", "covidence"),
    ],
)
def test_detect_format_of_fixtures(fixtures: Path, name: str, expected: str) -> None:
    assert detect_format(fixtures / name) == expected


def test_detect_format_from_content_of_text_files(fixtures: Path, tmp_path: Path) -> None:
    pubmed = tmp_path / "pubmed-export.txt"
    pubmed.write_text((fixtures / "sample.nbib").read_text())
    ris = tmp_path / "export.txt"
    ris.write_text((fixtures / "sample.ris").read_text())
    tabbed = tmp_path / "savedrecs.txt"
    tabbed.write_text("TI\tAB\nA title\tAn abstract\n")
    assert detect_format(pubmed) == "medline"
    assert detect_format(ris) == "ris"
    assert detect_format(tabbed) == "tsv"
    assert read_records(tabbed)[0].abstract == "An abstract"


@pytest.mark.parametrize("name", ["records.xls", "records.docx", "notes.txt"])
def test_detect_format_rejects_unknown_files(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("plain text without structure\n")
    with pytest.raises(UnknownFormatError):
        detect_format(path)


def test_read_records_rejects_unknown_format_name(fixtures: Path) -> None:
    with pytest.raises(UnknownFormatError, match="Unknown format"):
        read_records(fixtures / "sample.csv", fmt="bibtex")
