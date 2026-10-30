"""Tests for the RIS reader and writer."""

from pathlib import Path

from llm_literature_screener import Assessment, Decision, Record
from llm_literature_screener.io.ris import read_ris, write_ris


def test_read_ris_parses_standard_and_endnote_tags(fixtures: Path) -> None:
    records = read_ris(fixtures / "sample.ris")
    assert [record.id for record in records] == ["R1", "2", "R3"]

    first = records[0]
    assert first.title.startswith("Supervised exercise therapy")
    assert first.authors == ["Example, Ann", "Sample, Ben"]
    assert first.year == 2021
    assert first.journal == "Journal of Synthetic Examples"
    assert first.doi == "10.5555/synth.0001"
    assert first.abstract == (
        "Background: this abstract was written for software tests. "
        "This line continues the abstract."
    )
    assert first.label is True

    second = records[1]
    assert second.title == "An EndNote-style record with alternative tags"
    assert second.abstract == "Abstract stored in N2, as EndNote does."
    assert second.authors == ["Placeholder, Cara"]
    assert second.year == 2019
    assert second.journal == "Synthetic Reviews in Medicine"
    assert second.doi == "10.5555/synth.0002"
    assert second.pmid == "90000002"
    assert second.label is False

    third = records[2]
    assert third.abstract == ""
    assert third.year == 2018
    assert third.pmid == "90000003"
    assert third.label is None


def test_read_ris_accepts_byte_order_mark_and_windows_line_endings(
    fixtures: Path, tmp_path: Path
) -> None:
    text = (fixtures / "sample.ris").read_text(encoding="utf-8")
    path = tmp_path / "windows.ris"
    path.write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
    records = read_ris(path)
    assert [record.id for record in records] == ["R1", "2", "R3"]
    assert records[0].doi == "10.5555/synth.0001"


def test_read_ris_closes_a_record_without_end_tag(tmp_path: Path) -> None:
    path = tmp_path / "unterminated.ris"
    path.write_text("TY  - JOUR\nTI  - First\nTY  - JOUR\nTI  - Second\n", encoding="utf-8")
    assert [record.title for record in read_ris(path)] == ["First", "Second"]


def test_write_ris_round_trip_keeps_source_fields_and_adds_output(
    fixtures: Path, tmp_path: Path
) -> None:
    records = read_ris(fixtures / "sample.ris")
    for record in records:
        record.assessment = Assessment(score=1.5, decision=Decision.INCLUDE, reason="Adults, RCT.")
    out = tmp_path / "out.ris"
    write_ris(records, out)
    text = out.read_text(encoding="utf-8")

    assert text.startswith("TY  - JOUR\n")
    assert "TY  - CHAP" in text
    assert "KW  - exercise" in text
    assert "N1  - lls_decision: include" in text
    assert "N1  - lls_reason: Adults, RCT." in text
    assert text.count("ER  - ") == 3

    again = read_ris(out)
    assert [(r.id, r.title, r.doi, r.label) for r in again] == [
        (r.id, r.title, r.doi, r.label) for r in records
    ]

    # Writing a screened file again replaces the earlier output instead of repeating it.
    for record in again:
        record.assessment = Assessment(score=-3.0, decision=Decision.EXCLUDE)
    second = tmp_path / "second.ris"
    write_ris(again, second)
    text = second.read_text(encoding="utf-8")
    assert text.count("lls_decision") == 3
    assert "lls_decision: exclude" in text
    assert "lls_decision: include" not in text


def test_write_ris_describes_records_from_other_formats(tmp_path: Path) -> None:
    record = Record(
        id="X1",
        title="A record built in code",
        abstract="Abstract.",
        authors=["Example, Ann"],
        year=2020,
        journal="Journal of Synthetic Examples",
        doi="10.5555/synth.9001",
        pmid="90009001",
    )
    out = tmp_path / "new.ris"
    write_ris([record], out)
    (parsed,) = read_ris(out)
    assert (parsed.id, parsed.title, parsed.year, parsed.doi, parsed.pmid) == (
        "X1",
        "A record built in code",
        2020,
        "10.5555/synth.9001",
        "90009001",
    )
