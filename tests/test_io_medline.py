"""Tests for the PubMed MEDLINE/NBIB reader and writer."""

from pathlib import Path

from llm_literature_screener import Assessment, Decision
from llm_literature_screener.io.medline import read_medline, write_medline


def test_read_medline_joins_continuation_lines(fixtures: Path) -> None:
    first, second = read_medline(fixtures / "sample.nbib")

    assert first.id == first.pmid == "90000101"
    assert first.title == (
        "Synthetic trial of aquatic exercise for chronic low back pain in adults: a "
        "randomized controlled trial."
    )
    assert first.abstract.startswith("BACKGROUND: This synthetic abstract")
    assert first.abstract.endswith("RESULTS: Pain improved.")
    assert first.authors == ["Example, Ann", "Sample, Ben"]
    assert first.year == 2022
    assert first.journal == "Journal of Synthetic Examples"
    assert first.doi == "10.5555/synth.0101"
    assert first.label is None

    assert second.id == "90000102"
    assert second.abstract == ""
    assert second.authors == ["Test D"]
    assert second.journal == "Synth Rev"
    assert second.doi == "10.5555/synth.0102"


def test_write_medline_round_trip(fixtures: Path, tmp_path: Path) -> None:
    records = read_medline(fixtures / "sample.nbib")
    records[0].assessment = Assessment(
        score=2.0, decision=Decision.INCLUDE, reason="RCT in adults."
    )
    records[1].assessment = Assessment(score=-9.0, decision=Decision.EXCLUDE)
    out = tmp_path / "out.nbib"
    write_medline(records, out)
    text = out.read_text(encoding="utf-8")

    assert text.startswith("PMID- 90000101\n")
    assert "GN  - lls_decision: include" in text
    assert "GN  - lls_reason: RCT in adults." in text
    assert "AD  - Department of Synthetic Data, Example University." in text
    assert all(len(line) <= 82 for line in text.splitlines() if " " in line.strip())

    again = read_medline(out)
    assert [(r.id, r.title, r.abstract, r.doi) for r in again] == [
        (r.id, r.title, r.abstract, r.doi) for r in records
    ]
    write_medline(again, out)
    assert out.read_text(encoding="utf-8").count("lls_decision") == 0
