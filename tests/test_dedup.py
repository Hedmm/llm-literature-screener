"""Tests for duplicate detection."""

import csv
from pathlib import Path

from llm_literature_screener import Record
from llm_literature_screener.dedup import find_duplicates, title_key, write_dedup_report

TITLE = "Supervised exercise therapy for chronic low back pain"


def test_records_sharing_a_doi_are_merged() -> None:
    records = [
        Record(id="a", title="First copy", doi="10.5555/x.1"),
        Record(id="b", title="Unrelated", doi="10.5555/x.2"),
        Record(id="c", title="Second copy", doi="10.5555/x.1"),
    ]
    result = find_duplicates(records)
    assert [record.id for record in result.records] == ["a", "b"]
    assert result.removed == 1
    (match,) = result.matches
    assert (match.kept_id, match.removed_id, match.matched_id) == ("a", "c", "a")
    assert (match.reason, match.key) == ("doi", "10.5555/x.1")


def test_records_sharing_a_pmid_are_merged() -> None:
    records = [Record(id="a", pmid="90000001"), Record(id="b", pmid="90000001")]
    result = find_duplicates(records)
    assert [record.id for record in result.records] == ["a"]
    assert result.matches[0].reason == "pmid"


def test_title_and_year_matching_ignores_case_accents_and_punctuation() -> None:
    records = [
        Record(id="a", title=f"{TITLE}: a randomised trial", year=2020),
        Record(id="b", title=f"{TITLE.upper()} - A RANDOMISED TRIAL.", year=2020),
        Record(id="c", title=f"{TITLE}: a randomised trial", year=2021),
        Record(id="d", title="Exercice thérapeutique supervisé pour la lombalgie", year=2019),
        Record(id="e", title="Exercice therapeutique supervise pour la lombalgie", year=2019),
    ]
    result = find_duplicates(records)
    assert [record.id for record in result.records] == ["a", "c", "d"]
    assert {(m.kept_id, m.removed_id, m.reason) for m in result.matches} == {
        ("a", "b", "title+year"),
        ("d", "e", "title+year"),
    }


def test_short_titles_and_missing_years_are_never_matched() -> None:
    records = [
        Record(id="a", title="Editorial", year=2020),
        Record(id="b", title="Editorial", year=2020),
        Record(id="c", title=TITLE),
        Record(id="d", title=TITLE),
    ]
    assert title_key(records[0]) == ""
    assert title_key(records[2]) == ""
    assert find_duplicates(records).removed == 0


def test_matches_are_transitive_and_reported_with_the_direct_link() -> None:
    records = [
        Record(id="a", title=TITLE, year=2020, doi="10.5555/x.1"),
        Record(id="b", title="Another title with enough words here", year=2020, pmid="9001"),
        Record(id="c", title="Another title with enough words here", year=2020, doi="10.5555/x.1"),
    ]
    result = find_duplicates(records)
    assert [record.id for record in result.records] == ["a"]
    by_removed = {match.removed_id: match for match in result.matches}
    assert (by_removed["c"].matched_id, by_removed["c"].reason) == ("a", "doi")
    assert (by_removed["b"].matched_id, by_removed["b"].reason) == ("c", "title+year")
    assert {match.kept_id for match in result.matches} == {"a"}


def test_kept_record_is_completed_from_its_duplicates() -> None:
    kept = Record(id="a", title=TITLE, doi="10.5555/x.1", label=None)
    duplicate = Record(
        id="b",
        title=TITLE,
        abstract="The full abstract.",
        authors=["Example, Ann"],
        year=2020,
        journal="Journal of Synthetic Examples",
        doi="10.5555/x.1",
        pmid="9001",
        label=True,
    )
    (merged,) = find_duplicates([kept, duplicate]).records
    assert merged.id == "a"
    assert merged.abstract == "The full abstract."
    assert (merged.year, merged.pmid, merged.authors) == (2020, "9001", ["Example, Ann"])
    assert merged.label is True
    # The input records are not modified.
    assert kept.abstract == ""
    assert kept.label is None


def test_conflicting_labels_keep_the_inclusion() -> None:
    records = [
        Record(id="a", doi="10.5555/x.1", label=False),
        Record(id="b", doi="10.5555/x.1", label=True),
    ]
    assert find_duplicates(records).records[0].label is True


def test_matching_keys_can_be_switched_off() -> None:
    records = [Record(id="a", doi="10.5555/x.1"), Record(id="b", doi="10.5555/x.1")]
    assert find_duplicates(records, by_doi=False).removed == 0


def test_dedup_report_lists_every_removed_record(tmp_path: Path) -> None:
    records = [
        Record(id="a", title="Kept title", doi="10.5555/x.1"),
        Record(id="b", title="Removed title", doi="10.5555/x.1"),
    ]
    result = find_duplicates(records)
    path = tmp_path / "duplicates.csv"
    write_dedup_report(result, records, path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "kept_id": "a",
            "removed_id": "b",
            "matched_id": "a",
            "reason": "doi",
            "key": "10.5555/x.1",
            "kept_title": "Kept title",
            "removed_title": "Removed title",
        }
    ]
