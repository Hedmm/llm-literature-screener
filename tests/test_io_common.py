"""Tests for the parsing helpers shared by all formats."""

import pytest

from llm_literature_screener import Assessment, Decision, Record
from llm_literature_screener.io.common import (
    assign_ids,
    normalize_doi,
    normalize_pmid,
    normalize_title,
    output_values,
    parse_label,
    parse_year,
    split_authors,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.5555/ABC.123", "10.5555/abc.123"),
        ("https://doi.org/10.5555/abc.123", "10.5555/abc.123"),
        ("http://dx.doi.org/10.5555/abc.123", "10.5555/abc.123"),
        ("doi:10.5555/abc.123.", "10.5555/abc.123"),
        ("10.5555/abc.123 [doi]", "10.5555/abc.123"),
        ("(see 10.5555/abc.123)", "10.5555/abc.123"),
        ("10.1016/S0140-6736(19)30000-0", "10.1016/s0140-6736(19)30000-0"),
        ("https://doi.org/10.5555%2Fabc.123", "10.5555/abc.123"),
        ("not a doi", ""),
        ("", ""),
    ],
)
def test_normalize_doi(value: str, expected: str) -> None:
    assert normalize_doi(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("90000001", "90000001"), (" PMID: 123 ", "123"), ("WOS:000123", ""), ("", "")],
)
def test_normalize_pmid(value: str, expected: str) -> None:
    assert normalize_pmid(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2019", 2019),
        ("2019///", 2019),
        ("2022 Mar", 2022),
        ("Spring 1998", 1998),
        ("2020.0", 2020),
        ("12345", None),
        ("", None),
    ],
)
def test_parse_year(value: str, expected: int | None) -> None:
    assert parse_year(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("include", True),
        ("Included", True),
        ("YES", True),
        ("1", True),
        ("1.0", True),
        ("true", True),
        ("exclude", False),
        ("Excluded", False),
        ("no", False),
        ("0", False),
        ("maybe", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_label(value: str | None, expected: bool | None) -> None:
    assert parse_label(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Example, Ann; Sample, Ben", ["Example, Ann", "Sample, Ben"]),
        ("Example, Ann and Sample, Ben", ["Example, Ann", "Sample, Ben"]),
        ("Example A, Sample B", ["Example A, Sample B"]),
        ("", []),
    ],
)
def test_split_authors(value: str, expected: list[str]) -> None:
    assert split_authors(value) == expected


def test_normalize_title_ignores_case_accents_and_punctuation() -> None:
    assert normalize_title("Café-Based  Exercise: A Trial!") == "cafe based exercise a trial"
    assert normalize_title("[Translated title.]") == "translated title"


def test_assign_ids_fills_missing_and_repeated_ids() -> None:
    assert assign_ids(["A", "", "B", "A", ""]) == ["A", "2", "B", "A-2", "5"]


def test_assign_ids_does_not_reuse_an_id_taken_by_another_record() -> None:
    # The second record has no id; its position "1" is already used by the first record.
    assert assign_ids(["1", "", "1"]) == ["1", "2", "1-2"]
    assert assign_ids(["", "1"]) == ["1-2", "1"]


def test_output_values_are_prefixed_and_empty_before_scoring() -> None:
    record = Record(id="1")
    assert output_values(record) == []
    record.assessment = Assessment(score=0.0, decision=Decision.INCLUDE, reason="Fits.\nWell.")
    assert output_values(record) == [
        ("lls_score", "0.0"),
        ("lls_p_include", "0.5"),
        ("lls_decision", "include"),
        ("lls_reason", "Fits. Well."),
    ]
