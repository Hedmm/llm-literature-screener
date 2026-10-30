"""Tests for the review protocol and the prompts built from it."""

from pathlib import Path
from typing import Any

import pytest

from llm_literature_screener import Record
from llm_literature_screener.criteria import Criteria, CriteriaError, YearRange
from llm_literature_screener.prompt import (
    SYSTEM_PROMPT,
    format_abstract,
    reason_messages,
    screening_messages,
    screening_prompt,
)

FULL_PROTOCOL = """\
title: Exercise for chronic low back pain
question: Does supervised exercise reduce pain in adults with chronic low back pain?
population: Adults with non-specific low back pain for 12 weeks or longer.
exposure: Supervised exercise therapy.
comparison: Usual care or no treatment.
outcomes: [Pain intensity, Disability]
study_designs: Randomized controlled trials
inclusion_criteria:
  - Adults aged 18 or older
  - Supervised sessions
exclude:
  - Acute low back pain
languages: [English, German]
years: 2000-2024
"""


def test_full_protocol_is_parsed_and_rendered_in_a_fixed_order(tmp_path: Path) -> None:
    path = tmp_path / "protocol.yaml"
    path.write_text(FULL_PROTOCOL, encoding="utf-8")
    criteria = Criteria.from_yaml(path)
    assert criteria.intervention == "Supervised exercise therapy."
    assert criteria.study_designs == ("Randomized controlled trials",)
    assert criteria.years == YearRange(2000, 2024)
    assert criteria.render() == (
        "Review question: Does supervised exercise reduce pain in adults with chronic low "
        "back pain?\n"
        "Population: Adults with non-specific low back pain for 12 weeks or longer.\n"
        "Intervention or exposure: Supervised exercise therapy.\n"
        "Comparator: Usual care or no treatment.\n"
        "Outcomes: Pain intensity; Disability\n"
        "Study designs: Randomized controlled trials\n"
        "Languages: English, German\n"
        "Publication years: 2000 to 2024\n"
        "Include:\n"
        "- Adults aged 18 or older\n"
        "- Supervised sessions\n"
        "Exclude: Acute low back pain"
    )


def test_protocol_round_trips_through_yaml() -> None:
    criteria = Criteria.from_yaml_text(FULL_PROTOCOL)
    assert Criteria.from_yaml_text(criteria.to_yaml()) == criteria
    assert criteria.to_dict()["years"] == "2000-2024"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2000-2024", YearRange(2000, 2024)),
        ("2000–2024", YearRange(2000, 2024)),
        ("2000-", YearRange(2000, None)),
        ("-2024", YearRange(None, 2024)),
        (2000, YearRange(2000, None)),
        ("2010", YearRange(2010, None)),
        ({"from": 2000, "to": 2024}, YearRange(2000, 2024)),
        ({"to": 2024}, YearRange(None, 2024)),
        ([2000, 2024], YearRange(2000, 2024)),
        (None, None),
    ],
)
def test_year_ranges(value: Any, expected: YearRange | None) -> None:
    assert Criteria.from_dict({"question": "Q?", "years": value}).years == expected


def test_year_range_descriptions() -> None:
    assert YearRange(2000, None).describe() == "2000 or later"
    assert YearRange(None, 2024).describe() == "2024 or earlier"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"population": "Adults"}, "needs a 'question'"),
        ({"question": "Q?", "populaton": "Adults"}, "Unknown protocol field 'populaton'"),
        ({"question": "Q?", "intervention": "A", "exposure": "B"}, "more than once"),
        ({"question": ["a", "b"]}, "must be text"),
        ({"question": "Q?", "include": {"a": 1}}, "text or a list"),
        ({"question": "Q?", "include": [["nested"]]}, "must be text"),
        ({"question": "Q?", "years": "recent"}, "four-digit years"),
        ({"question": "Q?", "years": [2024, 2000]}, "starts after it ends"),
        ({"question": "Q?", "years": {"since": 2000}}, "'from' and 'to'"),
        ({"question": "Q?", "years": True}, "range such as"),
    ],
)
def test_invalid_protocols_are_rejected_with_clear_messages(
    data: dict[str, Any], message: str
) -> None:
    with pytest.raises(CriteriaError, match=message):
        Criteria.from_dict(data)


def test_yaml_that_is_not_a_protocol_is_rejected() -> None:
    with pytest.raises(CriteriaError, match="mapping"):
        Criteria.from_yaml_text("- just\n- a list\n")
    with pytest.raises(CriteriaError, match="not valid YAML"):
        Criteria.from_yaml_text("question: [unclosed\n")


# The case study's prompt function, copied verbatim apart from the names, as the reference.
CASE_STUDY_MAX_WORDS = 350


def case_study_prompt(review: dict[str, str], title: Any, abstract: Any) -> str:
    words = str(abstract).split() if isinstance(abstract, str) and abstract.strip() else []
    abstract = (
        " ".join(words[:CASE_STUDY_MAX_WORDS])
        + (" ..." if len(words) > CASE_STUDY_MAX_WORDS else "")
        if words
        else "not available"
    )
    return (
        f"Review question: {review['question']}\nInclude: {review['include']}\n"
        f"Exclude: {review['exclude']}\n\n"
        f"Title: {title if isinstance(title, str) else 'not available'}\nAbstract: {abstract}\n\n"
        "Should this record go to full-text review? Answer Yes or No."
    )


WOLTERS = {
    "question": "Coronary heart disease, heart failure, and the risk of dementia.",
    "include": "Longitudinal studies (prospective or retrospective cohort, nested case-control) in "
    "community or population-based samples of adults, in which coronary heart disease "
    "(myocardial infarction, angina, coronary artery disease) or heart failure is the exposure "
    "and incident dementia (all-cause, Alzheimer's disease or vascular dementia) is the outcome.",
    "exclude": "Cross-sectional studies, studies with only cognitive test scores or cognitive "
    "decline and no dementia diagnosis, reviews, editorials, case reports, and animal studies.",
}


@pytest.mark.parametrize(
    ("title", "abstract"),
    [
        ("Heart failure and incident dementia in a synthetic cohort", "A short abstract."),
        ("A title", " ".join(f"word{i}" for i in range(400))),
        ("A title", " ".join(f"word{i}" for i in range(350))),
        ("A title without abstract", None),
        (None, "An abstract without a title."),
    ],
)
def test_prompt_is_identical_to_the_case_study(title: str | None, abstract: str | None) -> None:
    criteria = Criteria.from_dict(WOLTERS)
    record = Record(id="1", title=title or "", abstract=abstract or "")
    assert screening_prompt(criteria, record) == case_study_prompt(WOLTERS, title, abstract)


def test_screening_messages_use_the_validated_system_prompt() -> None:
    criteria = Criteria.from_dict(WOLTERS)
    system, user = screening_messages(criteria, Record(id="1", title="T"))
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert user["content"].endswith("Answer Yes or No.")


def test_reason_messages_ask_for_one_sentence() -> None:
    criteria = Criteria.from_dict(WOLTERS)
    _, user = reason_messages(criteria, Record(id="1", title="T", abstract="A"))
    assert "Title: T\nAbstract: A" in user["content"]
    assert user["content"].endswith("based only on the title and abstract.")


def test_format_abstract_cuts_long_abstracts() -> None:
    assert format_abstract("one  two\nthree", max_words=2) == "one two ..."
    assert format_abstract("   ") == "not available"
