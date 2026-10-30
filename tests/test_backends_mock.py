"""Tests for the deterministic mock backend and the backend factory."""

import pytest

from llm_literature_screener import Record
from llm_literature_screener.backends import BackendError, MockScorer, Scorer, create_scorer
from llm_literature_screener.backends.mock import terms
from llm_literature_screener.criteria import Criteria

CRITERIA = Criteria.from_dict(
    {
        "question": "Does supervised exercise therapy reduce pain in adults with chronic low "
        "back pain?",
        "study_designs": ["Randomized controlled trials"],
        "exclude": ["Studies in children or adolescents", "Animal experiments"],
    }
)
RELEVANT = Record(
    id="a",
    title="Supervised exercise therapy for chronic low back pain",
    abstract="A randomized controlled trial in adults; pain decreased after exercise therapy.",
)
OFF_TOPIC = Record(id="b", title="Dietary salt and blood pressure", abstract="A cohort study.")
EXCLUDED = Record(
    id="c",
    title="Exercise therapy for chronic back pain in children and adolescents",
    abstract="A randomized trial.",
)


def test_mock_scorer_satisfies_the_scorer_protocol() -> None:
    assert isinstance(MockScorer(), Scorer)
    assert MockScorer().info.backend == "mock"
    assert "not from a language model" in MockScorer().warnings[0]


def test_terms_are_stemmed_content_words() -> None:
    assert terms("Randomized trials in Adults, with the patients") == {
        "randomized",
        "trial",
        "adult",
        "patient",
    }


def test_relevant_records_score_above_off_topic_and_excluded_ones() -> None:
    relevant, off_topic, excluded = MockScorer().score([RELEVANT, OFF_TOPIC, EXCLUDED], CRITERIA)
    assert relevant > excluded
    assert relevant > off_topic
    assert relevant > 0 > off_topic


def test_mock_scores_are_deterministic_and_reported_in_input_order() -> None:
    progress: list[tuple[int, int]] = []
    first = MockScorer().score(
        [RELEVANT, OFF_TOPIC], CRITERIA, progress=lambda d, t: progress.append((d, t))
    )
    second = MockScorer().score([OFF_TOPIC, RELEVANT], CRITERIA)
    assert first == second[::-1]
    assert progress == [(2, 2)]


def test_mock_explanations_name_the_terms_found() -> None:
    relevant, off_topic, excluded = MockScorer().explain([RELEVANT, OFF_TOPIC, EXCLUDED], CRITERIA)
    assert relevant.startswith("Mentions") and "exercise" in relevant
    assert off_topic == "Mentions no protocol terms."
    assert "exclusion terms (adolescent, children)" in excluded


def test_create_scorer_builds_backends_by_name() -> None:
    assert isinstance(create_scorer("mock"), MockScorer)
    openai = create_scorer("openai", "my-model", base_url="http://localhost:9/v1")
    assert openai.info.model == "my-model"
    with pytest.raises(BackendError, match="needs --model"):
        create_scorer("openai")
    with pytest.raises(BackendError, match="Unknown backend"):
        create_scorer("gpt")
