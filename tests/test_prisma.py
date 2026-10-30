"""Tests for the PRISMA 2020 counts and the methods paragraph."""

from llm_literature_screener import Decision
from llm_literature_screener.backends.base import ScorerInfo
from llm_literature_screener.calibrate import calibrate, fixed_threshold
from llm_literature_screener.prisma import methods_paragraph, prisma_counts, prisma_markdown

HF_INFO = ScorerInfo(
    backend="hf",
    model="Qwen/Qwen2.5-7B-Instruct",
    description="Hugging Face transformers on cuda (bfloat16)",
    score_definition=(
        'the difference between the logits of "Yes" and "No" for the first answer token'
    ),
)
DECISIONS = [Decision.INCLUDE] * 3 + [Decision.UNCERTAIN] * 2 + [Decision.EXCLUDE] * 5


def test_counts_follow_the_decisions() -> None:
    counts = prisma_counts(12, 2, DECISIONS, sources=[("refs.ris", 12)])
    assert (counts.identified, counts.duplicates_removed, counts.assessed_by_tool) == (12, 2, 10)
    assert (counts.excluded_by_tool, counts.to_screen) == (5, 5)
    assert (counts.include, counts.uncertain) == (3, 2)
    assert counts.to_dict()["sources"] == {"refs.ris": 12}


def test_methods_paragraph_states_model_rule_target_and_human_decisions() -> None:
    scores = [5.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0]
    labels: list[bool | None] = [True, True, None, True, None, False, None, None, None, None]
    calibration = calibrate(scores, labels, target_recall=0.95)
    counts = prisma_counts(12, 2, DECISIONS)
    text = methods_paragraph(HF_INFO, calibration, counts, "0.1.0", reasons=True)
    assert "llm-literature-screener 0.1.0" in text
    assert "Qwen/Qwen2.5-7B-Instruct" in text
    assert "2 duplicate records were removed" in text
    assert 'logits of "Yes" and "No"' in text
    assert "highest score that kept at least 95% of the known inclusions" in text
    assert "4 records that the review team had already screened, 3 of them known inclusions" in text
    assert "3 of 3 known inclusions were kept (recall 100.0%" in text
    assert "(n = 5) were marked as ineligible by an automation tool" in text
    assert "one-line reason" in text
    assert "every inclusion and exclusion was decided by the reviewers" in text


def test_methods_paragraph_for_a_fixed_threshold_and_the_mock_backend() -> None:
    mock = ScorerInfo("mock", "mock-keyword-scorer", "keywords", "a keyword-overlap score")
    calibration = fixed_threshold([1.0, -1.0], [None, None], threshold=0.5)
    text = methods_paragraph(
        mock, calibration, prisma_counts(2, 0, DECISIONS[:2]), "0.1.0", deduplicated=False
    )
    assert "set by the review team at a score of 0.50" in text
    assert "mock scorer" in text
    assert "duplicate" not in text
    assert "recall" not in text


def test_prisma_markdown_lists_the_boxes() -> None:
    counts = prisma_counts(12, 2, DECISIONS, sources=[("refs.ris", 12)])
    text = prisma_markdown(counts, "Methods text.")
    assert "| Records identified | 12 |" in text
    assert "| Records identified from refs.ris | 12 |" in text
    assert "| Records marked as ineligible by automation tools | 5 |" in text
    assert "| Records screened by the review team | 5 |" in text
    assert "to be completed by the review team" in text
    assert text.rstrip().endswith("Methods text.")
