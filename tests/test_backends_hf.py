"""Tests for the Hugging Face transformers backend.

The batching tests need nothing extra. The model tests need PyTorch and transformers and are
skipped without them. Most of them use tiny random models built offline; one downloads a
tiny model from the Hugging Face Hub and is skipped when it cannot be downloaded.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from llm_literature_screener import Record
from llm_literature_screener.backends.base import BackendError, Scorer
from llm_literature_screener.backends.hf import token_budget_batches
from llm_literature_screener.criteria import Criteria
from tests.tiny_models import build_tiny_model

CRITERIA = Criteria.from_dict(
    {
        "question": "Does exercise therapy reduce chronic low back pain in adults?",
        "include": ["Randomized trial in adults"],
        "exclude": ["children"],
    }
)
RECORDS = [
    Record(id="1", title="Exercise therapy for chronic low back pain", abstract="trial " * 120),
    Record(id="2", title="Heart disease and dementia", abstract="A cohort."),
    Record(id="3"),
    Record(id="4", title="Treatment of anxiety disorder in children", abstract="patients " * 40),
]


def test_batches_are_sorted_by_length_and_fit_the_budget() -> None:
    lengths = [50, 10, 30, 10, 40]
    batches = token_budget_batches(lengths, token_budget=90, max_batch_size=64)
    assert batches == [[1, 3, 2], [4], [0]]
    for batch in batches:
        assert len(batch) * max(lengths[i] for i in batch) <= 90
    assert token_budget_batches(lengths, token_budget=60) == [[1, 3], [2], [4], [0]]
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))


def test_batches_respect_the_maximum_batch_size_and_oversized_prompts() -> None:
    assert token_budget_batches([1] * 5, token_budget=100, max_batch_size=2) == [
        [0, 1],
        [2, 3],
        [4],
    ]
    assert token_budget_batches([500, 20], token_budget=100) == [[1], [0]]
    assert token_budget_batches([]) == []


@pytest.fixture(scope="module")
def tiny_model_dirs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("tiny-models")
    yield {
        "llama": build_tiny_model(root / "llama", "llama"),
        "gpt2": build_tiny_model(root / "gpt2", "gpt2"),
        "plain": build_tiny_model(root / "plain", "gpt2", chat_template=False),
    }


def make_scorer(path: Path, **options: Any) -> Any:
    from llm_literature_screener.backends.hf import HFScorer

    return HFScorer(str(path), device="cpu", **options)


@pytest.mark.parametrize("name", ["llama", "gpt2", "plain"])
def test_scores_do_not_depend_on_batching_or_padding(
    tiny_model_dirs: dict[str, Path], name: str
) -> None:
    alone = make_scorer(tiny_model_dirs[name], token_budget=1)
    together = make_scorer(tiny_model_dirs[name], token_budget=10**6)
    single = alone.score(RECORDS, CRITERIA)
    batched = together.score(RECORDS, CRITERIA)
    assert single == pytest.approx(batched, abs=1e-4)
    assert len(set(single)) == len(single)


def test_hf_scorer_reports_progress_and_describes_itself(tiny_model_dirs: dict[str, Path]) -> None:
    scorer = make_scorer(tiny_model_dirs["llama"], token_budget=400)
    progress: list[tuple[int, int]] = []
    scorer.score(RECORDS, CRITERIA, progress=lambda done, total: progress.append((done, total)))
    assert progress[-1] == (4, 4)
    assert len(progress) > 1
    assert scorer.uses_chat_template
    assert scorer.info.backend == "hf"
    assert scorer.info.description == "Hugging Face transformers on cpu (float32)"
    assert scorer.warnings == []
    assert isinstance(scorer, Scorer)


def test_model_without_chat_template_gets_a_plain_prompt_and_a_warning(
    tiny_model_dirs: dict[str, Path],
) -> None:
    scorer = make_scorer(tiny_model_dirs["plain"])
    assert not scorer.uses_chat_template
    assert "no chat template" in scorer.warnings[0]


def test_hf_scorer_generates_reasons(tiny_model_dirs: dict[str, Path]) -> None:
    scorer = make_scorer(tiny_model_dirs["llama"])
    scorer.max_new_tokens = 5
    reasons = scorer.explain(RECORDS[:2], CRITERIA)
    assert len(reasons) == 2
    assert all(isinstance(reason, str) for reason in reasons)


def test_invalid_options_and_missing_models_raise_backend_errors(
    tiny_model_dirs: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(BackendError, match="Unknown dtype"):
        make_scorer(tiny_model_dirs["llama"], dtype="int4")
    with pytest.raises(BackendError, match="Could not load"):
        make_scorer(tmp_path / "missing")


@pytest.mark.network
def test_hf_scorer_with_a_tiny_model_from_the_hub() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from llm_literature_screener.backends.hf import HFScorer

    try:
        scorer = HFScorer("sshleifer/tiny-gpt2", device="cpu")
    except BackendError as error:
        pytest.skip(f"The tiny model could not be downloaded: {error}")
    scores = scorer.score(RECORDS[:2], CRITERIA)
    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)
