"""Model backends that score records: transformers, OpenAI-compatible servers and a mock.

Use :func:`create_scorer` to build one by name. The transformers backend is imported only
when it is requested, so PyTorch is needed only for ``backend="hf"``.
"""

from __future__ import annotations

from llm_literature_screener.backends.base import (
    BackendError,
    ProgressCallback,
    Scorer,
    ScorerInfo,
)
from llm_literature_screener.backends.mock import MockScorer
from llm_literature_screener.backends.openai_compat import DEFAULT_BASE_URL, OpenAICompatScorer
from llm_literature_screener.prompt import MAX_ABSTRACT_WORDS

BACKENDS: dict[str, str] = {
    "hf": "Hugging Face transformers, local model (needs the hf extra)",
    "openai": "OpenAI-compatible server: vLLM, llama.cpp server, Ollama",
    "mock": "Deterministic keyword scorer for tests and demos (no model)",
}
"""Backend names accepted by :func:`create_scorer`."""

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

__all__ = [
    "BACKENDS",
    "DEFAULT_BASE_URL",
    "DEFAULT_HF_MODEL",
    "BackendError",
    "MockScorer",
    "OpenAICompatScorer",
    "ProgressCallback",
    "Scorer",
    "ScorerInfo",
    "create_scorer",
]


def create_scorer(
    backend: str,
    model: str | None = None,
    *,
    device: str = "auto",
    dtype: str = "auto",
    token_budget: int = 24000,
    max_batch_size: int = 64,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    concurrency: int = 8,
    max_abstract_words: int = MAX_ABSTRACT_WORDS,
) -> Scorer:
    """Create a scorer by backend name.

    Args:
        backend: ``hf``, ``openai`` or ``mock``.
        model: Model id. Defaults to ``Qwen/Qwen2.5-7B-Instruct`` for ``hf``; required for
            ``openai``; ignored for ``mock``.
        device: Torch device for ``hf``.
        dtype: Precision for ``hf``.
        token_budget: Padded tokens per batch for ``hf``.
        max_batch_size: Prompts per batch for ``hf``.
        base_url: API base URL for ``openai``.
        api_key: API key for ``openai``, if the server needs one.
        concurrency: Parallel requests for ``openai``.
        max_abstract_words: Abstracts are cut to this many words.

    Raises:
        BackendError: If the backend is unknown or cannot be created.
    """
    if backend == "mock":
        return MockScorer()
    if backend == "hf":
        from llm_literature_screener.backends.hf import HFScorer

        return HFScorer(
            model or DEFAULT_HF_MODEL,
            device=device,
            dtype=dtype,
            token_budget=token_budget,
            max_batch_size=max_batch_size,
            max_abstract_words=max_abstract_words,
        )
    if backend == "openai":
        if not model:
            raise BackendError(
                "The openai backend needs --model with the model name the server uses."
            )
        return OpenAICompatScorer(
            model,
            base_url=base_url,
            api_key=api_key,
            concurrency=concurrency,
            max_abstract_words=max_abstract_words,
        )
    raise BackendError(f"Unknown backend {backend!r}; use one of: {', '.join(BACKENDS)}.")
