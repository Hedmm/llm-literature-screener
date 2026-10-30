"""Hugging Face transformers backend: Yes/No logit scoring with a local model.

This is the method of the validated case study. Each prompt is built with the model's chat
template; the score is logit(Yes) - logit(No) at the position of the first answer token, so
no text is generated. Prompts are sorted by token length and grouped into batches that fit a
budget of padded tokens, which keeps padding small and the GPU busy.

PyTorch and transformers are imported only when a scorer is created, so the rest of the
package works without them.
"""

from __future__ import annotations

import copy
import inspect
import logging
import re
from collections.abc import Sequence
from typing import Any

from llm_literature_screener.backends.base import BackendError, ProgressCallback, ScorerInfo
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.prompt import (
    MAX_ABSTRACT_WORDS,
    ChatMessage,
    reason_messages,
    screening_messages,
)
from llm_literature_screener.record import Record

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TOKEN_BUDGET = 24000
DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_MAX_NEW_TOKENS = 60
DTYPES = ("auto", "bfloat16", "float16", "float32")


def token_budget_batches(
    lengths: Sequence[int],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> list[list[int]]:
    """Group prompts into batches of similar length that fit a budget of padded tokens.

    Prompts are sorted by length (ties keep their input order), so every batch is padded to
    the length of its last prompt. A batch grows while ``batch size × longest length`` stays
    within ``token_budget`` and the batch has fewer than ``max_batch_size`` prompts. A prompt
    longer than the budget is put in a batch of its own.

    Returns:
        Batches of prompt indices, shortest prompts first.
    """
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    batches: list[list[int]] = []
    start = 0
    while start < len(order):
        size = 1
        while (
            start + size < len(order)
            and size < max_batch_size
            and (size + 1) * lengths[order[start + size]] <= token_budget
        ):
            size += 1
        batches.append(order[start : start + size])
        start += size
    return batches


def _version(module: Any) -> tuple[int, int]:
    major, minor, *_ = [*(int(part) for part in re.findall(r"\d+", str(module.__version__))), 0, 0]
    return major, minor


class HFScorer:
    """Score records with a local causal language model through Hugging Face transformers.

    Args:
        model: Model id on the Hugging Face Hub, or a local directory.
        device: ``auto`` (CUDA, then Apple MPS, then CPU) or a torch device such as ``cpu``
            or ``cuda:1``.
        dtype: ``auto`` (bfloat16 on CUDA, float16 on MPS, float32 on CPU), ``bfloat16``,
            ``float16`` or ``float32``.
        token_budget: Maximum number of padded tokens in one batch.
        max_batch_size: Maximum number of prompts in one batch.
        max_abstract_words: Abstracts are cut to this many words.
        max_new_tokens: Length limit of generated reasons.
        trust_remote_code: Allow models that ship their own modeling code.

    Raises:
        BackendError: If PyTorch or transformers are missing, or the model cannot be loaded.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        dtype: str = "auto",
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_abstract_words: int = MAX_ABSTRACT_WORDS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            import transformers
        except ImportError as error:
            raise BackendError(
                'The "hf" backend needs PyTorch and transformers. Install them with: '
                'pip install "llm-literature-screener[hf]"'
            ) from error
        if dtype not in DTYPES:
            raise BackendError(f"Unknown dtype {dtype!r}; use one of: {', '.join(DTYPES)}.")
        if token_budget < 1 or max_batch_size < 1:
            raise BackendError("token_budget and max_batch_size must be positive.")
        self._torch: Any = torch
        self.model_id = model
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        self.max_abstract_words = max_abstract_words
        self.max_new_tokens = max_new_tokens
        self._warnings: list[str] = []

        # transformers 4.56 renamed the ``torch_dtype`` argument to ``dtype``.
        dtype_argument = "dtype" if _version(transformers) >= (4, 56) else "torch_dtype"
        try:
            self.tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
                model, trust_remote_code=trust_remote_code
            )
            self.model: Any = transformers.AutoModelForCausalLM.from_pretrained(
                model, trust_remote_code=trust_remote_code, **{dtype_argument: self.dtype}
            )
        except (OSError, ValueError) as error:
            raise BackendError(f"Could not load the model {model!r}: {error}") from error
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.uses_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        if not self.uses_chat_template:
            self._warn(
                f"{model} has no chat template, so a plain-text prompt is used. Instruction-tuned "
                "models with a chat template give much better screening scores."
            )
        self.yes_id, self.no_id = self._answer_token_ids()
        parameters = inspect.signature(self.model.forward).parameters
        self._logits_argument = next(
            (name for name in ("logits_to_keep", "num_logits_to_keep") if name in parameters), None
        )
        self._pass_position_ids = "position_ids" in parameters

    @property
    def info(self) -> ScorerInfo:
        """Describe the backend, model, device and precision."""
        dtype_name = str(self.dtype).replace("torch.", "")
        return ScorerInfo(
            backend="hf",
            model=self.model_id,
            description=f"Hugging Face transformers on {self.device} ({dtype_name})",
            score_definition=(
                'the difference between the logits of "Yes" and "No" for the first answer token'
            ),
        )

    @property
    def warnings(self) -> list[str]:
        """Problems noticed while loading the model or scoring."""
        return list(self._warnings)

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self._warnings.append(message)

    def _resolve_device(self, device: str) -> str:
        torch = self._torch
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, dtype: str) -> Any:
        torch = self._torch
        if dtype == "auto":
            if self.device.startswith("cuda"):
                return torch.bfloat16
            if self.device.startswith("mps"):
                return torch.float16
            return torch.float32
        return getattr(torch, dtype)

    def _answer_token_ids(self) -> tuple[int, int]:
        """Find the first token of the answers "Yes" and "No".

        With a chat template the answer starts right after the assistant header, so the
        words are encoded without a leading space, as in the case study. A plain-text prompt
        ends with "Answer:", so the answer starts with a space.
        """
        prefixes = ("",) if self.uses_chat_template else (" ", "")
        for prefix in prefixes:
            yes = self.tokenizer.encode(prefix + "Yes", add_special_tokens=False)
            no = self.tokenizer.encode(prefix + "No", add_special_tokens=False)
            if yes and no and yes[0] != no[0]:
                if len(yes) > 1 or len(no) > 1:
                    self._warn(
                        "This tokenizer splits 'Yes' or 'No' into several tokens; only the "
                        "first token of each answer is scored."
                    )
                return int(yes[0]), int(no[0])
        raise BackendError(
            f"The tokenizer of {self.model_id} has no distinct first tokens for 'Yes' and 'No'."
        )

    def _prompt_text(self, messages: list[ChatMessage]) -> str:
        if self.uses_chat_template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return str(text)
        system, user = messages
        return f"{system['content']}\n\n{user['content']}\nAnswer:"

    def _encode(self, texts: Sequence[str]) -> list[list[int]]:
        # A chat template already adds the model's special tokens.
        encoded = self.tokenizer(list(texts), add_special_tokens=not self.uses_chat_template)
        return [list(ids) for ids in encoded["input_ids"]]

    def _batch_inputs(self, sequences: Sequence[list[int]]) -> dict[str, Any]:
        torch = self._torch
        padded = self.tokenizer.pad({"input_ids": list(sequences)}, return_tensors="pt")
        input_ids = padded["input_ids"].to(self.device)
        attention_mask = padded["attention_mask"].to(self.device)
        inputs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._pass_position_ids:
            # Count positions from the first real token, so left padding does not shift them
            # and a record gets the same score whatever batch it lands in.
            position_ids = attention_mask.to(torch.long).cumsum(-1) - 1
            inputs["position_ids"] = position_ids.clamp(min=0)
        return inputs

    def score(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[float]:
        """Return logit(Yes) - logit(No) of the first answer token for each record."""
        torch = self._torch
        texts = [
            self._prompt_text(screening_messages(criteria, record, self.max_abstract_words))
            for record in records
        ]
        sequences = self._encode(texts)
        scores = [0.0] * len(texts)
        done = 0
        with torch.inference_mode():
            for batch in token_budget_batches(
                [len(sequence) for sequence in sequences], self.token_budget, self.max_batch_size
            ):
                inputs = self._batch_inputs([sequences[index] for index in batch])
                inputs["use_cache"] = False
                if self._logits_argument:
                    inputs[self._logits_argument] = 1
                logits = self.model(**inputs).logits[:, -1, :].float()
                differences = (logits[:, self.yes_id] - logits[:, self.no_id]).tolist()
                for index, value in zip(batch, differences, strict=True):
                    scores[index] = float(value)
                done += len(batch)
                if progress is not None:
                    progress(done, len(texts))
        return scores

    def explain(
        self,
        records: Sequence[Record],
        criteria: Criteria,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        """Generate a short reason for each record with greedy decoding."""
        torch = self._torch
        texts = [
            self._prompt_text(reason_messages(criteria, record, self.max_abstract_words))
            for record in records
        ]
        sequences = self._encode(texts)
        config = copy.deepcopy(self.model.generation_config)
        config.do_sample = False
        config.temperature = None
        config.top_p = None
        config.top_k = None
        config.max_new_tokens = self.max_new_tokens
        config.pad_token_id = self.tokenizer.pad_token_id
        reasons = [""] * len(texts)
        done = 0
        lengths = [len(sequence) + self.max_new_tokens for sequence in sequences]
        with torch.inference_mode():
            for batch in token_budget_batches(lengths, self.token_budget, self.max_batch_size):
                inputs = self._batch_inputs([sequences[index] for index in batch])
                inputs.pop("position_ids", None)
                generated = self.model.generate(**inputs, generation_config=config)
                new_tokens = generated[:, inputs["input_ids"].shape[1] :]
                decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                for index, text in zip(batch, decoded, strict=True):
                    reasons[index] = str(text).strip()
                done += len(batch)
                if progress is not None:
                    progress(done, len(texts))
        return reasons
