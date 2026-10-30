"""Build tiny random causal language models offline, for testing the transformers backend.

The models have a word-level tokenizer and random weights, so their scores mean nothing,
but they exercise the real code path: chat templates, padding, batching and logits.
"""

from __future__ import annotations

from pathlib import Path

CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

SPECIAL_TOKENS = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"]

WORDS = """
Yes No Answer : . , ? - ( ) system user assistant You screen titles and abstracts for a
systematic review with one word if the record should go to full text No otherwise is
missing or does not give enough detail answer Review question Population Intervention
exposure Comparator Outcomes Study designs Languages Publication years Include Exclude Title
Abstract available Should this In sentence of at most words say which eligibility criteria
meets fails based only on adults trial randomized exercise pain back chronic low therapy
children cohort heart dementia anxiety disorder cognitive behavioral treatment patients
"""


def build_tiny_model(
    directory: Path, architecture: str = "llama", chat_template: bool = True
) -> Path:
    """Save a tiny random model and its tokenizer to ``directory`` and return the path.

    Args:
        directory: Where to save the model.
        architecture: ``llama`` (rotary position embeddings) or ``gpt2`` (absolute
            position embeddings, which make left-padding mistakes visible).
        chat_template: Whether the tokenizer gets a chat template.
    """
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    vocabulary = SPECIAL_TOKENS + sorted(set(WORDS.split()) - set(SPECIAL_TOKENS))
    ids = {token: index for index, token in enumerate(vocabulary)}
    backend = Tokenizer(models.WordLevel(vocab=ids, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(SPECIAL_TOKENS)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>"
    )
    if chat_template:
        tokenizer.chat_template = CHAT_TEMPLATE

    torch.manual_seed(0)
    special = {
        "pad_token_id": ids["<pad>"],
        "bos_token_id": ids["<|im_start|>"],
        "eos_token_id": ids["<|im_end|>"],
    }
    if architecture == "llama":
        llama = LlamaConfig(
            vocab_size=len(ids),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            **special,
        )
        model = LlamaForCausalLM(llama)
    elif architecture == "gpt2":
        gpt2 = GPT2Config(
            vocab_size=len(ids), n_positions=2048, n_embd=32, n_layer=2, n_head=4, **special
        )
        model = GPT2LMHeadModel(gpt2)
    else:
        raise ValueError(f"Unknown architecture {architecture!r}")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory
