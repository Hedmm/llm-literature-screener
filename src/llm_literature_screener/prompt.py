"""Prompts shown to the model, ported unchanged from the validated case study.

For each record the model reads the review question, the eligibility criteria, the title
and the abstract (cut to 350 words), and is asked to answer Yes or No. The score is read
from the logits of the first answer token, so no text is generated for scoring.
"""

from __future__ import annotations

from typing import TypedDict

from llm_literature_screener.criteria import Criteria
from llm_literature_screener.record import Record

SYSTEM_PROMPT = (
    "You screen titles and abstracts for a systematic review. Answer with one word: Yes if the "
    "record should go to full-text review, No otherwise. If the abstract is missing or does not "
    "give enough detail, answer Yes."
)
"""System message for scoring, as validated in the case study."""

QUESTION = "Should this record go to full-text review? Answer Yes or No."
"""Final line of the scoring prompt."""

REASON_SYSTEM_PROMPT = (
    "You screen titles and abstracts for a systematic review and explain your judgements in "
    "plain, factual language."
)
"""System message for the optional one-line reasons."""

REASON_QUESTION = (
    "In one sentence of at most 30 words, say which eligibility criteria this record meets or "
    "fails, based only on the title and abstract."
)
"""Final line of the reason prompt."""

MAX_ABSTRACT_WORDS = 350
"""Abstracts are cut to this many words, which keeps prompts short with little loss."""

MISSING = "not available"
"""Shown to the model in place of a missing title or abstract."""


class ChatMessage(TypedDict):
    """One chat message in the format used by chat templates and OpenAI-compatible APIs."""

    role: str
    content: str


def format_title(title: str) -> str:
    """Return the title, or ``not available`` when it is empty."""
    text = " ".join(title.split())
    return text or MISSING


def format_abstract(abstract: str, max_words: int = MAX_ABSTRACT_WORDS) -> str:
    """Return the abstract cut to ``max_words`` words, or ``not available`` when it is empty.

    A cut abstract ends with `` ...`` so the model can tell that text is missing.
    """
    words = abstract.split()
    if not words:
        return MISSING
    text = " ".join(words[:max_words])
    return text + " ..." if len(words) > max_words else text


def record_text(record: Record, max_words: int = MAX_ABSTRACT_WORDS) -> str:
    """Return the title and abstract block of a prompt."""
    return (
        f"Title: {format_title(record.title)}\n"
        f"Abstract: {format_abstract(record.abstract, max_words)}"
    )


def screening_prompt(
    criteria: Criteria, record: Record, max_words: int = MAX_ABSTRACT_WORDS
) -> str:
    """Return the user message that asks whether ``record`` should go to full-text review."""
    return f"{criteria.render()}\n\n{record_text(record, max_words)}\n\n{QUESTION}"


def screening_messages(
    criteria: Criteria, record: Record, max_words: int = MAX_ABSTRACT_WORDS
) -> list[ChatMessage]:
    """Return the system and user messages used for scoring."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": screening_prompt(criteria, record, max_words)},
    ]


def reason_messages(
    criteria: Criteria, record: Record, max_words: int = MAX_ABSTRACT_WORDS
) -> list[ChatMessage]:
    """Return the messages that ask for a one-line reason for a record."""
    user = f"{criteria.render()}\n\n{record_text(record, max_words)}\n\n{REASON_QUESTION}"
    return [
        {"role": "system", "content": REASON_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
