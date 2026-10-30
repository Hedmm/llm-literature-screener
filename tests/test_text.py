"""Tests for the wording helpers."""

from llm_literature_screener.text import plural


def test_plural() -> None:
    assert plural(1, "record") == "1 record"
    assert plural(0, "record") == "0 records"
    assert plural(1200, "record") == "1,200 records"
    assert plural(2, "study", "studies") == "2 studies"
