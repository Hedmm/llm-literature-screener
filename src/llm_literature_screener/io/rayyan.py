"""Reader and writer for Rayyan CSV exports.

A Rayyan export has one row per article with columns such as ``key``, ``title``,
``authors`` (separated by `` and ``), ``year``, ``journal``, ``pubmed_id``, ``doi``,
``abstract`` and ``notes``. Screening decisions are stored inside ``notes``, for example::

    RAYYAN-INCLUSION: {"Alice"=>"Included", "Bob"=>"Excluded"} | RAYYAN-LABELS: RCT

A record counts as a known inclusion when at least one reviewer included it, and as a known
exclusion when every reviewer who decided excluded it. "Maybe" decisions leave it unknown.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_literature_screener.io.table import (
    DEFAULT_COLUMNS,
    TableProfile,
    read_table_records,
    write_delimited,
)
from llm_literature_screener.record import Record

_INCLUSION_BLOCK = re.compile(r"RAYYAN-INCLUSION:\s*\{(?P<body>[^}]*)\}", re.IGNORECASE)
_DECISION = re.compile(
    r"\"?(?P<name>[^\"{},]*?)\"?\s*(?:=>|:)\s*\"?(?P<decision>Included|Excluded|Maybe|Undecided)\"?",
    re.IGNORECASE,
)

RAYYAN_COLUMNS: dict[str, tuple[str, ...]] = {
    **DEFAULT_COLUMNS,
    "id": ("key", *DEFAULT_COLUMNS["id"]),
    "pmid": ("pubmed_id", *DEFAULT_COLUMNS["pmid"]),
}

SIGNATURE = frozenset({"key", "title", "authors", "notes"})
"""Lower-case headers that together identify a Rayyan export."""


def reviewer_decisions(notes: str) -> dict[str, str]:
    """Return each reviewer's decision (``included``, ``excluded``, ``maybe``) from the notes."""
    match = _INCLUSION_BLOCK.search(notes)
    if not match:
        return {}
    return {
        decision.group("name").strip(): decision.group("decision").lower()
        for decision in _DECISION.finditer(match.group("body"))
    }


def label_from_notes(notes: str) -> bool | None:
    """Turn the reviewers' Rayyan decisions into one human label (see the module docstring)."""
    decisions = set(reviewer_decisions(notes).values())
    if "included" in decisions:
        return True
    if decisions and decisions <= {"excluded"}:
        return False
    return None


def _label_from_row(row: Mapping[str, str]) -> bool | None:
    for header, value in row.items():
        if header.strip().lower() == "notes":
            return label_from_notes(value)
    return None


RAYYAN_PROFILE = TableProfile(name="Rayyan", columns=RAYYAN_COLUMNS, label_parser=_label_from_row)


def read_rayyan(path: Path, mapping: Mapping[str, str] | None = None) -> list[Record]:
    """Read a Rayyan CSV export, taking human labels from the ``notes`` column."""
    return read_table_records(path, "csv", RAYYAN_PROFILE, mapping)


def write_rayyan(records: Sequence[Record], path: Path) -> None:
    """Write records back as a Rayyan CSV with the tool's output columns appended."""
    write_delimited(records, path)
