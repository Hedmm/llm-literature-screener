"""Parsing helpers shared by the file format readers and writers."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote

from llm_literature_screener.record import Record

logger = logging.getLogger(__name__)

OUTPUT_PREFIX = "lls_"
"""Prefix of the fields this tool adds to output files, so they never clash with source fields."""

OUTPUT_FIELDS = ("score", "p_include", "decision", "reason")
"""Names of the added fields, without the prefix, in the order they are written."""

_DOI_PATTERN = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_PMID_PATTERN = re.compile(r"^\s*(?:pmid\s*:?\s*)?(\d{1,9})\s*$", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)")
_NON_ALPHANUMERIC = re.compile(r"[^0-9a-z]+")

_TRUE_LABELS = {"1", "1.0", "true", "yes", "y", "include", "included", "relevant", "in"}
_FALSE_LABELS = {"0", "0.0", "false", "no", "n", "exclude", "excluded", "irrelevant", "out"}


def clean_text(value: str) -> str:
    """Collapse runs of whitespace (including line breaks) into single spaces."""
    return " ".join(value.split())


def normalize_doi(value: str) -> str:
    """Extract a DOI from ``value`` and return it in lower case.

    Accepts bare DOIs, ``doi:`` prefixes, ``https://doi.org/`` URLs and MEDLINE values such as
    ``10.1000/xyz [doi]``. Returns an empty string when no DOI is found.
    """
    text = value.strip()
    if "%2f" in text.lower():
        text = unquote(text)
    match = _DOI_PATTERN.search(text)
    if not match:
        return ""
    doi = match.group(0).rstrip(".,;'\"")
    # Drop closing brackets that belong to the surrounding text, not to the DOI.
    for opening, closing in (("(", ")"), ("[", "]"), ("<", ">")):
        while doi.endswith(closing) and doi.count(closing) > doi.count(opening):
            doi = doi[:-1]
    return doi.lower()


def normalize_pmid(value: str) -> str:
    """Return the PubMed id in ``value`` (digits only), or an empty string."""
    match = _PMID_PATTERN.match(value)
    return match.group(1).lstrip("0") if match else ""


def parse_year(value: str) -> int | None:
    """Return the first plausible publication year (1500-2099) found in ``value``."""
    match = _YEAR_PATTERN.search(value)
    return int(match.group(1)) if match else None


def parse_label(value: str | None) -> bool | None:
    """Interpret a human screening decision.

    ``include``, ``included``, ``yes``, ``1`` and ``true`` (any case) mean include;
    ``exclude``, ``excluded``, ``no``, ``0`` and ``false`` mean exclude. Anything else,
    including an empty value or ``maybe``, means the decision is unknown.
    """
    if value is None:
        return None
    text = value.strip().lower()
    if text in _TRUE_LABELS:
        return True
    if text in _FALSE_LABELS:
        return False
    return None


def format_label(label: bool | None) -> str:
    """Return ``include``, ``exclude`` or an empty string for an unknown label."""
    if label is None:
        return ""
    return "include" if label else "exclude"


def split_authors(value: str) -> list[str]:
    """Split an author list written in one cell.

    Semicolons, `` and `` (used by Rayyan) and vertical bars are treated as separators.
    Commas are not, because they also separate family names from given names.
    """
    text = value.strip()
    if not text:
        return []
    for separator in (";", " and ", "|"):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def normalize_title(title: str) -> str:
    """Normalize a title for duplicate matching.

    Accents are removed, letters are lower-cased and every run of characters other than
    letters and digits becomes a single space.
    """
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _NON_ALPHANUMERIC.sub(" ", ascii_only.lower()).strip()


def assign_ids(candidates: Sequence[str]) -> list[str]:
    """Turn candidate ids from a file into ids that are unique and never empty.

    A record without a candidate id gets its 1-based position in the file. When several
    records share a candidate id, the first keeps it and the others get ``<id>-2``,
    ``<id>-3`` and so on.
    """
    taken = {candidate.strip() for candidate in candidates if candidate.strip()}
    duplicates = [
        value
        for value, count in Counter(c.strip() for c in candidates).items()
        if value and count > 1
    ]
    if duplicates:
        logger.warning(
            "%d record id(s) occur more than once (for example %r); repeated ids get a suffix.",
            len(duplicates),
            duplicates[0],
        )
    used: set[str] = set()
    ids: list[str] = []
    for position, candidate in enumerate(candidates, start=1):
        base = candidate.strip() or str(position)
        ident = base
        suffix = 1
        while ident in used or (ident != candidate.strip() and ident in taken):
            suffix += 1
            ident = f"{base}-{suffix}"
        used.add(ident)
        ids.append(ident)
    return ids


def read_text(path: Path) -> str:
    """Read a text file, accepting UTF-8 (with or without a byte order mark) or Windows-1252."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s is not valid UTF-8; reading it as Windows-1252.", path.name)
        return data.decode("cp1252", errors="replace")


def format_number(value: float) -> str:
    """Format a float with full precision so that it reads back unchanged."""
    return repr(float(value))


def output_values(record: Record) -> list[tuple[str, str]]:
    """Return the fields this tool adds to an output record, prefixed with ``lls_``.

    Returns an empty list for a record that has not been assessed yet.
    """
    assessment = record.assessment
    if assessment is None:
        return []
    values = (
        format_number(assessment.score),
        format_number(assessment.p_include),
        assessment.decision.value,
        clean_text(assessment.reason or ""),
    )
    return [
        (OUTPUT_PREFIX + name, value) for name, value in zip(OUTPUT_FIELDS, values, strict=True)
    ]


def is_output_field(name: str) -> bool:
    """Return True for field names written by this tool (they start with ``lls_``)."""
    return name.startswith(OUTPUT_PREFIX)
