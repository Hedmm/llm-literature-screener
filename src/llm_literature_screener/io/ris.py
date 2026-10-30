"""Reader and writer for RIS files (EndNote, Zotero, Mendeley, Rayyan, Covidence and databases).

Each record starts with ``TY  - <type>`` and ends with ``ER  -``. Lines that do not start
with a tag continue the previous field. The tool's output is written as ``N1`` notes such as
``N1  - lls_decision: include``, so that reference managers keep it with the record.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from llm_literature_screener.io.common import (
    assign_ids,
    clean_text,
    is_output_field,
    normalize_doi,
    normalize_pmid,
    output_values,
    parse_year,
    read_text,
)
from llm_literature_screener.record import RawFields, Record

_TAG_LINE = re.compile(r"^([A-Z][A-Z0-9])\s{1,2}-(?:\s(.*))?$")
_PMID_IN_TEXT = re.compile(r"\bPMID\s*:?\s*(\d{1,9})\b", re.IGNORECASE)

TITLE_TAGS = ("TI", "T1", "CT", "BT")
ABSTRACT_TAGS = ("AB", "N2")
AUTHOR_TAGS = ("AU", "A1")
YEAR_TAGS = ("PY", "Y1", "DA", "Y2")
JOURNAL_TAGS = ("T2", "JF", "JO", "JA", "J2", "J1")
DOI_TAGS = ("DO", "DI")
DOI_FALLBACK_TAGS = ("UR", "L3", "L2", "M3", "N1")

ASREVIEW_RELEVANT = "asreview_relevant"
ASREVIEW_IRRELEVANT = "asreview_irrelevant"


def parse_entries(text: str) -> list[RawFields]:
    """Split RIS text into records, each a list of (tag, value) pairs in file order."""
    entries: list[RawFields] = []
    current: RawFields | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _TAG_LINE.match(line.rstrip())
        if match:
            tag, value = match.group(1), (match.group(2) or "").strip()
            if tag == "TY":
                if current:
                    entries.append(current)
                current = [(tag, value)]
            elif tag == "ER":
                if current:
                    entries.append(current)
                current = None
            else:
                if current is None:
                    current = []
                current.append((tag, value))
        elif current:
            tag, value = current[-1]
            current[-1] = (tag, f"{value} {line.strip()}".strip())
    if current:
        entries.append(current)
    return entries


def _first(fields: RawFields, tags: Iterable[str]) -> str:
    for tag in tags:
        for key, value in fields:
            if key == tag and value.strip():
                return clean_text(value)
    return ""


def _doi(fields: RawFields) -> str:
    for tags in (DOI_TAGS, DOI_FALLBACK_TAGS):
        for tag in tags:
            for key, value in fields:
                if key == tag:
                    doi = normalize_doi(value)
                    if doi:
                        return doi
    return ""


def _pmid(fields: RawFields) -> str:
    for key, value in fields:
        if key == "AN":
            pmid = normalize_pmid(value)
            if pmid:
                return pmid
    for _, value in fields:
        match = _PMID_IN_TEXT.search(value)
        if match:
            return match.group(1)
    return ""


def _label(fields: RawFields) -> bool | None:
    """Read an ASReview label from the notes, the only common way RIS files carry decisions."""
    for key, value in fields:
        if key == "N1":
            note = value.strip().lower()
            if ASREVIEW_RELEVANT in note:
                return True
            if ASREVIEW_IRRELEVANT in note:
                return False
    return None


def record_from_fields(fields: RawFields, record_id: str) -> Record:
    """Build a Record from the (tag, value) pairs of one RIS entry."""
    return Record(
        id=record_id,
        title=_first(fields, TITLE_TAGS),
        abstract=_first(fields, ABSTRACT_TAGS),
        authors=[clean_text(v) for k, v in fields if k in AUTHOR_TAGS and v.strip()],
        year=parse_year(_first(fields, YEAR_TAGS)),
        journal=_first(fields, JOURNAL_TAGS),
        doi=_doi(fields),
        pmid=_pmid(fields),
        raw=list(fields),
        label=_label(fields),
    )


def read_ris(path: Path) -> list[Record]:
    """Read every record in a RIS file.

    The record id is the ``ID`` tag when present, otherwise the record's position in the file.
    """
    entries = parse_entries(read_text(path))
    ids = assign_ids([_first(fields, ("ID",)) for fields in entries])
    return [record_from_fields(fields, ident) for fields, ident in zip(entries, ids, strict=True)]


def _fields_for_new_record(record: Record) -> RawFields:
    """Describe a record that was not read from RIS with standard RIS tags."""
    fields: RawFields = [("TY", "JOUR"), ("ID", record.id)]
    if record.title:
        fields.append(("TI", record.title))
    fields.extend(("AU", author) for author in record.authors)
    if record.year is not None:
        fields.append(("PY", str(record.year)))
    if record.journal:
        fields.append(("T2", record.journal))
    if record.doi:
        fields.append(("DO", record.doi))
    if record.pmid:
        fields.append(("AN", record.pmid))
    if record.abstract:
        fields.append(("AB", record.abstract))
    return fields


def format_ris(records: Sequence[Record]) -> str:
    """Return RIS text for ``records``, keeping every source field and adding the tool's output."""
    lines: list[str] = []
    for record in records:
        fields = record.raw or _fields_for_new_record(record)
        kept = [
            (tag, value)
            for tag, value in fields
            if tag not in ("TY", "ER") and not (tag == "N1" and is_output_field(value))
        ]
        record_type = next((value for tag, value in fields if tag == "TY" and value), "JOUR")
        lines.append(f"TY  - {record_type}")
        lines.extend(f"{tag}  - {clean_text(value)}" for tag, value in kept)
        lines.extend(f"N1  - {name}: {value}" for name, value in output_values(record))
        lines.append("ER  - ")
        lines.append("")
    return "\n".join(lines)


def write_ris(records: Sequence[Record], path: Path) -> None:
    """Write ``records`` to a RIS file (UTF-8)."""
    path.write_text(format_ris(records), encoding="utf-8")
