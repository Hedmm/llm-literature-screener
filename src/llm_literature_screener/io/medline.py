"""Reader and writer for the PubMed MEDLINE format (``.nbib`` and PubMed "Save as PubMed" files).

Each line starts with a tag padded to four characters and ``- ``, for example ``TI  - Title``
or ``PMID- 12345678``. Lines that start with spaces continue the previous field, and a blank
line separates records. The tool's output is written as general notes such as
``GN  - lls_decision: include``.
"""

from __future__ import annotations

import re
import textwrap
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

_TAG_LINE = re.compile(r"^([A-Z][A-Z0-9]{0,3}) *- (.*)$")
_DOI_IN_SOURCE = re.compile(r"\bdoi:\s*(10\.\S+)", re.IGNORECASE)

LINE_WIDTH = 82
"""PubMed wraps lines at this width, with continuation lines indented by six spaces."""


def parse_entries(text: str) -> list[RawFields]:
    """Split MEDLINE text into records, each a list of (tag, value) pairs in file order."""
    entries: list[RawFields] = []
    current: RawFields = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                entries.append(current)
                current = []
            continue
        match = _TAG_LINE.match(line) if not line[0].isspace() else None
        if match:
            tag, value = match.group(1), match.group(2).strip()
            if tag == "PMID" and current:
                entries.append(current)
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
    for tag in ("LID", "AID"):
        for key, value in fields:
            if key == tag and value.strip().lower().endswith("[doi]"):
                return normalize_doi(value)
    for key, value in fields:
        if key == "SO":
            match = _DOI_IN_SOURCE.search(value)
            if match:
                return normalize_doi(match.group(1))
    return ""


def record_from_fields(fields: RawFields, record_id: str) -> Record:
    """Build a Record from the (tag, value) pairs of one MEDLINE entry."""
    full_names = [clean_text(v) for k, v in fields if k == "FAU" and v.strip()]
    short_names = [clean_text(v) for k, v in fields if k == "AU" and v.strip()]
    return Record(
        id=record_id,
        title=_first(fields, ("TI", "BTI", "TT")),
        abstract=_first(fields, ("AB",)),
        authors=full_names or short_names,
        year=parse_year(_first(fields, ("DP", "DEP", "EDAT"))),
        journal=_first(fields, ("JT", "TA")),
        doi=_doi(fields),
        pmid=normalize_pmid(_first(fields, ("PMID",))),
        raw=list(fields),
    )


def read_medline(path: Path) -> list[Record]:
    """Read every record in a MEDLINE/NBIB file. The record id is the PMID."""
    entries = parse_entries(read_text(path))
    ids = assign_ids([_first(fields, ("PMID",)) for fields in entries])
    return [record_from_fields(fields, ident) for fields, ident in zip(entries, ids, strict=True)]


def _fields_for_new_record(record: Record) -> RawFields:
    """Describe a record that was not read from MEDLINE with standard MEDLINE tags."""
    fields: RawFields = [("PMID", record.pmid or record.id)]
    if record.title:
        fields.append(("TI", record.title))
    if record.abstract:
        fields.append(("AB", record.abstract))
    fields.extend(("FAU", author) for author in record.authors)
    if record.year is not None:
        fields.append(("DP", str(record.year)))
    if record.journal:
        fields.append(("JT", record.journal))
    if record.doi:
        fields.append(("LID", f"{record.doi} [doi]"))
    return fields


def _format_field(tag: str, value: str) -> list[str]:
    prefix = f"{tag:<4}- "
    wrapped = textwrap.wrap(
        clean_text(value),
        width=LINE_WIDTH,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [prefix.rstrip()]


def format_medline(records: Sequence[Record]) -> str:
    """Return MEDLINE text for ``records``, keeping every source field and adding the output."""
    lines: list[str] = []
    for record in records:
        fields = record.raw or _fields_for_new_record(record)
        for tag, value in fields:
            if not (tag == "GN" and is_output_field(value)):
                lines.extend(_format_field(tag, value))
        for name, value in output_values(record):
            lines.extend(_format_field("GN", f"{name}: {value}"))
        lines.append("")
    return "\n".join(lines)


def write_medline(records: Sequence[Record], path: Path) -> None:
    """Write ``records`` to a MEDLINE/NBIB file (UTF-8)."""
    path.write_text(format_medline(records), encoding="utf-8")
