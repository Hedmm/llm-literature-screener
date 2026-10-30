"""Read human screening decisions from a labels file and attach them to records.

A labels file is a CSV, TSV or Excel table with a label column (``label``, ``included``,
``decision``, ...) and at least one column that identifies the record: ``id``, ``doi``,
``pmid`` or ``title``. Records are matched by id first, then DOI, then PMID, then normalized
title, so labels exported from another tool can be used even when the ids differ.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from llm_literature_screener.io.common import (
    clean_text,
    normalize_doi,
    normalize_pmid,
    normalize_title,
    parse_label,
)
from llm_literature_screener.io.table import Table, read_delimited, read_excel
from llm_literature_screener.record import Record

logger = logging.getLogger(__name__)

ID_COLUMNS = ("id", "record_id", "record id", "key", "covidence #", "refid", "ref id")
LABEL_COLUMNS = (
    "label",
    "label_included",
    "included",
    "include",
    "decision",
    "final decision",
    "screening decision",
)


class LabelFileError(ValueError):
    """Raised when a labels file has no usable label or identifier column."""


@dataclass(frozen=True)
class LabelEntry:
    """One human decision and the identifiers used to find its record."""

    label: bool
    id: str = ""
    doi: str = ""
    pmid: str = ""
    title: str = ""


@dataclass
class LabelReport:
    """What happened when labels were attached to records.

    Attributes:
        entries: Number of labels read from the file.
        matched: Number of labels that found a record.
        unmatched: Labels that did not match any record.
        changed: Records whose existing label (from the records file) was replaced.
    """

    entries: int
    matched: int
    unmatched: list[LabelEntry] = field(default_factory=list)
    changed: int = 0


def _column(headers: Sequence[str], candidates: Sequence[str]) -> int | None:
    normalized = [" ".join(header.lower().split()) for header in headers]
    for candidate in candidates:
        if candidate in normalized:
            return normalized.index(candidate)
    return None


def _read_table(path: Path) -> Table:
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return read_excel(path)
    return read_delimited(path, delimiter="\t" if path.suffix.lower() in {".tsv", ".tab"} else None)


def read_labels(path: Path, label_column: str | None = None) -> list[LabelEntry]:
    """Read human decisions from a CSV, TSV or Excel labels file.

    Args:
        path: The labels file.
        label_column: Name of the label column, when it is not one of the usual names.

    Returns:
        One entry per row with a recognized decision. Rows whose label is empty or not
        recognized (for example "maybe") are skipped with a warning.

    Raises:
        LabelFileError: If the label column or every identifier column is missing.
    """
    table = _read_table(path)
    label_index = _column(table.headers, [label_column.lower()] if label_column else LABEL_COLUMNS)
    if label_index is None:
        expected = label_column or ", ".join(LABEL_COLUMNS)
        raise LabelFileError(f"{path.name} has no label column (looked for: {expected}).")
    id_index = _column(table.headers, ID_COLUMNS)
    doi_index = _column(table.headers, ("doi",))
    pmid_index = _column(table.headers, ("pmid", "pubmed_id", "pubmed id"))
    title_index = _column(table.headers, ("title",))
    if all(index is None for index in (id_index, doi_index, pmid_index, title_index)):
        raise LabelFileError(
            f"{path.name} needs an id, doi, pmid or title column to match labels to records."
        )

    def cell(row: list[str], index: int | None) -> str:
        return clean_text(row[index]) if index is not None else ""

    entries: list[LabelEntry] = []
    skipped = 0
    for row in table.rows:
        label = parse_label(row[label_index])
        if label is None:
            skipped += 1
            continue
        entries.append(
            LabelEntry(
                label=label,
                id=cell(row, id_index),
                doi=normalize_doi(cell(row, doi_index)),
                pmid=normalize_pmid(cell(row, pmid_index)),
                title=normalize_title(cell(row, title_index)),
            )
        )
    if skipped:
        logger.warning(
            "%d row(s) in %s have an empty or unrecognized label and were skipped.",
            skipped,
            path.name,
        )
    return entries


def apply_labels(records: Sequence[Record], entries: Sequence[LabelEntry]) -> LabelReport:
    """Set ``record.label`` from matching label entries.

    Each entry is matched by id, then DOI, then PMID, then normalized title. A later entry
    for the same record overrides an earlier one.
    """
    by_id = {record.id: record for record in reversed(records)}
    by_doi = {record.doi: record for record in reversed(records) if record.doi}
    by_pmid = {record.pmid: record for record in reversed(records) if record.pmid}
    by_title = {
        normalize_title(record.title): record for record in reversed(records) if record.title
    }
    report = LabelReport(entries=len(entries), matched=0)
    for entry in entries:
        record = (
            (by_id.get(entry.id) if entry.id else None)
            or (by_doi.get(entry.doi) if entry.doi else None)
            or (by_pmid.get(entry.pmid) if entry.pmid else None)
            or (by_title.get(entry.title) if entry.title else None)
        )
        if record is None:
            report.unmatched.append(entry)
            continue
        if record.label is not None and record.label != entry.label:
            report.changed += 1
        record.label = entry.label
        report.matched += 1
    if report.unmatched:
        logger.warning(
            "%d of %d label(s) did not match any record.", len(report.unmatched), len(entries)
        )
    return report
