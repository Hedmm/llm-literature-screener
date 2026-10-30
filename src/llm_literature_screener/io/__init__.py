"""Readers and writers for bibliographic record files.

Supported formats: RIS, PubMed MEDLINE/NBIB, CSV, TSV, Excel, Rayyan CSV exports and
Covidence CSV exports. Each reader returns a list of :class:`~llm_literature_screener.Record`
objects, and each writer produces the same format back with the tool's output added.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path

from llm_literature_screener.io import covidence, rayyan
from llm_literature_screener.io.covidence import read_covidence, write_covidence
from llm_literature_screener.io.labels import LabelEntry, LabelReport, apply_labels, read_labels
from llm_literature_screener.io.medline import read_medline, write_medline
from llm_literature_screener.io.rayyan import read_rayyan, write_rayyan
from llm_literature_screener.io.results import read_results, read_scores, write_results
from llm_literature_screener.io.ris import read_ris, write_ris
from llm_literature_screener.io.table import (
    ColumnMappingError,
    read_table_records,
    write_delimited,
    write_excel,
)
from llm_literature_screener.record import Record

FORMATS: dict[str, str] = {
    "ris": "RIS",
    "medline": "PubMed MEDLINE/NBIB",
    "csv": "CSV",
    "tsv": "TSV",
    "excel": "Excel (.xlsx)",
    "rayyan": "Rayyan CSV export",
    "covidence": "Covidence CSV export",
}
"""Format names accepted by :func:`read_records` and :func:`write_records`."""

SUFFIXES: dict[str, str] = {
    "ris": ".ris",
    "medline": ".nbib",
    "csv": ".csv",
    "tsv": ".tsv",
    "excel": ".xlsx",
    "rayyan": ".csv",
    "covidence": ".csv",
}
"""File extension used when writing each format."""

TABLE_FORMATS = frozenset({"csv", "tsv", "excel", "rayyan", "covidence"})
"""Formats that accept a column mapping."""

_RIS_START = re.compile(r"^TY\s{1,2}-")

__all__ = [
    "FORMATS",
    "SUFFIXES",
    "TABLE_FORMATS",
    "ColumnMappingError",
    "LabelEntry",
    "LabelReport",
    "UnknownFormatError",
    "apply_labels",
    "detect_format",
    "read_labels",
    "read_records",
    "read_results",
    "read_scores",
    "write_records",
    "write_results",
]


class UnknownFormatError(ValueError):
    """Raised when a file's format cannot be detected or is not supported."""


def _head(path: Path, size: int = 65536) -> str:
    with path.open("rb") as handle:
        data = handle.read(size)
    return data.decode("utf-8-sig", errors="replace")


def _header_names(first_line: str) -> set[str]:
    counts = {delimiter: first_line.count(delimiter) for delimiter in (",", ";", "\t")}
    delimiter = max(counts, key=lambda d: counts[d])
    cells = next(csv.reader(StringIO(first_line), delimiter=delimiter), [])
    return {" ".join(cell.strip().lower().split()) for cell in cells}


def detect_format(path: Path) -> str:
    """Guess the format of a records file from its extension and first lines.

    CSV files are recognized as Rayyan or Covidence exports from their headers. Text files
    are recognized as MEDLINE (``PMID-``) or RIS (``TY  -``) from their first tag.

    Raises:
        UnknownFormatError: If the format cannot be determined.
    """
    suffix = path.suffix.lower()
    if suffix == ".ris":
        return "ris"
    if suffix in {".nbib", ".medline"}:
        return "medline"
    if suffix in {".xlsx", ".xlsm"}:
        return "excel"
    if suffix == ".xls":
        raise UnknownFormatError(
            f"{path.name}: the old .xls format is not supported; save it as .xlsx or CSV."
        )
    if suffix in {".tsv", ".tab"}:
        return "tsv"
    if suffix in {".csv", ".txt"}:
        head = _head(path).lstrip()
        if head.startswith("PMID-"):
            return "medline"
        if _RIS_START.match(head):
            return "ris"
        lines = head.splitlines()
        if lines and lines[0].strip().lower().startswith("sep="):
            lines = lines[1:]
        first_line = lines[0] if lines else ""
        if suffix == ".txt":
            if "\t" in first_line:
                return "tsv"
            raise UnknownFormatError(
                f"{path.name}: cannot tell the format of this text file; pass the format "
                f"explicitly (one of: {', '.join(FORMATS)})."
            )
        headers = _header_names(first_line)
        if headers >= rayyan.SIGNATURE:
            return "rayyan"
        if headers >= covidence.SIGNATURE:
            return "covidence"
        return "csv"
    raise UnknownFormatError(
        f"{path.name}: unsupported file extension {suffix or '(none)'}; pass the format "
        f"explicitly (one of: {', '.join(FORMATS)})."
    )


def read_records(
    path: Path,
    fmt: str | None = None,
    columns: Mapping[str, str] | None = None,
    sheet: str | None = None,
) -> list[Record]:
    """Read records from a file.

    Args:
        path: The records file.
        fmt: One of :data:`FORMATS`; detected from the file when omitted.
        columns: Field-to-column mapping for table formats, for example
            ``{"title": "Article Title"}``.
        sheet: Worksheet name for Excel files (the first worksheet by default).

    Raises:
        UnknownFormatError: If the format is unknown or cannot be detected.
        ColumnMappingError: If a column mapping is invalid or is given for a tagged format.
    """
    fmt = fmt or detect_format(path)
    if fmt not in FORMATS:
        raise UnknownFormatError(f"Unknown format {fmt!r}; use one of: {', '.join(FORMATS)}.")
    if columns and fmt not in TABLE_FORMATS:
        raise ColumnMappingError(
            "A column mapping applies only to CSV, TSV, Excel, Rayyan and Covidence files."
        )
    if fmt == "ris":
        return read_ris(path)
    if fmt == "medline":
        return read_medline(path)
    if fmt == "rayyan":
        return read_rayyan(path, columns)
    if fmt == "covidence":
        return read_covidence(path, columns)
    return read_table_records(path, fmt, mapping=columns, sheet=sheet)


def write_records(records: Sequence[Record], path: Path, fmt: str) -> None:
    """Write records in the given format, adding score, p_include, decision and reason.

    Raises:
        UnknownFormatError: If the format is unknown.
    """
    if fmt == "ris":
        write_ris(records, path)
    elif fmt == "medline":
        write_medline(records, path)
    elif fmt == "rayyan":
        write_rayyan(records, path)
    elif fmt == "covidence":
        write_covidence(records, path)
    elif fmt == "csv":
        write_delimited(records, path)
    elif fmt == "tsv":
        write_delimited(records, path, delimiter="\t")
    elif fmt == "excel":
        write_excel(records, path)
    else:
        raise UnknownFormatError(f"Unknown format {fmt!r}; use one of: {', '.join(FORMATS)}.")
