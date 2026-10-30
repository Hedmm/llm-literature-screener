"""Reader and writer for Covidence CSV exports.

Covidence exports one list of references at a time (for example the studies to screen or the
irrelevant ones) with columns such as ``Title``, ``Authors``, ``Abstract``,
``Published Year``, ``Journal``, ``Accession Number``, ``DOI`` and ``Covidence #``. The
``Covidence #`` column becomes the record id. Because the export holds no decision column,
human labels for calibration usually come from a separate labels file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_literature_screener.io.table import (
    DEFAULT_COLUMNS,
    TableProfile,
    read_table_records,
    write_delimited,
)
from llm_literature_screener.record import Record

COVIDENCE_COLUMNS: dict[str, tuple[str, ...]] = {
    **DEFAULT_COLUMNS,
    "id": ("covidence #", *DEFAULT_COLUMNS["id"]),
    "year": ("published year", *DEFAULT_COLUMNS["year"]),
    "pmid": (*DEFAULT_COLUMNS["pmid"], "accession number"),
}

SIGNATURE = frozenset({"covidence #"})
"""Lower-case headers that identify a Covidence export."""

COVIDENCE_PROFILE = TableProfile(name="Covidence", columns=COVIDENCE_COLUMNS)


def read_covidence(path: Path, mapping: Mapping[str, str] | None = None) -> list[Record]:
    """Read a Covidence CSV export. Accession numbers that look like PMIDs are kept as PMIDs."""
    return read_table_records(path, "csv", COVIDENCE_PROFILE, mapping)


def write_covidence(records: Sequence[Record], path: Path) -> None:
    """Write records back as a Covidence CSV with the tool's output columns appended."""
    write_delimited(records, path)
