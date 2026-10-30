"""The flat results table written by ``lls screen`` and read by the other commands.

``screening_results.csv`` has one row per screened record, ranked by score (highest first),
with the columns in :data:`RESULT_COLUMNS`. It is the common format that ``lls calibrate``,
``lls evaluate`` and ``lls report`` work on, whatever the format of the original records.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from llm_literature_screener.io.common import (
    clean_text,
    format_label,
    format_number,
    parse_label,
    read_text,
    split_authors,
)
from llm_literature_screener.io.table import read_delimited
from llm_literature_screener.metrics import rank_order
from llm_literature_screener.record import Assessment, Decision, Record

RESULT_COLUMNS = (
    "rank",
    "id",
    "title",
    "authors",
    "year",
    "journal",
    "doi",
    "pmid",
    "label",
    "score",
    "p_include",
    "decision",
    "reason",
)


class ResultsFileError(ValueError):
    """Raised when a results or scores file is missing a required column or value."""


def write_results(records: Sequence[Record], path: Path) -> None:
    """Write assessed records to a results CSV, ranked by score.

    Raises:
        ValueError: If a record has not been assessed.
    """
    assessments = []
    for record in records:
        if record.assessment is None:
            raise ValueError(f"Record {record.id} has no assessment to write.")
        assessments.append(record.assessment)
    order = rank_order([assessment.score for assessment in assessments])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_COLUMNS)
        for rank, index in enumerate(order, start=1):
            record, assessment = records[index], assessments[index]
            writer.writerow(
                [
                    rank,
                    record.id,
                    record.title,
                    "; ".join(record.authors),
                    "" if record.year is None else record.year,
                    record.journal,
                    record.doi,
                    record.pmid,
                    format_label(record.label),
                    format_number(assessment.score),
                    format_number(assessment.p_include),
                    assessment.decision.value,
                    clean_text(assessment.reason or ""),
                ]
            )


def _rows(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    if not read_text(path).strip():
        raise ResultsFileError(f"{path.name} is empty.")
    table = read_delimited(path)
    headers = [header.strip().lower() for header in table.headers]
    missing = [column for column in required if column not in headers]
    if missing:
        raise ResultsFileError(f"{path.name} is missing the column(s): {', '.join(missing)}.")
    return [dict(zip(headers, row, strict=True)) for row in table.rows]


def _record(row: dict[str, str]) -> Record:
    year = row.get("year", "").strip()
    return Record(
        id=row["id"].strip(),
        title=row.get("title", ""),
        authors=split_authors(row.get("authors", "")),
        year=int(year) if year.isdigit() else None,
        journal=row.get("journal", ""),
        doi=row.get("doi", ""),
        pmid=row.get("pmid", ""),
        label=parse_label(row.get("label")),
    )


def _score(row: dict[str, str], path: Path) -> float:
    try:
        return float(row["score"])
    except ValueError:
        raise ResultsFileError(
            f"{path.name}: record {row['id']!r} has a score that is not a number: {row['score']!r}."
        ) from None


def read_scores(path: Path) -> tuple[list[Record], list[float]]:
    """Read records and their scores from a results file or any CSV with id and score columns.

    Returns:
        The records (with labels when the file has a ``label`` column) and their scores.
    """
    rows = _rows(path, ("id", "score"))
    return [_record(row) for row in rows], [_score(row, path) for row in rows]


def read_results(path: Path) -> list[Record]:
    """Read a results file written by :func:`write_results`, with decisions and reasons."""
    rows = _rows(path, ("id", "score", "decision"))
    records = []
    for row in rows:
        record = _record(row)
        try:
            decision = Decision(row["decision"].strip().lower())
        except ValueError:
            raise ResultsFileError(
                f"{path.name}: record {record.id!r} has an unknown decision {row['decision']!r}."
            ) from None
        record.assessment = Assessment(
            score=_score(row, path), decision=decision, reason=row.get("reason") or None
        )
        records.append(record)
    return records
