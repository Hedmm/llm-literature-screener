"""Duplicate detection by DOI, PMID, and normalized title plus year.

Records that share a DOI, a PMID, or the same normalized title and publication year are
grouped (matches are transitive). In each group the first record in file order is kept and
the others are removed. Empty fields of the kept record are filled from its duplicates, so
a missing abstract can come from another copy. Every removal is reported with the reason.

Title matching is deliberately conservative: it needs a year on both records and a title of
at least four words, so that short generic titles such as "Editorial" are never merged.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from llm_literature_screener.io.common import normalize_title
from llm_literature_screener.record import Record

MIN_TITLE_WORDS = 4
"""Titles with fewer words are never used for matching."""


@dataclass(frozen=True)
class DuplicateMatch:
    """One record removed as a duplicate of another.

    Attributes:
        kept_id: Id of the record that was kept for the group.
        removed_id: Id of the removed duplicate.
        matched_id: Id of the record the removed one matched directly. It differs from
            ``kept_id`` when records are linked through a chain of matches.
        reason: ``doi``, ``pmid`` or ``title+year``: the key of that direct match.
        key: The shared value of that key.
    """

    kept_id: str
    removed_id: str
    matched_id: str
    reason: str
    key: str


@dataclass
class DedupResult:
    """Unique records and a report of the duplicates that were merged into them."""

    records: list[Record]
    matches: list[DuplicateMatch] = field(default_factory=list)

    @property
    def removed(self) -> int:
        """Number of duplicate records removed."""
        return len(self.matches)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            # The earlier record becomes the root, so it is the one that is kept.
            self.parent[max(a, b)] = min(a, b)


def title_key(record: Record) -> str:
    """Return the normalized title plus year used for matching, or "" if it is not usable."""
    title = normalize_title(record.title)
    if record.year is None or len(title.split()) < MIN_TITLE_WORDS:
        return ""
    return f"{title} ({record.year})"


def _doi_key(record: Record) -> str:
    return record.doi


def _pmid_key(record: Record) -> str:
    return record.pmid


def _merge(kept: Record, duplicates: Sequence[Record]) -> Record:
    """Fill empty fields of ``kept`` from its duplicates and combine their labels."""
    merged = replace(kept, authors=list(kept.authors), raw=list(kept.raw))
    for duplicate in duplicates:
        for name in ("title", "journal", "doi", "pmid"):
            if not getattr(merged, name) and getattr(duplicate, name):
                setattr(merged, name, getattr(duplicate, name))
        if len(duplicate.abstract) > len(merged.abstract):
            # The most complete abstract gives the model the most to go on.
            merged.abstract = duplicate.abstract
        if merged.year is None:
            merged.year = duplicate.year
        if not merged.authors:
            merged.authors = list(duplicate.authors)
    labels = [record.label for record in (kept, *duplicates) if record.label is not None]
    if labels:
        # A known inclusion anywhere in the group wins, so no inclusion is lost.
        merged.label = any(labels)
    return merged


def find_duplicates(
    records: Sequence[Record],
    *,
    by_doi: bool = True,
    by_pmid: bool = True,
    by_title: bool = True,
) -> DedupResult:
    """Group duplicate records and keep one record per group.

    Args:
        records: Records in file order.
        by_doi: Match records that share a DOI.
        by_pmid: Match records that share a PMID.
        by_title: Match records that share a normalized title and publication year.

    Returns:
        The unique records in their original order, and one :class:`DuplicateMatch` per
        removed record.
    """
    groups = _UnionFind(len(records))
    edges: list[tuple[int, int, str, str]] = []
    keyers: list[tuple[str, Callable[[Record], str]]] = []
    if by_doi:
        keyers.append(("doi", _doi_key))
    if by_pmid:
        keyers.append(("pmid", _pmid_key))
    if by_title:
        keyers.append(("title+year", title_key))
    for reason, key_of in keyers:
        first_seen: dict[str, int] = {}
        for index, record in enumerate(records):
            key = key_of(record)
            if not key:
                continue
            earlier = first_seen.setdefault(key, index)
            if earlier != index and groups.find(index) != groups.find(earlier):
                groups.union(index, earlier)
                edges.append((index, earlier, reason, key))

    # For each record, the first direct match that linked it to its group.
    link_of: dict[int, tuple[int, str, str]] = {}
    for later, earlier, reason, key in edges:
        link_of.setdefault(later, (earlier, reason, key))
        link_of.setdefault(earlier, (later, reason, key))

    members: dict[int, list[int]] = {}
    for index in range(len(records)):
        members.setdefault(groups.find(index), []).append(index)

    unique: list[Record] = []
    matches: list[DuplicateMatch] = []
    for root in sorted(members):
        group = members[root]
        kept = records[root]
        duplicates = [records[index] for index in group if index != root]
        unique.append(_merge(kept, duplicates) if duplicates else kept)
        for index in group:
            if index == root:
                continue
            other, reason, key = link_of[index]
            matches.append(
                DuplicateMatch(kept.id, records[index].id, records[other].id, reason, key)
            )
    return DedupResult(records=unique, matches=matches)


def write_dedup_report(result: DedupResult, records: Sequence[Record], path: Path) -> None:
    """Write one CSV row per removed duplicate, with both titles for a quick visual check."""
    titles = {record.id: record.title for record in records}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["kept_id", "removed_id", "matched_id", "reason", "key", "kept_title", "removed_title"]
        )
        for match in result.matches:
            writer.writerow(
                [
                    match.kept_id,
                    match.removed_id,
                    match.matched_id,
                    match.reason,
                    match.key,
                    titles.get(match.kept_id, ""),
                    titles.get(match.removed_id, ""),
                ]
            )
