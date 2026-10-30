"""A self-contained HTML screening report.

The report is one HTML file with inline CSS, an inline SVG chart and a few lines of inline
JavaScript for the chart's hover readout. It loads nothing from the network, so it can be
archived with the review, emailed or opened offline. It shows the summary numbers, the
decision counts, the recall-vs-screened curve on the known inclusions, the calibration, the
PRISMA 2020 counts with a methods paragraph, and the uncertain records.
"""

from __future__ import annotations

import html
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from llm_literature_screener.backends.base import ScorerInfo
from llm_literature_screener.calibrate import Calibration
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.metrics import Evaluation, evaluation_order, rank_order, recall_curve
from llm_literature_screener.prisma import PrismaCounts
from llm_literature_screener.record import Decision, Record, sigmoid
from llm_literature_screener.text import plural

MAX_LISTED = 500
"""At most this many uncertain records are listed; the results CSV has all of them."""

REPO_URL = "https://github.com/Hedmm/llm-literature-screener"


@dataclass
class ReportData:
    """Everything the report shows.

    Attributes:
        title: Report heading, usually the review's name.
        records: Assessed records (each with an assessment).
        calibration: Thresholds and their recall on the known inclusions.
        info: The scorer that produced the scores.
        tool_version: Version of this package.
        generated_at: When the report was made.
        criteria: The review protocol, shown in a collapsible section.
        prisma: PRISMA 2020 counts for the screening stage.
        methods: Methods paragraph for the review's write-up.
        warnings: Messages that affect how far the results can be trusted.
        evaluation: Metrics against complete labels, for ``lls evaluate``.
        seconds: Scoring time, if known.
    """

    title: str
    records: Sequence[Record]
    calibration: Calibration
    info: ScorerInfo
    tool_version: str
    generated_at: datetime
    criteria: Criteria | None = None
    prisma: PrismaCounts | None = None
    methods: str | None = None
    warnings: Sequence[str] = field(default_factory=list)
    evaluation: Evaluation | None = None
    seconds: float | None = None


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _score(value: float) -> str:
    return f"{value:.2f}".replace("-", "−")


def _probability(value: float) -> str:
    if value >= 0.9995:
        return "&gt;0.999"
    if value < 0.001:
        return "&lt;0.001"
    return f"{value:.3f}"


def _assessed(records: Sequence[Record]) -> list[tuple[Record, float, Decision, str | None]]:
    rows = []
    for record in records:
        if record.assessment is None:
            raise ValueError(f"Record {record.id} has no assessment; score it before reporting.")
        assessment = record.assessment
        rows.append((record, assessment.score, assessment.decision, assessment.reason))
    return rows


# ----------------------------------------------------------------------------- page parts


def _tile(label: str, value: str, note: str) -> str:
    return (
        '<div class="tile">'
        f'<div class="tile-label">{_e(label)}</div>'
        f'<div class="tile-value">{value}</div>'
        f'<div class="tile-note">{note}</div>'
        "</div>"
    )


def _summary_tiles(data: ReportData, counts: dict[Decision, int]) -> str:
    total = sum(counts.values())
    flagged = counts[Decision.INCLUDE] + counts[Decision.UNCERTAIN]
    excluded = counts[Decision.EXCLUDE]
    calibration = data.calibration
    tiles = [
        _tile(
            "Records screened by the tool",
            f"{total:,}",
            f"{plural(data.prisma.duplicates_removed, 'duplicate')} removed first"
            if data.prisma
            else "unique records",
        ),
        _tile(
            "Left for people to screen",
            f"{flagged:,}",
            f"{_pct(flagged / total) if total else '0%'} of records: "
            f"{counts[Decision.INCLUDE]:,} include, {counts[Decision.UNCERTAIN]:,} uncertain",
        ),
        _tile(
            "Excluded by the tool",
            f"{excluded:,}",
            f"{_pct(excluded / total) if total else '0%'} of records nobody has to read",
        ),
    ]
    if calibration.recall is not None and calibration.recall_ci is not None:
        low, high = calibration.recall_ci
        tiles.append(
            _tile(
                "Known inclusions kept",
                f"{calibration.found:,} of {calibration.known_inclusions:,}",
                f"recall {_pct(calibration.recall)} "
                f"({calibration.confidence:.0%} CI {_pct(low)} to {_pct(high)})",
            )
        )
    else:
        tiles.append(
            _tile("Known inclusions kept", "unknown", "no labeled inclusions to check against")
        )
    return f'<section class="tiles" aria-label="Summary">{"".join(tiles)}</section>'


_WARNING_ICON = (
    '<svg class="icon" viewBox="0 0 16 16" aria-hidden="true">'
    '<path d="M8 1.5 15 14H1z" fill="var(--warning)"/>'
    '<path d="M8 6v3.6" stroke="#0b0b0b" stroke-width="1.6" stroke-linecap="round"/>'
    '<circle cx="8" cy="11.6" r="0.9" fill="#0b0b0b"/></svg>'
)


def _warnings(messages: Sequence[str]) -> str:
    if not messages:
        return ""
    items = "".join(
        f'<li><span class="status">{_WARNING_ICON}<strong>Warning</strong></span> '
        f"{_e(message)}</li>"
        for message in messages
    )
    return f'<section aria-label="Warnings"><ul class="warnings">{items}</ul></section>'


def _decision_section(data: ReportData, counts: dict[Decision, int]) -> str:
    total = sum(counts.values())
    calibration = data.calibration
    meaning = {
        Decision.INCLUDE: (
            f"Score at or above {_score(calibration.upper_threshold)}: the model leans towards "
            "including the record."
        ),
        Decision.UNCERTAIN: (
            f"Score between {_score(calibration.threshold)} and "
            f"{_score(calibration.upper_threshold)}: the model leans against the record, but "
            "it is above the calibrated threshold, so people still screen it."
        ),
        Decision.EXCLUDE: (
            f"Score below {_score(calibration.threshold)}: marked as ineligible by the tool."
        ),
    }
    blocks = []
    legend = []
    rows = []
    for decision in (Decision.INCLUDE, Decision.UNCERTAIN, Decision.EXCLUDE):
        count = counts[decision]
        share = count / total if total else 0.0
        name = decision.value.capitalize()
        if count:
            label = _pct(share, 0) if share >= 0.08 else ""
            blocks.append(
                f'<div class="segment seg-{decision.value}" style="flex-grow:{count}" '
                f'title="{_e(name)}: {count:,} records ({_pct(share)})">'
                f"<span>{label}</span></div>"
            )
        legend.append(f'<span class="key"><i class="swatch seg-{decision.value}"></i>{name}</span>')
        rows.append(
            f"<tr><td><div class='record-title'>{name}</div>"
            f"<div class='muted small'>{_e(meaning[decision])}</div></td>"
            f"<td class='num'>{count:,}</td><td class='num'>{_pct(share)}</td></tr>"
        )
    return (
        '<section class="card"><h2>Decisions</h2>'
        f'<div class="segmented" role="img" aria-label="Decisions: '
        f"{counts[Decision.INCLUDE]:,} include, {counts[Decision.UNCERTAIN]:,} uncertain, "
        f'{counts[Decision.EXCLUDE]:,} exclude">{"".join(blocks)}</div>'
        f'<div class="legend">{"".join(legend)}</div>'
        "<table><thead><tr><th>Decision</th><th class='num'>Records</th>"
        "<th class='num'>Share</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


# Chart geometry in SVG user units. The chart is at most 760 px wide, and on narrow screens
# the stylesheet enlarges its text so that labels stay readable after scaling down.
_W, _H = 640, 420
_LEFT, _RIGHT, _TOP, _BOTTOM = 76, 30, 44, 76
_PW, _PH = _W - _LEFT - _RIGHT, _H - _TOP - _BOTTOM


def _sx(x: float) -> float:
    return round(_LEFT + x * _PW, 2)


def _sy(y: float) -> float:
    return round(_TOP + (1 - y) * _PH, 2)


def _curve_svg(
    curve: Sequence[tuple[float, float]],
    marker: tuple[float, float] | None,
    target: float | None,
) -> str:
    parts = [
        f'<svg class="chart" viewBox="0 0 {_W} {_H}" role="img" '
        'aria-labelledby="curve-title curve-desc">',
        '<title id="curve-title">Recall versus share of records read</title>',
        '<desc id="curve-desc">Share of the known inclusions found as records are read in '
        "the model's order, compared with reading in random order.</desc>",
    ]
    for tick in range(6):
        value = tick / 5
        y, x = _sy(value), _sx(value)
        parts.append(f'<line class="grid" x1="{_LEFT}" x2="{_W - _RIGHT}" y1="{y}" y2="{y}"/>')
        parts.append(
            f'<text class="tick" x="{_LEFT - 10}" y="{y + 4}" text-anchor="end">{tick * 20}%</text>'
        )
        parts.append(
            f'<text class="tick" x="{x}" y="{_H - _BOTTOM + 26}" '
            f'text-anchor="middle">{tick * 20}%</text>'
        )
    parts.append(
        f'<line class="axis" x1="{_LEFT}" x2="{_W - _RIGHT}" y1="{_sy(0)}" y2="{_sy(0)}"/>'
        f'<text class="axis-title" x="{_LEFT + _PW / 2}" y="{_H - 14}" text-anchor="middle">'
        "Share of records read, in the model's order</text>"
        f'<text class="axis-title" x="12" y="{_TOP - 20}">'
        "Share of known inclusions found</text>"
    )
    # Reading in random order finds inclusions in proportion to the records read.
    angle = -math.degrees(math.atan2(_PH, _PW))
    parts.append(
        f'<line class="reference" x1="{_sx(0)}" y1="{_sy(0)}" x2="{_sx(1)}" y2="{_sy(1)}"/>'
        f'<text class="ref-label" x="{_sx(0.74)}" y="{_sy(0.74) + 26}" text-anchor="middle" '
        f'transform="rotate({angle:.1f} {_sx(0.74)} {_sy(0.74) + 26})">random order</text>'
    )
    if target is not None:
        parts.append(
            f'<line class="target" x1="{_LEFT}" x2="{_W - _RIGHT}" y1="{_sy(target)}" '
            f'y2="{_sy(target)}"/><text class="ref-label" x="{_W - _RIGHT}" '
            f'y="{_sy(target) + 15}" text-anchor="end">{target:.0%} target</text>'
        )
    path = " ".join(f"{'M' if i == 0 else 'L'}{_sx(x)} {_sy(y)}" for i, (x, y) in enumerate(curve))
    area = f"{path} L{_sx(1)} {_sy(0)} L{_sx(0)} {_sy(0)} Z"
    parts.append(f'<path class="wash" d="{area}"/><path class="curve" d="{path}"/>')
    if marker is not None:
        mx, my = marker
        anchor, dx = ("end", -12) if mx > 0.6 else ("start", 12)
        parts.append(
            f'<circle class="marker" cx="{_sx(mx)}" cy="{_sy(my)}" r="5"/>'
            f'<text class="marker-label" x="{_sx(mx) + dx}" y="{_sy(my) + 26}" '
            f'text-anchor="{anchor}">Threshold</text>'
        )
    parts.append(
        f'<line class="crosshair" x1="0" x2="0" y1="{_TOP}" y2="{_sy(0)}" visibility="hidden"/>'
        '<circle class="focus-dot" r="4" cx="0" cy="0" visibility="hidden"/>'
        f'<rect class="hit" x="{_LEFT}" y="{_TOP}" width="{_PW}" height="{_PH}"/>'
        "</svg>"
    )
    return "".join(parts)


def _value_at(curve: Sequence[tuple[float, float]], x: float) -> float:
    """Recall of the step curve after reading share ``x`` of the records."""
    value = 0.0
    for (x0, y0), (x1, y1) in itertools.pairwise(curve):
        if x0 <= x <= x1:
            return y1 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        value = y1
    return value


def _recall_section(
    data: ReportData, rows: Sequence[tuple[Record, float, Decision, str | None]]
) -> str:
    scores = [score for _, score, _, _ in rows]
    labels = [record.label is True for record, _, _, _ in rows]
    inclusions = sum(labels)
    heading = "<h2>Recall on the known inclusions</h2>"
    if not inclusions:
        return (
            f'<section class="card">{heading}<p class="muted">No record is labeled as an '
            "inclusion, so the curve cannot be drawn. Give the tool decisions your team has "
            "already made (a labels file) to see how early the model finds relevant records."
            "</p></section>"
        )
    curve = recall_curve(scores, labels)
    total = len(scores)
    threshold = data.calibration.threshold
    flagged = sum(1 for score in scores if score >= threshold)
    found = sum(
        1 for score, label in zip(scores, labels, strict=True) if label and score >= threshold
    )
    marker = (flagged / total, found / inclusions)
    svg = _curve_svg(curve, marker, data.calibration.target_recall)
    payload = json.dumps(
        {
            "points": [[round(x, 6), round(y, 6)] for x, y in curve],
            "records": total,
            "inclusions": inclusions,
            "left": _LEFT,
            "top": _TOP,
            "width": _PW,
            "height": _PH,
            "viewWidth": _W,
            "viewHeight": _H,
        }
    ).replace("</", "<\\/")
    table_rows = []
    for tenth in range(1, 11):
        x = tenth / 10
        y = _value_at(curve, x)
        table_rows.append(
            f"<tr><td class='num'>{_pct(x, 0)}</td><td class='num'>{round(x * total):,}</td>"
            f"<td class='num'>{round(y * inclusions):,} of {inclusions:,}</td>"
            f"<td class='num'>{_pct(y)}</td></tr>"
        )
    table_rows.append(
        f"<tr class='highlight'><td class='num'>{_pct(marker[0])}</td><td class='num'>"
        f"{flagged:,}</td><td class='num'>{found:,} of {inclusions:,}</td>"
        f"<td class='num'>{_pct(marker[1])}</td></tr>"
    )
    order = evaluation_order(scores, labels)
    last_found = max(position for position, index in enumerate(order, 1) if labels[index])
    caption = (
        f"Reading the records from the highest score down finds every known inclusion after "
        f"{last_found:,} of {total:,} records ({_pct(last_found / total)}). Among records with "
        "the same score, known inclusions are counted last, so the curve never flatters the model."
    )
    if data.calibration.labeled < total and data.evaluation is None:
        caption += (
            f" Only {data.calibration.labeled:,} records have a human decision, so the curve "
            "shows how early these known inclusions are found, not every relevant record."
        )
    return (
        f'<section class="card">{heading}'
        '<div class="legend">'
        '<span class="key"><i class="line-key"></i>Reading in the model\'s order</span>'
        '<span class="key"><i class="line-key reference-key"></i>Reading in random order</span>'
        f'<span class="key"><i class="dot-key"></i>Calibrated threshold: {_pct(marker[1])} '
        f"found after reading {_pct(marker[0])} of the records</span></div>"
        f'<figure class="chart-box" tabindex="0" aria-describedby="curve-caption">{svg}'
        '<div class="tooltip" role="status" hidden></div></figure>'
        f'<p id="curve-caption" class="muted">{_e(caption)}</p>'
        "<details><summary>Show the curve as a table</summary><table><thead><tr>"
        "<th class='num'>Share read</th><th class='num'>Records read</th>"
        "<th class='num'>Known inclusions found</th><th class='num'>Recall</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>"
        "<p class='muted'>The highlighted row is the calibrated threshold.</p></details>"
        f'<script type="application/json" id="curve-data">{payload}</script></section>'
    )


def _calibration_section(calibration: Calibration) -> str:
    rows: list[tuple[str, str]] = []
    if calibration.method == "target_recall" and calibration.target_recall is not None:
        rows.append(
            (
                "Threshold rule",
                f"The highest score that keeps at least {_pct(calibration.target_recall, 0)} "
                "of the known inclusions",
            )
        )
    else:
        rows.append(("Threshold rule", "Set by the user"))
    rows.extend(
        [
            (
                "Exclusion threshold",
                f"score {_score(calibration.threshold)} "
                f"(P(include) = {_probability(sigmoid(calibration.threshold))})",
            ),
            (
                "Include threshold",
                f"score {_score(calibration.upper_threshold)} "
                f"(P(include) = {_probability(sigmoid(calibration.upper_threshold))})",
            ),
            (
                "Labeled records",
                f"{calibration.labeled:,} ({calibration.known_inclusions:,} known inclusions)",
            ),
        ]
    )
    if calibration.recall is not None and calibration.recall_ci is not None:
        low, high = calibration.recall_ci
        rows.append(
            (
                "Recall on known inclusions",
                f"{calibration.found:,} of {calibration.known_inclusions:,} = "
                f"{_pct(calibration.recall)} ({calibration.confidence:.0%} Wilson CI "
                f"{_pct(low)} to {_pct(high)})",
            )
        )
    body = "".join(f"<tr><th scope='row'>{_e(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return (
        f'<section class="card"><h2>Calibration</h2><table class="facts">{body}</table></section>'
    )


def _evaluation_section(evaluation: Evaluation | None) -> str:
    if evaluation is None:
        return ""
    low, high = evaluation.recall_ci

    def optional(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    rows = [
        ("Records", f"{evaluation.records:,} ({evaluation.inclusions:,} inclusions)"),
        ("ROC AUC", optional(evaluation.auc)),
        ("WSS@95", f"{evaluation.wss95:.3f}"),
        (
            "Records read to find 95% / 100% of inclusions",
            f"{evaluation.screened_for_95:,} / {evaluation.screened_for_100:,}",
        ),
        (
            "Recall at the threshold",
            f"{evaluation.found:,} of {evaluation.inclusions:,} = {_pct(evaluation.recall)} "
            f"({evaluation.confidence:.0%} Wilson CI {_pct(low)} to {_pct(high)})",
        ),
        ("Workload saved", _pct(evaluation.workload_saved)),
        (
            "Precision of the flagged records",
            "n/a" if evaluation.precision is None else _pct(evaluation.precision),
        ),
        (
            f"Cohen's kappa against the {evaluation.kappa_reference}",
            optional(evaluation.kappa),
        ),
    ]
    body = "".join(f"<tr><th scope='row'>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows)
    return (
        '<section class="card"><h2>Evaluation against complete labels</h2>'
        f'<table class="facts">{body}</table></section>'
    )


def _prisma_section(counts: PrismaCounts | None, methods: str | None) -> str:
    if counts is None:
        return ""
    rows = [("Records identified", f"{counts.identified:,}")]
    rows.extend((f"from {name}", f"{count:,}") for name, count in counts.sources)
    rows.extend(
        [
            ("Duplicate records removed", f"{counts.duplicates_removed:,}"),
            ("Records marked as ineligible by automation tools", f"{counts.excluded_by_tool:,}"),
            ("Records screened by the review team", f"{counts.to_screen:,}"),
            ("Records excluded by the review team", "to be completed"),
        ]
    )
    body = "".join(
        f"<tr><th scope='row'>{_e(k)}</th><td class='num'>{_e(v)}</td></tr>" for k, v in rows
    )
    methods_html = (
        f'<h3>Methods paragraph</h3><p class="methods">{_e(methods)}</p>' if methods else ""
    )
    return (
        '<section class="card"><h2>PRISMA 2020: identification and screening</h2>'
        f'<table class="facts">{body}</table>{methods_html}</section>'
    )


def _uncertain_section(rows: Sequence[tuple[Record, float, Decision, str | None]]) -> str:
    scores = [score for _, score, _, _ in rows]
    rank = {index: position for position, index in enumerate(rank_order(scores), start=1)}
    uncertain = sorted(
        (index for index, row in enumerate(rows) if row[2] is Decision.UNCERTAIN),
        key=lambda index: rank[index],
    )
    heading = f"<h2>Uncertain records ({len(uncertain):,})</h2>"
    if not uncertain:
        return (
            f'<section class="card">{heading}<p class="muted">No uncertain records.</p></section>'
        )
    body = []
    for index in uncertain[:MAX_LISTED]:
        record, score, _, reason = rows[index]
        details = " · ".join(
            part for part in (str(record.year or ""), record.journal, f"id {record.id}") if part
        )
        body.append(
            f"<tr><td class='num'>{rank[index]:,}</td>"
            f"<td><div class='record-title'>{_e(record.title or '(no title)')}</div>"
            f"<div class='muted small'>{_e(details)}</div></td>"
            f"<td class='num'>{_score(score)}</td>"
            f"<td class='num'>{_probability(sigmoid(score))}</td>"
            f"<td>{_e(reason) if reason else '<span class=muted>none</span>'}</td></tr>"
        )
    note = (
        f"<p class='muted'>Showing the first {MAX_LISTED:,} of {len(uncertain):,}; "
        "screening_results.csv lists every record.</p>"
        if len(uncertain) > MAX_LISTED
        else ""
    )
    return (
        f'<section class="card">{heading}<p class="muted">Above the calibrated threshold, but '
        "the model leans against them. Ranked by score, highest first.</p>"
        "<div class='table-scroll'><table class='records'><thead><tr><th class='num'>Rank</th>"
        "<th>Record</th><th class='num'>Score</th><th class='num'>P(include)</th>"
        f"<th>Reason</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>{note}</section>"
    )


def _criteria_section(criteria: Criteria | None) -> str:
    if criteria is None:
        return ""
    return (
        '<section class="card"><details><summary>Review question and criteria shown to the '
        f'model</summary><pre class="criteria">{_e(criteria.render())}</pre></details></section>'
    )


def build_report(data: ReportData) -> str:
    """Return the complete HTML document for the report."""
    rows = _assessed(data.records)
    counts = dict.fromkeys(Decision, 0)
    for _, _, decision, _ in rows:
        counts[decision] += 1
    timing = f" · scored in {data.seconds:.1f} s" if data.seconds is not None else ""
    subtitle = data.criteria.question if data.criteria else ""
    warnings = [*data.calibration.warnings, *data.warnings]
    sections = [
        _summary_tiles(data, counts),
        '<p class="notice"><strong>A pre-screen, not a decision.</strong> The tool ranks the '
        "records and marks those below a calibrated threshold as ineligible. People screen "
        "every other record and make all final eligibility decisions. Precision is low by "
        "design: the threshold is set for recall.</p>",
        _warnings(warnings),
        _decision_section(data, counts),
        _recall_section(data, rows),
        _evaluation_section(data.evaluation),
        _calibration_section(data.calibration),
        _prisma_section(data.prisma, data.methods),
        _uncertain_section(rows),
        _criteria_section(data.criteria),
    ]
    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(data.title)} · screening report</title><style>{_CSS}</style></head><body>"
        "<main>"
        f"<header><p class='eyebrow'>Screening report</p><h1>{_e(data.title)}</h1>"
        + (f"<p class='subtitle'>{_e(subtitle)}</p>" if subtitle else "")
        + f"<p class='meta'>Model {_e(data.info.model)} · {_e(data.info.description)}"
        f"{timing} · {_e(data.generated_at.strftime('%Y-%m-%d %H:%M'))}</p></header>"
        + "".join(section for section in sections if section)
        + "<footer>Made with <a href='"
        + REPO_URL
        + f"'>llm-literature-screener</a> {_e(data.tool_version)}. Scores rank records; "
        "they are not probabilities that a study belongs in the review.</footer>"
        f"</main><script>{_SCRIPT}</script></body></html>\n"
    )


def write_report(data: ReportData, path: Path) -> None:
    """Write the report to ``path`` (UTF-8)."""
    path.write_text(build_report(data), encoding="utf-8")


_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--text:#0b0b0b;--text-2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series:#2a78d6;
--wash:rgba(42,120,214,.10);--include:#1c5cab;--include-ink:#fff;--uncertain:#3987e5;
--uncertain-ink:#0b0b0b;--exclude:#86b6ef;--exclude-ink:#0b0b0b;--warning:#fab219;
--notice:#eef4fc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--text:#f0efec;--text-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--series:#3987e5;--wash:rgba(57,135,229,.14);
--include:#9ec5f4;--include-ink:#0b0b0b;--uncertain:#5598e7;--uncertain-ink:#0b0b0b;
--exclude:#256abf;--exclude-ink:#fff;--notice:#16202e}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--text:#f0efec;
--text-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--series:#3987e5;--wash:rgba(57,135,229,.14);--include:#9ec5f4;--include-ink:#0b0b0b;
--uncertain:#5598e7;--uncertain-ink:#0b0b0b;--exclude:#256abf;--exclude-ink:#fff;--notice:#16202e}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 48px}
header{margin-bottom:20px}
h1{font-size:28px;line-height:1.2;margin:4px 0 6px;font-weight:650}
h2{font-size:18px;margin:0 0 14px;font-weight:620}
h3{font-size:15px;margin:20px 0 8px}
.eyebrow{margin:0;color:var(--text-2);font-size:13px;text-transform:uppercase;letter-spacing:.06em}
.subtitle{margin:0 0 6px;color:var(--text-2);font-size:16px}
.meta,.muted{color:var(--text-2)}.meta{font-size:13px;margin:0}.small{font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;
margin:0 0 16px}
.tile,.card{background:var(--surface);border:1px solid var(--border);border-radius:12px}
.tile{padding:16px 18px}
.tile-label{color:var(--text-2);font-size:13px}
.tile-value{font-size:30px;font-weight:620;line-height:1.25;margin:4px 0 2px}
.tile-note{color:var(--text-2);font-size:13px}
.card{padding:20px 22px;margin:0 0 16px}
.notice{background:var(--notice);border:1px solid var(--border);border-radius:12px;
padding:12px 16px;margin:0 0 16px}
.warnings{list-style:none;padding:0;margin:0 0 16px}
.warnings li{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:10px 14px;margin:0 0 8px}
.status{display:inline-flex;align-items:center;gap:6px;margin-right:4px}
.icon{width:16px;height:16px;flex:none}
.segmented{display:flex;gap:2px;height:28px;margin:4px 0 10px}
.segment{display:flex;align-items:center;padding:0 8px;border-radius:4px;min-width:2px;
font-size:13px;font-weight:600;white-space:nowrap;overflow:visible}
.seg-include{background:var(--include);color:var(--include-ink)}
.seg-uncertain{background:var(--uncertain);color:var(--uncertain-ink)}
.seg-exclude{background:var(--exclude);color:var(--exclude-ink)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0 0 12px;color:var(--text-2);font-size:13px}
.key{display:inline-flex;align-items:center;gap:6px}
.swatch{display:inline-block;width:12px;height:12px;border-radius:3px}
.line-key{display:inline-block;width:18px;height:2px;border-radius:1px;background:var(--series)}
.reference-key{background:var(--muted)}
.dot-key{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--series);
box-shadow:0 0 0 2px var(--surface)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
thead th{color:var(--text-2);font-weight:560;font-size:13px}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.facts th{width:38%;font-weight:500;color:var(--text-2)}
.highlight td{font-weight:620}
.table-scroll{overflow-x:auto}
.records td:nth-child(2){min-width:260px}
.record-title{font-weight:520}
.methods{background:var(--page);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
pre.criteria{white-space:pre-wrap;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
background:var(--page);padding:12px;border-radius:8px;margin:12px 0 0}
details summary{cursor:pointer;color:var(--text-2);margin:8px 0}
.chart-box{position:relative;margin:0 auto;max-width:760px;outline:none}
.chart-box:focus-visible{box-shadow:0 0 0 2px var(--series);border-radius:6px}
.chart{display:block;width:100%;height:auto}
.chart text{font-family:inherit}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.axis-title{fill:var(--text-2);font-size:11.5px}
.reference{stroke:var(--muted);stroke-width:1.5}
.target{stroke:var(--muted);stroke-width:1}
.ref-label{fill:var(--muted);font-size:11px}
.curve{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.wash{fill:var(--wash);stroke:none}
.marker,.focus-dot{fill:var(--series);stroke:var(--surface);stroke-width:2}
.marker-label{fill:var(--text);font-size:11.5px;font-weight:560}
.crosshair{stroke:var(--text-2);stroke-width:1}
.hit{fill:transparent;cursor:crosshair}
.tooltip{position:absolute;pointer-events:none;background:var(--surface);color:var(--text);
border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:13px;
box-shadow:0 4px 14px rgba(0,0,0,.12);min-width:180px}
.tooltip strong{font-size:15px}
footer{color:var(--text-2);font-size:13px;margin-top:24px}
a{color:var(--series)}
@media (max-width:640px){h1{font-size:23px}.card{padding:16px}.tile-value{font-size:26px}
.tick,.ref-label{font-size:21px}.axis-title,.marker-label{font-size:22px}}
"""

_SCRIPT = """
(function () {
  var source = document.getElementById("curve-data");
  if (!source) { return; }
  var data = JSON.parse(source.textContent);
  var box = document.querySelector(".chart-box");
  var svg = box.querySelector("svg");
  var tip = box.querySelector(".tooltip");
  var line = svg.querySelector(".crosshair");
  var dot = svg.querySelector(".focus-dot");
  var hit = svg.querySelector(".hit");
  var pts = data.points;
  var current = null;
  function valueAt(x) {
    for (var i = 1; i < pts.length; i++) {
      var a = pts[i - 1], b = pts[i];
      if (x >= a[0] && x <= b[0]) {
        return b[0] === a[0] ? b[1] : a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]);
      }
    }
    return pts[pts.length - 1][1];
  }
  function pct(v) { return (v * 100).toFixed(1) + "%"; }
  function show(x) {
    current = Math.min(1, Math.max(0, x));
    var y = valueAt(current);
    var px = data.left + current * data.width;
    var py = data.top + (1 - y) * data.height;
    line.setAttribute("x1", px); line.setAttribute("x2", px);
    line.setAttribute("visibility", "visible");
    dot.setAttribute("cx", px); dot.setAttribute("cy", py);
    dot.setAttribute("visibility", "visible");
    var read = Math.round(current * data.records);
    var found = Math.round(y * data.inclusions);
    tip.textContent = "";
    var value = document.createElement("strong");
    value.textContent = pct(y) + " found";
    var detail = document.createElement("div");
    detail.textContent = found.toLocaleString() + " of " +
      data.inclusions.toLocaleString() + " known inclusions after reading " +
      read.toLocaleString() + " records (" + pct(current) + ")";
    tip.appendChild(value); tip.appendChild(detail);
    tip.hidden = false;
    var scale = box.clientWidth / data.viewWidth;
    var left = px * scale + 12;
    if (left + tip.offsetWidth > box.clientWidth) { left = px * scale - tip.offsetWidth - 12; }
    tip.style.left = left + "px";
    tip.style.top = Math.max(0, py * scale - tip.offsetHeight - 8) + "px";
  }
  function hide() {
    tip.hidden = true;
    line.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
  }
  hit.addEventListener("pointermove", function (event) {
    var rect = svg.getBoundingClientRect();
    var vx = (event.clientX - rect.left) * data.viewWidth / rect.width;
    show((vx - data.left) / data.width);
  });
  hit.addEventListener("pointerleave", hide);
  box.addEventListener("focus", function () { show(current === null ? 0.5 : current); });
  box.addEventListener("blur", hide);
  box.addEventListener("keydown", function (event) {
    var step = event.shiftKey ? 0.1 : 0.01;
    if (event.key === "ArrowRight") { show((current || 0) + step); event.preventDefault(); }
    if (event.key === "ArrowLeft") { show((current || 0) - step); event.preventDefault(); }
    if (event.key === "Escape") { hide(); }
  });
})();
"""
