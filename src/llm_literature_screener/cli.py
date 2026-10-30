"""The ``lls`` command line: screen, calibrate, evaluate, report and ui."""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from llm_literature_screener import __version__
from llm_literature_screener.backends import BACKENDS, DEFAULT_BASE_URL, BackendError, create_scorer
from llm_literature_screener.backends.base import Scorer, ScorerInfo
from llm_literature_screener.calibrate import (
    DEFAULT_TARGET_RECALL,
    DEFAULT_UPPER_THRESHOLD,
    CalibrationError,
    calibrate,
    fixed_threshold,
    recall_ladder,
)
from llm_literature_screener.criteria import Criteria, CriteriaError
from llm_literature_screener.dedup import find_duplicates
from llm_literature_screener.io import (
    FORMATS,
    ColumnMappingError,
    TableFileError,
    UnknownFormatError,
    apply_labels,
    read_labels,
    read_scores,
    write_results,
)
from llm_literature_screener.io.labels import LabelFileError
from llm_literature_screener.io.results import ResultsFileError
from llm_literature_screener.metrics import Evaluation, evaluate
from llm_literature_screener.pipeline import (
    REPORT_FILE,
    RESULTS_FILE,
    SUMMARY_FILE,
    NoRecordsError,
    ScreeningOptions,
    StageProgress,
    combine_sources,
    load_saved_run,
    load_source,
    report_from_results,
    results_path,
    run_screening,
    unknown_scorer,
    write_outputs,
)
from llm_literature_screener.record import Assessment, Record
from llm_literature_screener.report import ReportData, write_report

app = typer.Typer(
    name="lls",
    help=(
        "Pre-screen titles and abstracts for systematic reviews with an open LLM.\n\n"
        "The model answers Yes or No for each record; the score is logit(Yes) - logit(No), and "
        "a threshold calibrated on records your team has already screened keeps a target share "
        "of the known inclusions. People make the final decisions."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

out = Console()
err = Console(stderr=True)

STAGES = {"scoring": "Scoring records", "reasons": "Writing reasons"}
BACKEND_PANEL = "Model backend"
INPUT_PANEL = "Input"
DECISION_PANEL = "Threshold and decisions"
KNOWN_ERRORS = (
    BackendError,
    CalibrationError,
    ColumnMappingError,
    CriteriaError,
    FileNotFoundError,
    LabelFileError,
    NoRecordsError,
    ResultsFileError,
    TableFileError,
    UnknownFormatError,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=err, show_time=False, show_path=False, markup=False)],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@contextmanager
def _friendly_errors() -> Iterator[None]:
    """Turn expected errors into a one-line message and exit code 1."""
    try:
        yield
    except KNOWN_ERRORS as error:
        err.print(f"[bold red]Error:[/] {escape(str(error))}", highlight=False)
        raise typer.Exit(code=1) from None


@contextmanager
def _progress_bars() -> Iterator[StageProgress]:
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=err,
    )
    tasks: dict[str, TaskID] = {}

    def update(stage: str, done: int, total: int) -> None:
        if stage not in tasks:
            tasks[stage] = progress.add_task(STAGES.get(stage, stage), total=total)
        progress.update(tasks[stage], completed=done, total=total)

    with progress:
        yield update


def _parse_columns(values: list[str] | None) -> dict[str, str]:
    columns: dict[str, str] = {}
    for value in values or []:
        field_name, separator, header = value.partition("=")
        if not separator or not field_name.strip() or not header.strip():
            raise typer.BadParameter(
                f"{value!r} is not FIELD=COLUMN, for example title='Article Title'.",
                param_hint="--column",
            )
        columns[field_name.strip().lower()] = header.strip()
    return columns


def _check_unit_interval(value: float | None, name: str) -> None:
    if value is not None and not 0 < value <= 1:
        raise typer.BadParameter("must be greater than 0 and at most 1.", param_hint=name)


def _version(value: bool) -> None:
    if value:
        out.print(f"llm-literature-screener {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Pre-screen titles and abstracts for systematic reviews with an open LLM."""


# Options shared by the commands that run a model.
BackendOption = Annotated[
    str,
    typer.Option(
        "--backend",
        "-b",
        help="hf (local transformers model), openai (vLLM, llama.cpp, Ollama) or mock.",
        rich_help_panel=BACKEND_PANEL,
    ),
]
ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        "-m",
        help="Model id or path. Default for hf: Qwen/Qwen2.5-7B-Instruct.",
        rich_help_panel=BACKEND_PANEL,
    ),
]
DeviceOption = Annotated[
    str,
    typer.Option(help="hf: auto, cpu, cuda, cuda:1 or mps.", rich_help_panel=BACKEND_PANEL),
]
DtypeOption = Annotated[
    str,
    typer.Option(
        help="hf: auto (bf16 on GPU), bfloat16, float16 or float32.",
        rich_help_panel=BACKEND_PANEL,
    ),
]
TokenBudgetOption = Annotated[
    int,
    typer.Option(min=1, help="hf: padded tokens per batch.", rich_help_panel=BACKEND_PANEL),
]
BatchSizeOption = Annotated[
    int,
    typer.Option(min=1, help="hf: maximum prompts per batch.", rich_help_panel=BACKEND_PANEL),
]
BaseUrlOption = Annotated[
    str, typer.Option(help="openai: API base URL.", rich_help_panel=BACKEND_PANEL)
]
ApiKeyOption = Annotated[
    str | None,
    typer.Option(
        envvar="LLS_API_KEY",
        help="openai: API key, if the server needs one (or set LLS_API_KEY).",
        rich_help_panel=BACKEND_PANEL,
        show_default=False,
    ),
]
ConcurrencyOption = Annotated[
    int,
    typer.Option(min=1, help="openai: requests in flight at once.", rich_help_panel=BACKEND_PANEL),
]
MaxWordsOption = Annotated[
    int,
    typer.Option(
        min=1, help="Abstracts are cut to this many words.", rich_help_panel=BACKEND_PANEL
    ),
]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Show more log messages.")]


def _make_scorer(
    backend: str,
    model: str | None,
    device: str,
    dtype: str,
    token_budget: int,
    max_batch_size: int,
    base_url: str,
    api_key: str | None,
    concurrency: int,
    max_abstract_words: int,
) -> Scorer:
    if backend not in BACKENDS:
        raise typer.BadParameter(f"choose one of: {', '.join(BACKENDS)}.", param_hint="--backend")
    if backend == "hf":
        err.print(f"Loading {model or 'Qwen/Qwen2.5-7B-Instruct'} ...", markup=False)
    return create_scorer(
        backend,
        model,
        device=device,
        dtype=dtype,
        token_budget=token_budget,
        max_batch_size=max_batch_size,
        base_url=base_url,
        api_key=api_key,
        concurrency=concurrency,
        max_abstract_words=max_abstract_words,
    )


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


SCREEN_EPILOG = (
    "[bold]Examples[/]\n\n"
    "Try the bundled synthetic example with the mock backend (no model download):\n\n"
    "  [cyan]lls screen --records examples/records.ris --criteria examples/protocol.yaml "
    "--labels examples/labels.csv --backend mock --reasons --out results/[/]\n\n"
    "Screen with a local model on a GPU, keeping 95% of the known inclusions:\n\n"
    "  [cyan]lls screen --records refs.ris --criteria protocol.yaml --backend hf "
    "--model Qwen/Qwen2.5-7B-Instruct --labels seed.csv --target-recall 0.95 --out results/[/]\n\n"
    "Combine database exports (duplicates across files are removed) and use a vLLM server:\n\n"
    "  [cyan]lls screen --records pubmed.nbib --records embase.ris --criteria protocol.yaml "
    "--labels seed.csv --backend openai --model Qwen/Qwen2.5-7B-Instruct "
    "--base-url http://localhost:8000/v1 --out results/[/]\n\n"
    "Map the columns of a spreadsheet:\n\n"
    "  [cyan]lls screen --records refs.xlsx --column title='Article Title' "
    "--column abstract=Summary --criteria protocol.yaml --threshold -8 --out results/[/]"
)


@app.command(epilog=SCREEN_EPILOG)
def screen(
    records: Annotated[
        list[Path],
        typer.Option(
            "--records",
            "-r",
            exists=True,
            dir_okay=False,
            help="Records file: RIS, PubMed .nbib, CSV, TSV, Excel, Rayyan or Covidence export. "
            "Repeat to combine several files.",
            rich_help_panel=INPUT_PANEL,
        ),
    ],
    criteria: Annotated[
        Path,
        typer.Option(
            "--criteria",
            "-c",
            exists=True,
            dir_okay=False,
            help="Review protocol (YAML): question, eligibility criteria.",
            rich_help_panel=INPUT_PANEL,
        ),
    ],
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            "-l",
            exists=True,
            dir_okay=False,
            help="Decisions your team has already made (CSV/TSV/Excel with id, doi, pmid or "
            "title, and label). Used to calibrate the threshold.",
            rich_help_panel=INPUT_PANEL,
        ),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out", "-o", file_okay=False, help="Output directory.", rich_help_panel=INPUT_PANEL
        ),
    ] = Path("results"),
    fmt: Annotated[
        str | None,
        typer.Option(
            "--format",
            help=f"Records format if it cannot be detected: {', '.join(FORMATS)}.",
            rich_help_panel=INPUT_PANEL,
        ),
    ] = None,
    column: Annotated[
        list[str] | None,
        typer.Option(
            help="Column mapping for tables, FIELD=COLUMN, for example title='Article Title'. "
            "Fields: id, title, abstract, authors, year, journal, doi, pmid, label.",
            rich_help_panel=INPUT_PANEL,
        ),
    ] = None,
    sheet: Annotated[
        str | None,
        typer.Option(help="Excel worksheet (default: the first).", rich_help_panel=INPUT_PANEL),
    ] = None,
    title: Annotated[
        str | None, typer.Option(help="Title of the report.", rich_help_panel=INPUT_PANEL)
    ] = None,
    target_recall: Annotated[
        float,
        typer.Option(
            help="Share of the known inclusions the threshold must keep.",
            rich_help_panel=DECISION_PANEL,
        ),
    ] = DEFAULT_TARGET_RECALL,
    threshold: Annotated[
        float | None,
        typer.Option(
            help="Fixed exclusion threshold (score). Skips calibration.",
            rich_help_panel=DECISION_PANEL,
        ),
    ] = None,
    upper_threshold: Annotated[
        float,
        typer.Option(
            help="Flagged records scoring at least this are 'include', the others 'uncertain'.",
            rich_help_panel=DECISION_PANEL,
        ),
    ] = DEFAULT_UPPER_THRESHOLD,
    reasons: Annotated[
        bool,
        typer.Option(
            "--reasons/--no-reasons",
            help="Write a one-line reason for each include or uncertain record.",
            rich_help_panel=DECISION_PANEL,
        ),
    ] = False,
    dedup: Annotated[
        bool,
        typer.Option(
            "--dedup/--no-dedup",
            help="Remove duplicates by DOI, PMID and title plus year.",
            rich_help_panel=DECISION_PANEL,
        ),
    ] = True,
    backend: BackendOption = "hf",
    model: ModelOption = None,
    device: DeviceOption = "auto",
    dtype: DtypeOption = "auto",
    token_budget: TokenBudgetOption = 24000,
    max_batch_size: BatchSizeOption = 64,
    base_url: BaseUrlOption = DEFAULT_BASE_URL,
    api_key: ApiKeyOption = None,
    concurrency: ConcurrencyOption = 8,
    max_abstract_words: MaxWordsOption = 350,
    verbose: VerboseOption = False,
) -> None:
    """Screen records: deduplicate, score, calibrate, decide and write a report.

    Writes to the output directory: screening_results.csv (ranked), each records file in its
    own format with score, p_include, decision and reason added, prisma.md (PRISMA 2020 counts
    and a methods paragraph), report.html, calibration.json, duplicates.csv, criteria.yaml and
    run.json.
    """
    _setup_logging(verbose)
    _check_unit_interval(target_recall, "--target-recall")
    columns = _parse_columns(column)
    with _friendly_errors():
        protocol = Criteria.from_yaml(criteria)
        sources = [load_source(path, fmt, columns or None, sheet) for path in records]
        combined, source_of = combine_sources(sources)
        if labels is not None:
            report = apply_labels(combined, read_labels(labels))
            err.print(
                f"Labels: {report.matched} of {report.entries} matched a record.", highlight=False
            )
        options = ScreeningOptions(
            target_recall=target_recall,
            threshold=threshold,
            upper_threshold=upper_threshold,
            deduplicate=dedup,
            reasons=reasons,
        )
        if threshold is None and not any(record.label is True for record in combined):
            raise CalibrationError(
                "No record is labeled as a known inclusion, so the threshold cannot be "
                "calibrated. Pass --labels with decisions your team has already made, or set "
                "--threshold."
            )
        scorer = _make_scorer(
            backend,
            model,
            device,
            dtype,
            token_budget,
            max_batch_size,
            base_url,
            api_key,
            concurrency,
            max_abstract_words,
        )
        with _progress_bars() as progress:
            run = run_screening(
                combined,
                protocol,
                scorer,
                options,
                progress=progress,
                sources=[(source.path.name, len(source.records)) for source in sources],
                source_of=source_of,
                title=title,
            )
        paths = write_outputs(run, out_dir, sources)
    _print_run_summary(run.prisma.to_dict(), run.calibration.to_dict(), paths)
    for warning in [*run.calibration.warnings, *run.warnings]:
        err.print(f"[yellow]Warning:[/] {escape(warning)}", highlight=False)


def _print_run_summary(
    prisma: dict[str, object], calibration: dict[str, object], paths: dict[str, Path]
) -> None:
    table = Table(title="Screening summary", show_header=False, title_justify="left")
    table.add_column("What")
    table.add_column("Count", justify="right")
    table.add_row("Records identified", f"{prisma['identified']:,}")
    table.add_row("Duplicates removed", f"{prisma['duplicates_removed']:,}")
    table.add_row("Excluded by the tool", f"{prisma['excluded_by_tool']:,}")
    table.add_row(
        "Left for people to screen",
        f"{prisma['to_screen']:,} ({prisma['include']:,} include, "
        f"{prisma['uncertain']:,} uncertain)",
    )
    threshold = calibration["threshold"]
    table.add_row(
        "Exclusion threshold (score)", f"{threshold:.3f}" if isinstance(threshold, float) else ""
    )
    recall = calibration.get("recall")
    interval = calibration.get("recall_ci")
    if isinstance(recall, float) and isinstance(interval, list):
        table.add_row(
            "Known inclusions kept",
            f"{calibration['found']} of {calibration['known_inclusions']} = {_percent(recall)} "
            f"(95% CI {_percent(interval[0])} to {_percent(interval[1])})",
        )
    out.print(table)
    for label, kind in (("Report", "report"), ("Results", "results"), ("PRISMA", "prisma")):
        if kind in paths:
            out.print(
                f"{label + ':':9}{paths[kind]}", markup=False, highlight=False, soft_wrap=True
            )


CALIBRATE_EPILOG = (
    "[bold]Examples[/]\n\n"
    "See what each recall target would cost after labeling more records:\n\n"
    "  [cyan]lls calibrate results/ --labels more_labels.csv --target-recall 0.95[/]\n\n"
    "Save the calibration as JSON:\n\n"
    "  [cyan]lls calibrate results/screening_results.csv --out calibration.json[/]"
)


@app.command(name="calibrate", epilog=CALIBRATE_EPILOG)
def calibrate_command(
    scores: Annotated[
        Path,
        typer.Argument(
            exists=True, help="Results directory from lls screen, or a CSV with id and score."
        ),
    ],
    labels: Annotated[
        Path | None,
        typer.Option("--labels", "-l", exists=True, dir_okay=False, help="Extra human decisions."),
    ] = None,
    target_recall: Annotated[
        float, typer.Option(help="Share of the known inclusions to keep.")
    ] = DEFAULT_TARGET_RECALL,
    upper_threshold: Annotated[
        float, typer.Option(help="Score at or above which flagged records are 'include'.")
    ] = DEFAULT_UPPER_THRESHOLD,
    out_file: Annotated[
        Path | None,
        typer.Option("--out", "-o", dir_okay=False, help="Write the result as JSON."),
    ] = None,
    verbose: VerboseOption = False,
) -> None:
    """Calibrate the threshold on saved scores, with no model run.

    Prints the threshold for the target recall with a Wilson confidence interval, and a
    table of what other recall targets would cost in records left to screen.
    """
    _setup_logging(verbose)
    _check_unit_interval(target_recall, "--target-recall")
    with _friendly_errors():
        records, values = read_scores(results_path(scores))
        if labels is not None:
            apply_labels(records, read_labels(labels))
        known = [record.label for record in records]
        result = calibrate(values, known, target_recall, upper_threshold)
        ladder = recall_ladder(values, known)
    assert result.recall is not None and result.recall_ci is not None
    flagged = sum(1 for value in values if value >= result.threshold)
    out.print(
        f"Threshold for {target_recall:.0%} recall: score {result.threshold:.3f}. It keeps "
        f"{result.found} of {result.known_inclusions} known inclusions "
        f"({_percent(result.recall)}, 95% CI {_percent(result.recall_ci[0])} to "
        f"{_percent(result.recall_ci[1])}) and leaves {flagged:,} of {len(values):,} records "
        f"({_percent(flagged / len(values))}) for people to screen.",
        highlight=False,
    )
    table = Table(title="Other recall targets", title_justify="left")
    for name in ("Target", "Threshold", "Recall (95% CI)", "Left to screen", "Workload saved"):
        table.add_column(name, justify="right")
    for row in ladder:
        table.add_row(
            f"{row.target_recall:.0%}",
            f"{row.threshold:.3f}",
            f"{_percent(row.recall)} ({_percent(row.recall_ci[0])}-{_percent(row.recall_ci[1])})",
            f"{row.flagged:,}",
            _percent(row.workload_saved),
        )
    out.print(table)
    for warning in result.warnings:
        err.print(f"[yellow]Warning:[/] {escape(warning)}", highlight=False)
    if out_file is not None:
        data = result.to_dict()
        data["ladder"] = [
            {
                "target_recall": row.target_recall,
                "threshold": row.threshold,
                "found": row.found,
                "recall": row.recall,
                "recall_ci": list(row.recall_ci),
                "flagged": row.flagged,
                "workload_saved": row.workload_saved,
            }
            for row in ladder
        ]
        out_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        out.print(f"Saved {out_file}", markup=False, highlight=False, soft_wrap=True)


EVALUATE_EPILOG = (
    "[bold]Examples[/]\n\n"
    "Benchmark saved scores against complete labels at a fixed threshold:\n\n"
    "  [cyan]lls evaluate --scores results/screening_results.csv --labels all_labels.csv "
    "--threshold -11.25[/]\n\n"
    "Score a fully labeled review and set the threshold on a different, development review:\n\n"
    "  [cyan]lls evaluate --records test_review.csv --criteria test.yaml --backend hf "
    "--dev-scores dev/screening_results.csv --out evaluation/[/]\n\n"
    "Try it with the mock backend on the bundled example:\n\n"
    "  [cyan]lls evaluate --records examples/records.ris --criteria examples/protocol.yaml "
    "--labels examples/labels.csv --backend mock --drop-unlabeled[/]"
)


@app.command(name="evaluate", epilog=EVALUATE_EPILOG)
def evaluate_command(
    scores: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            help="Saved scores: a results directory or CSV with id, score and label.",
            rich_help_panel=INPUT_PANEL,
        ),
    ] = None,
    records: Annotated[
        Path | None,
        typer.Option(
            "--records",
            "-r",
            exists=True,
            dir_okay=False,
            help="Records to score (with labels), instead of --scores.",
            rich_help_panel=INPUT_PANEL,
        ),
    ] = None,
    criteria: Annotated[
        Path | None,
        typer.Option(
            "--criteria",
            "-c",
            exists=True,
            dir_okay=False,
            help="Review protocol, needed with --records.",
            rich_help_panel=INPUT_PANEL,
        ),
    ] = None,
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            "-l",
            exists=True,
            dir_okay=False,
            help="Human decisions, if not in the scores or records file.",
            rich_help_panel=INPUT_PANEL,
        ),
    ] = None,
    column: Annotated[
        list[str] | None,
        typer.Option(help="Column mapping for tables, FIELD=COLUMN.", rich_help_panel=INPUT_PANEL),
    ] = None,
    drop_unlabeled: Annotated[
        bool,
        typer.Option(
            help="Leave out records without a human decision instead of stopping.",
            rich_help_panel=INPUT_PANEL,
        ),
    ] = False,
    dedup: Annotated[
        bool,
        typer.Option(
            "--dedup/--no-dedup", help="Remove duplicates first.", rich_help_panel=INPUT_PANEL
        ),
    ] = False,
    threshold: Annotated[
        float | None,
        typer.Option(help="Fixed exclusion threshold.", rich_help_panel=DECISION_PANEL),
    ] = None,
    dev_scores: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            help="Calibrate the threshold on these labeled scores (another review) instead.",
            rich_help_panel=DECISION_PANEL,
        ),
    ] = None,
    target_recall: Annotated[
        float,
        typer.Option(
            help="Recall target for calibration (on --dev-scores, or in-sample).",
            rich_help_panel=DECISION_PANEL,
        ),
    ] = DEFAULT_TARGET_RECALL,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            file_okay=False,
            help="Write metrics.json, the report and (when scoring) the results here.",
            rich_help_panel=INPUT_PANEL,
        ),
    ] = None,
    backend: BackendOption = "hf",
    model: ModelOption = None,
    device: DeviceOption = "auto",
    dtype: DtypeOption = "auto",
    token_budget: TokenBudgetOption = 24000,
    max_batch_size: BatchSizeOption = 64,
    base_url: BaseUrlOption = DEFAULT_BASE_URL,
    api_key: ApiKeyOption = None,
    concurrency: ConcurrencyOption = 8,
    max_abstract_words: MaxWordsOption = 350,
    verbose: VerboseOption = False,
) -> None:
    """Benchmark against a fully labeled set: ROC AUC, WSS@95, recall, workload, kappa.

    Use a threshold set elsewhere (--threshold, or --dev-scores from another review) for an
    honest estimate. Without either, the threshold is calibrated on the evaluated records
    themselves, so the recall it reaches is optimistic by construction.
    """
    _setup_logging(verbose)
    _check_unit_interval(target_recall, "--target-recall")
    if (scores is None) == (records is None):
        raise typer.BadParameter("give either --scores or --records.", param_hint="--scores")
    if records is not None and criteria is None:
        raise typer.BadParameter("--records needs --criteria.", param_hint="--criteria")
    if threshold is not None and dev_scores is not None:
        raise typer.BadParameter(
            "use --threshold or --dev-scores, not both.", param_hint="--threshold"
        )
    with _friendly_errors():
        protocol: Criteria | None = None
        info = _saved_info(results_path(scores)) if scores is not None else None
        if scores is not None:
            items, values = read_scores(results_path(scores))
        else:
            assert records is not None and criteria is not None
            protocol = Criteria.from_yaml(criteria)
            items = load_source(records, columns=_parse_columns(column) or None).records
        if labels is not None:
            apply_labels(items, read_labels(labels))
        if scores is None:
            if dedup:
                items = find_duplicates(items).records
            items, _ = _labeled(items, [0.0] * len(items), drop_unlabeled)
            assert protocol is not None
            scorer = _make_scorer(
                backend,
                model,
                device,
                dtype,
                token_budget,
                max_batch_size,
                base_url,
                api_key,
                concurrency,
                max_abstract_words,
            )
            with _progress_bars() as progress:
                values = scorer.score(
                    items, protocol, progress=lambda done, total: progress("scoring", done, total)
                )
            info = scorer.info
        else:
            items, values = _labeled(items, values, drop_unlabeled)
        truth = [record.label is True for record in items]
        if not any(truth):
            raise LabelFileError(
                "None of the evaluated records is labeled as an inclusion, so recall cannot "
                "be measured."
            )
        if threshold is not None:
            cut = threshold
            note = "fixed threshold"
        elif dev_scores is not None:
            dev_records, dev_values = read_scores(results_path(dev_scores))
            cut = calibrate(dev_values, [r.label for r in dev_records], target_recall).threshold
            note = f"threshold calibrated on {dev_scores.name} for {target_recall:.0%} recall"
        else:
            cut = calibrate(values, list(truth), target_recall).threshold
            note = (
                f"threshold calibrated in-sample for {target_recall:.0%} recall (optimistic: "
                "use --dev-scores or --threshold for an honest estimate)"
            )
        result = evaluate(values, truth, cut)
    _print_evaluation(result, note)
    if out_dir is not None:
        assert info is not None
        _write_evaluation(
            out_dir, items, values, truth, result, note, info, protocol, scores is None
        )


def _saved_info(csv_path: Path) -> ScorerInfo:
    """Describe the model behind saved scores, from run.json when it is next to them."""
    if (csv_path.parent / SUMMARY_FILE).is_file():
        saved = load_saved_run(csv_path)
        if saved.info is not None:
            return saved.info
    return unknown_scorer(csv_path)


def _labeled(
    records: list[Record], values: list[float], drop_unlabeled: bool
) -> tuple[list[Record], list[float]]:
    missing = sum(1 for record in records if record.label is None)
    if missing and not drop_unlabeled:
        raise LabelFileError(
            f"{missing} of {len(records)} records have no human decision. lls evaluate needs a "
            "fully labeled set: add --labels, or pass --drop-unlabeled to leave them out."
        )
    if missing:
        err.print(f"Leaving out {missing} records without a human decision.", highlight=False)
    kept = [(r, v) for r, v in zip(records, values, strict=True) if r.label is not None]
    if not kept:
        raise LabelFileError("No record has a human decision.")
    return [r for r, _ in kept], [v for _, v in kept]


def _print_evaluation(result: Evaluation, note: str) -> None:
    table = Table(
        title="Evaluation against complete labels", show_header=False, title_justify="left"
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    low, high = result.recall_ci
    table.add_row("Records (inclusions)", f"{result.records:,} ({result.inclusions:,})")
    table.add_row("ROC AUC", "n/a" if result.auc is None else f"{result.auc:.4f}")
    table.add_row("WSS@95", f"{result.wss95:.4f}")
    table.add_row(
        "Read to find 95% / 100%", f"{result.screened_for_95:,} / {result.screened_for_100:,}"
    )
    table.add_row("Threshold", f"{result.threshold:.3f}")
    table.add_row(
        "Recall at threshold",
        f"{result.found} of {result.inclusions} = {_percent(result.recall)} "
        f"(95% CI {_percent(low)} to {_percent(high)})",
    )
    table.add_row("Workload saved", _percent(result.workload_saved))
    table.add_row("Precision", "n/a" if result.precision is None else _percent(result.precision))
    table.add_row("Cohen's kappa", "n/a" if result.kappa is None else f"{result.kappa:.3f}")
    out.print(table)
    out.print(f"Note: {note}.", markup=False, highlight=False)


def _write_evaluation(
    out_dir: Path,
    records: list[Record],
    values: list[float],
    truth: list[bool],
    result: Evaluation,
    note: str,
    info: ScorerInfo,
    protocol: Criteria | None,
    scored: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = result.to_dict()
    metrics["note"] = note
    metrics["scorer"] = {"backend": info.backend, "model": info.model}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    calibration = fixed_threshold(values, list(truth), result.threshold)
    for record, value in zip(records, values, strict=True):
        record.assessment = Assessment(value, calibration.decide(value))
    if scored:
        write_results(records, out_dir / RESULTS_FILE)
    data = ReportData(
        title="Evaluation",
        records=records,
        calibration=calibration,
        info=info,
        tool_version=__version__,
        generated_at=datetime.now(),
        criteria=protocol,
        warnings=[f"Threshold: {note}."],
        evaluation=result,
    )
    write_report(data, out_dir / REPORT_FILE)
    out.print(
        f"Saved {out_dir / 'metrics.json'} and {out_dir / REPORT_FILE}",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


REPORT_EPILOG = (
    "[bold]Examples[/]\n\n"
    "Rebuild the report of a screening run:\n\n"
    "  [cyan]lls report results/[/]\n\n"
    "Recalibrate with more labels and write a new report, without running the model again:\n\n"
    "  [cyan]lls report results/ --labels more_labels.csv --target-recall 0.95 "
    "--out results/report_v2.html[/]\n\n"
    "Report on scores from any tool (a CSV with id, score and label columns):\n\n"
    "  [cyan]lls report scores.csv --threshold -3 --out report.html[/]"
)


@app.command(name="report", epilog=REPORT_EPILOG)
def report_command(
    results: Annotated[
        Path,
        typer.Argument(exists=True, help="Results directory from lls screen, or a scores CSV."),
    ],
    out_file: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            dir_okay=False,
            help="Report file (default: report.html beside the results).",
        ),
    ] = None,
    labels: Annotated[
        Path | None,
        typer.Option("--labels", "-l", exists=True, dir_okay=False, help="Extra human decisions."),
    ] = None,
    target_recall: Annotated[
        float | None, typer.Option(help="Recalibrate for this recall target.")
    ] = None,
    threshold: Annotated[float | None, typer.Option(help="Use this fixed threshold.")] = None,
    upper_threshold: Annotated[
        float | None, typer.Option(help="Score at or above which flagged records are 'include'.")
    ] = None,
    title: Annotated[str | None, typer.Option(help="Title of the report.")] = None,
    verbose: VerboseOption = False,
) -> None:
    """Write the HTML report for saved results, optionally recalibrated.

    The report is a single self-contained HTML file: summary numbers, decision counts, the
    recall curve, calibration, PRISMA 2020 counts with a methods paragraph, and the list of
    uncertain records.
    """
    _setup_logging(verbose)
    _check_unit_interval(target_recall, "--target-recall")
    with _friendly_errors():
        data = report_from_results(
            results,
            labels=labels,
            threshold=threshold,
            target_recall=target_recall,
            upper_threshold=upper_threshold,
            title=title,
        )
        destination = out_file or results_path(results).parent / REPORT_FILE
        write_report(data, destination)
    out.print(f"Report: {destination}", markup=False, highlight=False, soft_wrap=True)


UI_EPILOG = "[bold]Example[/]\n\n  [cyan]lls ui --port 8501[/]"


@app.command(epilog=UI_EPILOG)
def ui(
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port of the web app.")] = 8501,
    headless: Annotated[bool, typer.Option(help="Do not open a browser window.")] = False,
) -> None:
    """Start the web app (Streamlit) to upload records, run screening and download results."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        err.print(
            "[bold red]Error:[/] the web app needs Streamlit: "
            + escape('pip install "llm-literature-screener[ui]"'),
            highlight=False,
        )
        raise typer.Exit(code=1) from None
    app_path = Path(__file__).parent / "ui" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true" if headless else "false",
    ]
    raise typer.Exit(code=subprocess.call(command, env=os.environ.copy()))


if __name__ == "__main__":  # pragma: no cover
    app()
