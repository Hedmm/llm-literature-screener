"""Benchmark on the three SYNERGY reviews of the validated case study.

The design follows the case study: the threshold is calibrated on the development review
(Appenzeller-Herzog_2020: the highest score that keeps 95% of its final inclusions) and then
applied unchanged to the two test reviews (Wolters_2018 and van_Dis_2020). For each review
the script reports ROC AUC, WSS@95, recall with a Wilson interval, workload saved, precision
and, where the dataset has title and abstract decisions, Cohen's kappa against them.

The datasets are the ASReview / SYNERGY files the case study used, downloaded from GitHub and
checked against pinned SHA-256 checksums. Usage:

    python benchmarks/synergy.py                              # Qwen2.5-7B-Instruct, one GPU
    python benchmarks/synergy.py --backend openai --base-url http://localhost:8000/v1
    python benchmarks/synergy.py --from-case-study            # published scores, no GPU
    python benchmarks/synergy.py --backend mock --limit 300   # quick pipeline check on CPU

Results go to benchmarks/results/<run name>/: metrics.json, run_info.json, missed.csv and,
per review, screening_results.csv and report.html.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_literature_screener import Assessment, Record, __version__
from llm_literature_screener.backends import DEFAULT_BASE_URL, ScorerInfo, create_scorer
from llm_literature_screener.calibrate import calibrate, fixed_threshold, wilson_interval
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.io import read_records, write_results
from llm_literature_screener.io.common import parse_label
from llm_literature_screener.metrics import Evaluation, evaluate
from llm_literature_screener.report import ReportData, write_report

HERE = Path(__file__).resolve().parent
DEV = "Appenzeller-Herzog_2020"
TEST = ("Wolters_2018", "van_Dis_2020")
TARGET_RECALL = 0.95
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DATA_URL = (
    "https://raw.githubusercontent.com/asreview/systematic-review-datasets/metadata-v1-final/"
    "datasets/{name}/output/{name}.csv"
)
CASE_STUDY_COMMIT = "ac51389f1aa9f7557b8cace7e7cceaf2225f3dd4"
PREDICTIONS_URL = (
    f"https://raw.githubusercontent.com/Hedmm/ml-case-studies/{CASE_STUDY_COMMIT}/"
    "llm-screening-systematic-reviews/preds/{name}.csv"
)
CASE_STUDY_URL = (
    "https://github.com/Hedmm/ml-case-studies/tree/main/llm-screening-systematic-reviews"
)


@dataclass(frozen=True)
class Dataset:
    """One review: where its data comes from and what it must contain."""

    name: str
    sha256: str
    records: int
    inclusions: int
    predictions_sha256: str


DATASETS = {
    dataset.name: dataset
    for dataset in (
        Dataset(
            "Appenzeller-Herzog_2020",
            "456f9306b0ce1aee840ac0244f11df3b2a168aa46007a92f5dcf59916d309a69",
            3453,
            29,
            "844719b1550d417d1a08e3fd1e87634e52dd4973a1307bb4225a5c4ff0068744",
        ),
        Dataset(
            "Wolters_2018",
            "c2219342435e15fbc365fa463d04ee5d74fcf9ce9eb5dd7cfdbd2c5abd184d49",
            5019,
            19,
            "040b7a817e7eeff26ac02cbfa48698876b8de62d357907e89e8d8d85232290c0",
        ),
        Dataset(
            "van_Dis_2020",
            "394211b15e3830eb7237799e8a894ed8ef93793733366c355fc8e6615c84d15c",
            10953,
            73,
            "6075e14ef8d53d7bcac0c86c210300ff2ffafc376014ba5ec86ffd59b2116b75",
        ),
    )
}


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected_sha256: str) -> Path:
    """Download ``url`` to ``path`` unless a copy with the right checksum is already there.

    Raises:
        RuntimeError: If the downloaded file does not have the expected checksum.
    """
    if path.is_file() and sha256(path) == expected_sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{path.name} has checksum {actual}, expected {expected_sha256}. The source file "
            "changed, so the benchmark would not use the data of the case study."
        )
    return path


def load_review(path: Path, dataset: Dataset | None = None) -> list[Record]:
    """Read a review's CSV (record_id, title, abstract, ..., label_included).

    The final inclusion becomes the record's label. When ``dataset`` is given, the number of
    records and inclusions is checked against the case study.
    """
    records = read_records(path, "csv")
    if dataset is not None:
        inclusions = sum(1 for record in records if record.label is True)
        if (len(records), inclusions) != (dataset.records, dataset.inclusions):
            raise RuntimeError(
                f"{path.name}: expected {dataset.records} records with {dataset.inclusions} "
                f"inclusions, found {len(records)} with {inclusions}."
            )
    return records


def abstract_decisions(records: list[Record]) -> list[bool] | None:
    """Return the human title and abstract decisions, if the dataset has them."""
    if not records or not any(key == "label_abstract_screening" for key, _ in records[0].raw):
        return None
    return [parse_label(record.raw_value("label_abstract_screening")) is True for record in records]


def stratified_head(records: list[Record], limit: int) -> list[Record]:
    """Keep every inclusion plus the first other records, up to ``limit`` records in total."""
    if limit <= 0 or limit >= len(records):
        return records
    inclusions = {id(record) for record in records if record.label is True}
    room = max(0, limit - len(inclusions))
    kept, others = [], 0
    for record in records:
        if id(record) in inclusions:
            kept.append(record)
        elif others < room:
            kept.append(record)
            others += 1
    return kept


def load_case_study_scores(path: Path) -> dict[str, float]:
    """Read the case study's published scores (record_id, score)."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["record_id"]: float(row["score"]) for row in csv.DictReader(handle)}


def evaluation_row(name: str, role: str, result: Evaluation) -> dict[str, Any]:
    """Flatten an evaluation for the summary table."""
    return {"review": name, "role": role, **result.to_dict()}


def write_review_outputs(
    folder: Path,
    name: str,
    records: list[Record],
    scores: list[float],
    threshold: float,
    result: Evaluation,
    info: ScorerInfo,
    criteria: Criteria,
) -> None:
    """Write one review's ranked results and HTML report."""
    folder.mkdir(parents=True, exist_ok=True)
    labels: list[bool | None] = [record.label is True for record in records]
    calibration = fixed_threshold(scores, labels, threshold)
    for record, score in zip(records, scores, strict=True):
        record.assessment = Assessment(score, calibration.decide(score))
    write_results(records, folder / "screening_results.csv")
    write_report(
        ReportData(
            title=criteria.title or name,
            records=records,
            calibration=calibration,
            info=info,
            tool_version=__version__,
            generated_at=datetime.now(),
            criteria=criteria,
            warnings=[
                f"The threshold was calibrated on {DEV} for {TARGET_RECALL:.0%} recall of its "
                "final inclusions and applied unchanged to this review."
            ]
            if name != DEV
            else [],
            evaluation=result,
        ),
        folder / "report.html",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and return the process exit code."""
    parser = argparse.ArgumentParser(
        description="Benchmark on the three SYNERGY reviews of the validated case study."
    )
    parser.add_argument("--backend", default="hf", choices=["hf", "openai", "mock"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--token-budget", type=int, default=24000)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--from-case-study",
        action="store_true",
        help="Evaluate the case study's published scores instead of running a model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Keep every inclusion plus other records up to this many per review (quick test).",
    )
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=None, help="Output folder.")
    args = parser.parse_args(argv)

    if args.from_case_study:
        run_name = "case-study-predictions"
    elif args.backend == "mock":
        run_name = "mock"
    else:
        run_name = f"{args.backend}-{args.model.replace('/', '_')}"
    out_dir: Path = args.out or HERE / "results" / run_name

    scorer = None
    if not args.from_case_study:
        scorer = create_scorer(
            args.backend,
            args.model,
            device=args.device,
            dtype=args.dtype,
            token_budget=args.token_budget,
            base_url=args.base_url,
            api_key=args.api_key,
            concurrency=args.concurrency,
        )

    scored: dict[str, tuple[list[Record], list[float], list[bool] | None]] = {}
    timing: dict[str, Any] = {}
    info = ScorerInfo(
        backend="case study",
        model=DEFAULT_MODEL,
        description=f"published scores from {CASE_STUDY_URL}",
        score_definition='the difference between the logits of "Yes" and "No"',
    )
    for name in (DEV, *TEST):
        dataset = DATASETS[name]
        path = download(DATA_URL.format(name=name), args.data_dir / f"{name}.csv", dataset.sha256)
        records = load_review(path, dataset)
        if args.limit and not args.from_case_study:
            records = stratified_head(records, args.limit)
        criteria = Criteria.from_yaml(HERE / "protocols" / f"{name}.yaml")
        if scorer is None:
            predictions = download(
                PREDICTIONS_URL.format(name=name),
                args.data_dir / "case-study-predictions" / f"{name}.csv",
                dataset.predictions_sha256,
            )
            published = load_case_study_scores(predictions)
            scores = [published[record.id] for record in records]
        else:
            print(f"Scoring {name}: {len(records)} records", flush=True)
            started = time.perf_counter()
            scores = scorer.score(records, criteria)
            seconds = time.perf_counter() - started
            timing[name] = {"records": len(records), "seconds": round(seconds, 1)}
            info = scorer.info
        scored[name] = (records, scores, abstract_decisions(records))

    dev_records, dev_scores, _ = scored[DEV]
    calibration = calibrate(dev_scores, [r.label for r in dev_records], TARGET_RECALL)
    threshold = calibration.threshold
    rows, missed = [], []
    totals = {"records": 0, "inclusions": 0, "found": 0, "flagged": 0}
    for name, (records, scores, human) in scored.items():
        labels = [record.label is True for record in records]
        result = evaluate(scores, labels, threshold, human=human)
        role = "dev" if name == DEV else "test"
        rows.append(evaluation_row(name, role, result))
        if role == "test":
            totals["records"] += result.records
            totals["inclusions"] += result.inclusions
            totals["found"] += result.found
            totals["flagged"] += result.flagged
        criteria = Criteria.from_yaml(HERE / "protocols" / f"{name}.yaml")
        write_review_outputs(
            out_dir / name, name, records, scores, threshold, result, info, criteria
        )
        for record, score in zip(records, scores, strict=True):
            if record.label is True and score < threshold:
                missed.append(
                    {
                        "review": name,
                        "record_id": record.id,
                        "score": score,
                        "has_abstract": bool(record.abstract),
                        "year": record.year,
                        "title": record.title,
                    }
                )

    test_total = {
        **totals,
        "recall": totals["found"] / totals["inclusions"],
        "recall_ci": list(wilson_interval(totals["found"], totals["inclusions"])),
        "workload_saved": 1 - totals["flagged"] / totals["records"],
    }
    metrics = {
        "dev": DEV,
        "test": list(TEST),
        "target_recall": TARGET_RECALL,
        "threshold_from_dev": threshold,
        "reviews": rows,
        "test_total": test_total,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "missed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["review", "record_id", "score", "has_abstract", "year", "title"]
        )
        writer.writeheader()
        writer.writerows(missed)
    run_info = {
        "tool_version": __version__,
        "scorer": {"backend": info.backend, "model": info.model, "description": info.description},
        "limit": args.limit or None,
        "timing": timing,
        "warnings": scorer.warnings if scorer is not None else [],
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    print(f"\nThreshold from {DEV}: {threshold:.3f} (keeps {TARGET_RECALL:.0%} of its inclusions)")
    print(f"{'review':26} {'role':5} {'AUC':>6} {'WSS@95':>7} {'recall':>14} {'saved':>7}")
    for row in rows:
        auc = "n/a" if row["auc"] is None else f"{row['auc']:.3f}"
        recall = f"{row['found']}/{row['inclusions']} {row['recall']:.1%}"
        print(
            f"{row['review']:26} {row['role']:5} {auc:>6} {row['wss95']:>7.3f} "
            f"{recall:>14} {row['workload_saved']:>7.1%}"
        )
    print(
        f"Test reviews together: {totals['found']} of {totals['inclusions']} inclusions kept "
        f"({test_total['recall']:.1%}), {test_total['workload_saved']:.1%} of records not read."
    )
    if args.backend == "mock" and not args.from_case_study:
        print("Note: the mock backend is a keyword stand-in; these numbers say nothing about LLMs.")
    print(f"Results: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
