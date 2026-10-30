"""Offline tests for benchmarks/synergy.py, using small synthetic files in the SYNERGY layout."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from llm_literature_screener.criteria import Criteria
from tests.conftest import REPO_ROOT


def load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "synergy", REPO_ROOT / "benchmarks" / "synergy.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["synergy"] = module
    spec.loader.exec_module(module)
    return module


synergy = load_benchmark()

# The criteria of the case study's screen.py, copied verbatim.
CASE_STUDY_REVIEWS = {
    "Appenzeller-Herzog_2020": {
        "question": (
            "Comparative effectiveness of common therapies for Wilson disease (controlled studies)."
        ),
        "include": (
            "Studies in patients with Wilson disease that compare a treatment "
            "(D-penicillamine, trientine, zinc salts, tetrathiomolybdate, liver "
            "transplantation or other) with another treatment, placebo or no treatment, i.e. "
            "studies with a control or comparison group, reporting clinical outcomes."
        ),
        "exclude": (
            "Single-arm case series and case reports, reviews, editorials, animal or in-vitro"
            " studies, and studies of diagnosis, genetics or pathophysiology without a "
            "treatment comparison."
        ),
    },
    "Wolters_2018": {
        "question": ("Coronary heart disease, heart failure, and the risk of dementia."),
        "include": (
            "Longitudinal studies (prospective or retrospective cohort, nested case-control) "
            "in community or population-based samples of adults, in which coronary heart "
            "disease (myocardial infarction, angina, coronary artery disease) or heart "
            "failure is the exposure and incident dementia (all-cause, Alzheimer's disease or"
            " vascular dementia) is the outcome."
        ),
        "exclude": (
            "Cross-sectional studies, studies with only cognitive test scores or cognitive "
            "decline and no dementia diagnosis, reviews, editorials, case reports, and animal"
            " studies."
        ),
    },
    "van_Dis_2020": {
        "question": (
            "Long-term outcomes of cognitive behavioral therapy for anxiety-related disorders."
        ),
        "include": (
            "Randomized controlled trials in adults with an anxiety-related disorder "
            "(generalized anxiety disorder, panic disorder, social anxiety disorder, specific"
            " phobia, post-traumatic stress disorder or obsessive-compulsive disorder) "
            "comparing cognitive behavioral therapy, including exposure therapy, with a "
            "control condition (waiting list, usual care, placebo or psychological placebo)."
        ),
        "exclude": (
            "Non-randomized studies, trials without a control condition, trials only in "
            "children or adolescents, reviews, meta-analyses and study protocols."
        ),
    },
}


@pytest.mark.parametrize("name", list(CASE_STUDY_REVIEWS))
def test_protocols_render_the_case_study_criteria(name: str) -> None:
    review = CASE_STUDY_REVIEWS[name]
    rendered = Criteria.from_yaml(REPO_ROOT / "benchmarks" / "protocols" / f"{name}.yaml").render()
    assert rendered == (
        f"Review question: {review['question']}\nInclude: {review['include']}\n"
        f"Exclude: {review['exclude']}"
    )


def write_review(path: Path, inclusions: int, others: int, with_abstract_labels: bool) -> None:
    header = ["record_id", "title", "abstract", "authors", "year", "doi", "label_included"]
    if with_abstract_labels:
        header.append("label_abstract_screening")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for number in range(1, inclusions + others + 1):
            included = number <= inclusions
            row = [
                str(number),
                f"Synthetic {'trial of therapy' if included else 'survey'} {number}",
                "Randomized controlled trial in adults." if included else "A survey.",
                "Example, Ann",
                "2020",
                "",
                "1" if included else "0",
            ]
            if with_abstract_labels:
                row.append("1" if included or number % 3 == 0 else "0")
            writer.writerow(row)


@pytest.fixture
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Replace the downloads with synthetic files and matching dataset descriptions."""
    data = tmp_path / "data"
    data.mkdir()
    datasets = {}
    for name, inclusions, others, abstract_labels in (
        ("Appenzeller-Herzog_2020", 4, 16, True),
        ("Wolters_2018", 3, 12, False),
        ("van_Dis_2020", 5, 20, True),
    ):
        write_review(data / f"{name}.csv", inclusions, others, abstract_labels)
        with (data / f"{name}.preds.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["record_id", "score"])
            for number in range(1, inclusions + others + 1):
                writer.writerow([number, 5.0 - number])
        datasets[name] = synergy.Dataset(name, "unused", inclusions + others, inclusions, "unused")

    def fake_download(url: str, path: Path, expected_sha256: str) -> Path:
        name = Path(url).stem
        return data / (f"{name}.preds.csv" if "ml-case-studies" in url else f"{name}.csv")

    monkeypatch.setattr(synergy, "DATASETS", datasets)
    monkeypatch.setattr(synergy, "download", fake_download)
    return tmp_path


def test_benchmark_runs_with_the_mock_backend(offline: Path) -> None:
    out = offline / "mock"
    assert (
        synergy.main(["--backend", "mock", "--data-dir", str(offline / "data"), "--out", str(out)])
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert [row["review"] for row in metrics["reviews"]] == [
        "Appenzeller-Herzog_2020",
        "Wolters_2018",
        "van_Dis_2020",
    ]
    assert [row["kappa_reference"] for row in metrics["reviews"]] == ["human", "labels", "human"]
    assert metrics["reviews"][0]["recall"] >= 0.95
    for name in ("Wolters_2018", "van_Dis_2020"):
        assert (out / name / "report.html").is_file()
        assert (out / name / "screening_results.csv").is_file()
    assert json.loads((out / "run_info.json").read_text())["scorer"]["backend"] == "mock"


def test_benchmark_evaluates_published_scores(offline: Path) -> None:
    out = offline / "published"
    assert (
        synergy.main(["--from-case-study", "--data-dir", str(offline / "data"), "--out", str(out)])
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    # Scores fall with the record number and inclusions come first, so ranking is perfect.
    assert metrics["threshold_from_dev"] == 1.0
    assert all(row["auc"] == 1.0 for row in metrics["reviews"])
    # The last van Dis inclusion scores 0, below the threshold, so it is reported as missed.
    assert (metrics["test_total"]["found"], metrics["test_total"]["inclusions"]) == (7, 8)
    with (out / "missed.csv").open(encoding="utf-8") as handle:
        missed = list(csv.DictReader(handle))
    assert [(row["review"], row["record_id"]) for row in missed] == [("van_Dis_2020", "5")]


def test_wrong_counts_are_rejected(offline: Path) -> None:
    dataset = synergy.Dataset("Wolters_2018", "unused", 99, 3, "unused")
    with pytest.raises(RuntimeError, match="expected 99 records"):
        synergy.load_review(offline / "data" / "Wolters_2018.csv", dataset)


def test_stratified_head_keeps_every_inclusion(offline: Path) -> None:
    records = synergy.load_review(offline / "data" / "Appenzeller-Herzog_2020.csv")
    kept = synergy.stratified_head(records, 6)
    assert len(kept) == 6
    assert sum(1 for record in kept if record.label) == 4
    assert synergy.stratified_head(records, 0) == records


def test_download_checks_the_checksum(tmp_path: Path) -> None:
    path = tmp_path / "file.csv"
    path.write_text("content")
    assert synergy.download("https://example.invalid/file.csv", path, synergy.sha256(path)) == path
