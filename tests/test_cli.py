"""Smoke tests for the ``lls`` command line with the mock backend."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_literature_screener import __version__
from llm_literature_screener.backends import BackendError
from llm_literature_screener.cli import app
from tests.conftest import EXAMPLES

runner = CliRunner()
RECORDS = str(EXAMPLES / "records.ris")
CRITERIA = str(EXAMPLES / "protocol.yaml")
LABELS = str(EXAMPLES / "labels.csv")


def text(output: str) -> str:
    """Join the lines that the terminal renderer wrapped."""
    return " ".join(output.split())


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("results")
    outcome = runner.invoke(
        app,
        [
            "screen",
            "--records",
            RECORDS,
            "--criteria",
            CRITERIA,
            "--labels",
            LABELS,
            "--backend",
            "mock",
            "--reasons",
            "--out",
            str(out),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    return out


def test_help_lists_commands_and_every_command_has_examples() -> None:
    outcome = runner.invoke(app, ["--help"])
    assert outcome.exit_code == 0
    for command in ("screen", "calibrate", "evaluate", "report", "ui"):
        assert command in outcome.output
        help_text = runner.invoke(app, [command, "--help"])
        assert help_text.exit_code == 0, help_text.output
        assert "Example" in help_text.output
        assert "lls " + command in help_text.output


def test_version() -> None:
    outcome = runner.invoke(app, ["--version"])
    assert outcome.exit_code == 0
    assert __version__ in outcome.output


def test_screen_writes_results_prisma_and_report(results: Path) -> None:
    for name in ("screening_results.csv", "records_screened.ris", "prisma.md", "report.html"):
        assert (results / name).is_file()
    prisma = (results / "prisma.md").read_text()
    assert "| Records identified | 26 |" in prisma
    assert "| Duplicate records removed | 1 |" in prisma
    assert "Records marked as ineligible by automation tools" in prisma
    assert "<svg" in (results / "report.html").read_text()


def test_screen_needs_labels_or_a_threshold(tmp_path: Path) -> None:
    outcome = runner.invoke(
        app,
        [
            "screen",
            "-r",
            RECORDS,
            "-c",
            CRITERIA,
            "--backend",
            "mock",
            "--out",
            str(tmp_path / "x"),
        ],
    )
    assert outcome.exit_code == 1
    assert "known inclusion" in text(outcome.output)
    assert not (tmp_path / "x").exists()
    fixed = runner.invoke(
        app,
        [
            "screen",
            "-r",
            RECORDS,
            "-c",
            CRITERIA,
            "--backend",
            "mock",
            "--threshold",
            "0",
            "--out",
            str(tmp_path / "y"),
        ],
    )
    assert fixed.exit_code == 0, fixed.output


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--column", "title"], "FIELD=COLUMN"),
        (["--backend", "gpt"], "choose one of"),
        (["--target-recall", "1.5"], "at most 1"),
    ],
)
def test_screen_rejects_bad_options(tmp_path: Path, arguments: list[str], message: str) -> None:
    outcome = runner.invoke(
        app,
        ["screen", "-r", RECORDS, "-c", CRITERIA, "-l", LABELS, "--out", str(tmp_path), *arguments],
    )
    assert outcome.exit_code == 2
    assert message in text(outcome.output)


def test_screen_reports_bad_protocols_without_a_traceback(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("population: adults\n")
    outcome = runner.invoke(
        app, ["screen", "-r", RECORDS, "-c", str(protocol), "-l", LABELS, "--backend", "mock"]
    )
    assert outcome.exit_code == 1
    assert "Error:" in outcome.output and "question" in outcome.output
    assert "Traceback" not in outcome.output


def test_calibrate_prints_the_threshold_and_the_ladder(results: Path, tmp_path: Path) -> None:
    out = tmp_path / "calibration.json"
    outcome = runner.invoke(app, ["calibrate", str(results), "--out", str(out)])
    assert outcome.exit_code == 0, outcome.output
    output = text(outcome.output)
    assert "Threshold for 95% recall" in output
    assert "7 of 7 known inclusions" in output
    assert "Other recall targets" in output
    data = json.loads(out.read_text())
    assert data["known_inclusions"] == 7
    assert [row["target_recall"] for row in data["ladder"]] == [0.8, 0.9, 0.95, 0.98, 1.0]


def test_evaluate_saved_scores(results: Path, tmp_path: Path) -> None:
    outcome = runner.invoke(
        app,
        [
            "evaluate",
            "--scores",
            str(results),
            "--labels",
            LABELS,
            "--drop-unlabeled",
            "--threshold",
            "-1",
            "--out",
            str(tmp_path),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    assert "ROC AUC" in outcome.output and "WSS@95" in outcome.output
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["records"] == 15
    assert metrics["threshold"] == -1.0
    assert metrics["scorer"]["backend"] == "mock"
    assert "Evaluation against complete labels" in (tmp_path / "report.html").read_text()


def test_evaluate_scores_records_with_a_dev_threshold(results: Path, tmp_path: Path) -> None:
    outcome = runner.invoke(
        app,
        [
            "evaluate",
            "--records",
            RECORDS,
            "--criteria",
            CRITERIA,
            "--labels",
            LABELS,
            "--backend",
            "mock",
            "--drop-unlabeled",
            "--dev-scores",
            str(results),
            "--out",
            str(tmp_path),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    assert "calibrated on" in text(outcome.output)
    assert (tmp_path / "screening_results.csv").is_file()


def test_evaluate_needs_complete_labels() -> None:
    outcome = runner.invoke(
        app, ["evaluate", "--records", RECORDS, "--criteria", CRITERIA, "--backend", "mock"]
    )
    assert outcome.exit_code == 1
    assert "fully labeled" in text(outcome.output)
    neither = runner.invoke(app, ["evaluate"])
    assert neither.exit_code == 2


def test_report_rebuilds_and_recalibrates(results: Path, tmp_path: Path) -> None:
    outcome = runner.invoke(app, ["report", str(results)])
    assert outcome.exit_code == 0, outcome.output
    assert (results / "report.html").is_file()
    custom = tmp_path / "strict.html"
    outcome = runner.invoke(
        app, ["report", str(results), "--threshold", "1", "--title", "Strict", "--out", str(custom)]
    )
    assert outcome.exit_code == 0, outcome.output
    page = custom.read_text()
    assert "Strict" in page
    assert "set by the review team at a score of 1.00" in page


def test_ui_starts_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    calls: list[list[str]] = []

    def fake_call(command: list[str], env: object = None) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr("llm_literature_screener.cli.subprocess.call", fake_call)
    outcome = runner.invoke(app, ["ui", "--port", "8600", "--headless"])
    assert outcome.exit_code == 0, outcome.output
    (command,) = calls
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert Path(command[4]).name == "app.py" and Path(command[4]).is_file()
    assert "8600" in command


def test_error_messages_keep_square_brackets(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_backend(*args: object, **kwargs: object) -> None:
        raise BackendError('Install it with: pip install "llm-literature-screener[hf]"')

    monkeypatch.setattr("llm_literature_screener.cli.create_scorer", broken_backend)
    outcome = runner.invoke(app, ["screen", "-r", RECORDS, "-c", CRITERIA, "-l", LABELS])
    assert outcome.exit_code == 1
    assert 'llm-literature-screener[hf]"' in text(outcome.output)
