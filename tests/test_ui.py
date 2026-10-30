"""Smoke tests for the Streamlit app, run headless with Streamlit's AppTest."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from llm_literature_screener.io import ColumnMappingError
from llm_literature_screener.ui import app as ui
from tests.conftest import EXAMPLES

APP = str(Path(ui.__file__))


def test_app_starts_without_errors() -> None:
    page = AppTest.from_file(APP, default_timeout=60).run()
    assert not page.exception
    assert page.title[0].value == "LLM literature screener"
    assert page.button[0].label == "Run screening"


def test_example_run_shows_results() -> None:
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.checkbox[0].check().run()
    page.button[0].click().run()
    assert not page.exception
    assert not page.error
    metrics = {metric.label: metric.value for metric in page.metric}
    assert metrics["Records screened"] == "25"
    assert metrics["Known inclusions kept"] == "7 of 7"
    assert any("known inclusions" in warning.value for warning in page.warning)


def test_example_run_with_a_fixed_threshold() -> None:
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.checkbox[0].check().run()
    page.sidebar.checkbox[0].check().run()
    page.sidebar.number_input[0].set_value(0.0).run()
    page.button[0].click().run()
    assert not page.exception
    assert not page.error
    assert {metric.label for metric in page.metric} >= {"Excluded by the tool"}


def test_running_without_records_asks_for_a_file() -> None:
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.button[0].click().run()
    assert page.error[0].value == "Upload a records file first."


def test_helpers(tmp_path: Path) -> None:
    assert ui.example_dir() == EXAMPLES
    assert ui.parse_columns("title = Article Title\n\nabstract=Summary") == {
        "title": "Article Title",
        "abstract": "Summary",
    }
    with pytest.raises(ColumnMappingError):
        ui.parse_columns("title")
    (tmp_path / "a.txt").write_text("a")
    assert ui.zip_folder(tmp_path).startswith(b"PK")
