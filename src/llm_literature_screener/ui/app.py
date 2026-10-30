"""Streamlit web app: upload records, give the criteria, choose a backend, run and download.

Start it with ``lls ui`` (or ``streamlit run`` on this file). Uploaded files are written to a
temporary folder for the session and read with the same code as the command line.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from llm_literature_screener import __version__
from llm_literature_screener.backends import (
    BACKENDS,
    DEFAULT_BASE_URL,
    DEFAULT_HF_MODEL,
    BackendError,
    Scorer,
    create_scorer,
)
from llm_literature_screener.calibrate import (
    DEFAULT_TARGET_RECALL,
    DEFAULT_UPPER_THRESHOLD,
    CalibrationError,
)
from llm_literature_screener.criteria import Criteria, CriteriaError
from llm_literature_screener.io import (
    FORMATS,
    ColumnMappingError,
    UnknownFormatError,
    apply_labels,
    read_labels,
)
from llm_literature_screener.io.labels import LabelFileError
from llm_literature_screener.pipeline import (
    ScreeningOptions,
    combine_sources,
    load_source,
    run_screening,
    write_outputs,
)

UPLOAD_TYPES = ["ris", "nbib", "txt", "csv", "tsv", "xlsx"]
EXPECTED_ERRORS = (
    BackendError,
    CalibrationError,
    ColumnMappingError,
    CriteriaError,
    LabelFileError,
    UnknownFormatError,
    ValueError,
)
BACKEND_LABELS = {
    "hf": "hf: local model (transformers)",
    "openai": "openai: OpenAI-compatible server",
    "mock": "mock: keyword demo, no model",
}
"""Short menu labels for the backends, which fit the sidebar."""
PROTOCOL_TEMPLATE = """question: <the review question>
population: <who the review is about>
intervention: <intervention or exposure>
comparator: <comparison>
outcomes:
  - <outcome>
study_designs:
  - Randomized controlled trials
exclude:
  - <exclusion criterion>
"""


def example_dir() -> Path | None:
    """Return the folder with the bundled synthetic example, if it is installed."""
    package = Path(__file__).resolve().parents[1]
    for candidate in (package / "examples", package.parents[1] / "examples"):
        if (candidate / "records.ris").is_file():
            return candidate
    return None


def session_dir() -> Path:
    """Return a temporary folder that lives as long as the browser session."""
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="lls-ui-")
    return Path(st.session_state.workdir)


def save_upload(upload: Any, folder: Path) -> Path:
    """Write an uploaded file to ``folder`` under its own (sanitized) name."""
    path = folder / Path(upload.name).name
    path.write_bytes(upload.getvalue())
    return path


def parse_columns(text: str) -> dict[str, str]:
    """Parse one ``field=Column header`` mapping per line."""
    columns: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip():
            field_name, separator, header = line.partition("=")
            if not separator:
                raise ColumnMappingError(f"{line!r} is not FIELD=COLUMN.")
            columns[field_name.strip().lower()] = header.strip()
    return columns


@st.cache_resource(show_spinner="Loading the model ...")
def load_scorer(backend: str, model: str, settings: tuple[tuple[str, Any], ...]) -> Scorer:
    """Create a scorer once per configuration, so the model is not reloaded on every run."""
    return create_scorer(backend, model or None, **dict(settings))


def describe_backend(name: str) -> str:
    """Label a backend in the backend menu."""
    return BACKEND_LABELS.get(name, name)


def backend_settings() -> tuple[str, str, tuple[tuple[str, Any], ...]]:
    """Draw the backend controls in the sidebar and return the chosen configuration."""
    st.sidebar.header("Model backend")
    names: list[str] = list(BACKENDS)
    backend = str(
        st.sidebar.selectbox(
            "Backend", names, index=names.index("mock"), format_func=describe_backend
        )
    )
    settings: dict[str, Any] = {}
    model = ""
    if backend == "hf":
        model = st.sidebar.text_input("Model", DEFAULT_HF_MODEL)
        settings["device"] = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda", "mps"])
        settings["dtype"] = st.sidebar.selectbox(
            "Precision", ["auto", "bfloat16", "float16", "float32"]
        )
        settings["token_budget"] = int(
            st.sidebar.number_input("Padded tokens per batch", 512, 1_000_000, 24000, step=512)
        )
    elif backend == "openai":
        model = st.sidebar.text_input("Model name on the server", DEFAULT_HF_MODEL)
        settings["base_url"] = st.sidebar.text_input("Base URL", DEFAULT_BASE_URL)
        settings["api_key"] = st.sidebar.text_input("API key (if needed)", type="password") or None
        settings["concurrency"] = int(st.sidebar.number_input("Parallel requests", 1, 64, 8))
    else:
        st.sidebar.caption(
            "The mock backend scores by keyword overlap. Use it to try the app, not to screen."
        )
    return backend, model, tuple(sorted(settings.items()))


def decision_settings() -> ScreeningOptions:
    """Draw the threshold controls in the sidebar and return the screening options."""
    st.sidebar.header("Threshold and decisions")
    target = st.sidebar.slider(
        "Target recall on known inclusions", 0.80, 1.00, DEFAULT_TARGET_RECALL, step=0.01
    )
    fixed = st.sidebar.checkbox("Set the threshold myself")
    threshold = (
        float(st.sidebar.number_input("Exclusion threshold (score)", value=-5.0, step=0.5))
        if fixed
        else None
    )
    upper = float(
        st.sidebar.number_input(
            "Include threshold (score)", value=DEFAULT_UPPER_THRESHOLD, step=0.5
        )
    )
    reasons = st.sidebar.checkbox("Write one-line reasons", value=True)
    dedup = st.sidebar.checkbox("Remove duplicates", value=True)
    return ScreeningOptions(
        target_recall=float(target),
        threshold=threshold,
        upper_threshold=upper,
        deduplicate=dedup,
        reasons=reasons,
    )


def zip_folder(folder: Path) -> bytes:
    """Pack every file in ``folder`` into a zip archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    return buffer.getvalue()


def show_page(page: str, height: int) -> None:
    """Embed a full HTML page in the app.

    Only the generated report is embedded, and it escapes every text that comes from the
    records or the model, so no untrusted markup reaches the page.
    """
    iframe = getattr(st, "iframe", None)  # Newer Streamlit; older versions use components
    if iframe is not None:
        iframe(page, height=height)
    else:
        components.html(page, height=height, scrolling=True)


def show_results(outputs: dict[str, Any]) -> None:
    """Show the summary of the last run and the download buttons."""
    prisma = outputs["prisma"]
    calibration = outputs["calibration"]
    st.subheader("Results")
    first, second, third, fourth = st.columns(4)
    first.metric("Records screened", f"{prisma['assessed_by_tool']:,}")
    second.metric("Left for people to screen", f"{prisma['to_screen']:,}")
    third.metric("Excluded by the tool", f"{prisma['excluded_by_tool']:,}")
    recall = calibration.get("recall")
    fourth.metric(
        "Known inclusions kept",
        f"{calibration['found']} of {calibration['known_inclusions']}"
        if recall is not None
        else "unknown",
    )
    for warning in outputs["warnings"]:
        st.warning(warning)
    st.download_button(
        "Download all results (zip)",
        outputs["zip"],
        file_name="screening_results.zip",
        mime="application/zip",
        type="primary",
    )
    for label, (name, data, mime) in outputs["files"].items():
        st.download_button(label, data, file_name=name, mime=mime)
    with st.expander("Report preview", expanded=True):
        show_page(outputs["report"], height=900)


def main() -> None:
    """Draw the app."""
    st.set_page_config(page_title="LLM literature screener", layout="wide")
    st.title("LLM literature screener")
    st.caption(
        f"Version {__version__}. Pre-screen titles and abstracts with an open LLM, calibrated "
        "for high recall. The tool excludes only records below a threshold set on decisions "
        "your team has already made; people make every final decision."
    )
    backend, model, settings = backend_settings()
    options = decision_settings()

    examples = example_dir()
    use_example = examples is not None and st.checkbox(
        "Use the bundled synthetic example (26 invented records on exercise for back pain)"
    )

    st.subheader("1. Records")
    upload = st.file_uploader(
        "RIS, PubMed .nbib, CSV, TSV, Excel, or a Rayyan or Covidence export",
        type=UPLOAD_TYPES,
        disabled=use_example,
    )
    fmt_choice = st.selectbox("Format", ["detect automatically", *FORMATS], disabled=use_example)
    with st.expander("Column mapping for spreadsheets (optional)"):
        column_text = st.text_area(
            "One FIELD=Column header per line, for example title=Article Title",
            disabled=use_example,
        )

    st.subheader("2. Review protocol")
    default_protocol = PROTOCOL_TEMPLATE
    if use_example and examples is not None:
        default_protocol = (examples / "protocol.yaml").read_text(encoding="utf-8")
    protocol_upload = st.file_uploader("Upload a protocol YAML", type=["yaml", "yml"])
    protocol_text = st.text_area(
        "Or paste it here",
        protocol_upload.getvalue().decode("utf-8") if protocol_upload else default_protocol,
        height=260,
        key=f"protocol-{use_example}",
    )

    st.subheader("3. Decisions your team has already made (optional)")
    labels_upload = st.file_uploader(
        "Labels file (id, doi, pmid or title, and label)",
        type=["csv", "tsv", "xlsx"],
        disabled=use_example,
    )

    if st.button("Run screening", type="primary"):
        folder = session_dir()
        progress_bar = st.progress(0.0, text="Starting ...")
        stage_names = {"scoring": "Scoring records", "reasons": "Writing reasons"}

        def progress(stage: str, done: int, total: int) -> None:
            progress_bar.progress(
                done / total if total else 1.0, text=f"{stage_names[stage]}: {done} of {total}"
            )

        try:
            criteria = Criteria.from_yaml_text(protocol_text)
            labels_path: Path | None
            if use_example and examples is not None:
                records_path, labels_path = examples / "records.ris", examples / "labels.csv"
                fmt = None
                columns: dict[str, str] = {}
            else:
                if upload is None:
                    st.error("Upload a records file first.")
                    return
                records_path = save_upload(upload, folder)
                labels_path = save_upload(labels_upload, folder) if labels_upload else None
                fmt = None if fmt_choice == "detect automatically" else fmt_choice
                columns = parse_columns(column_text)
            source = load_source(records_path, fmt, columns or None)
            records, source_of = combine_sources([source])
            if labels_path is not None:
                apply_labels(records, read_labels(labels_path))
            if options.threshold is None and not any(r.label is True for r in records):
                raise CalibrationError(
                    "No record is labeled as a known inclusion. Upload decisions your team has "
                    "already made, or set the threshold yourself in the sidebar."
                )
            scorer = load_scorer(backend, model, settings)
            run = run_screening(
                records,
                criteria,
                scorer,
                options,
                progress=progress,
                sources=[(source.path.name, len(source.records))],
                source_of=source_of,
            )
            out_dir = folder / "results"
            for old in out_dir.glob("*") if out_dir.is_dir() else []:
                old.unlink()
            paths = write_outputs(run, out_dir, [source])
        except EXPECTED_ERRORS as error:
            progress_bar.empty()
            st.error(str(error))
            return
        progress_bar.progress(1.0, text="Done")
        files = {
            "Ranked results (CSV)": (
                paths["results"].name,
                paths["results"].read_bytes(),
                "text/csv",
            ),
            "Report (HTML)": (paths["report"].name, paths["report"].read_bytes(), "text/html"),
            "PRISMA counts and methods (Markdown)": (
                paths["prisma"].name,
                paths["prisma"].read_bytes(),
                "text/markdown",
            ),
        }
        screened = paths[f"records:{source.path.name}"]
        files[f"Records with decisions ({screened.suffix})"] = (
            screened.name,
            screened.read_bytes(),
            "application/octet-stream",
        )
        st.session_state.outputs = {
            "prisma": run.prisma.to_dict(),
            "calibration": run.calibration.to_dict(),
            "warnings": [*run.calibration.warnings, *run.warnings],
            "files": files,
            "zip": zip_folder(out_dir),
            "report": paths["report"].read_text(encoding="utf-8"),
        }

    if "outputs" in st.session_state:
        show_results(st.session_state.outputs)


if __name__ == "__main__":
    main()
