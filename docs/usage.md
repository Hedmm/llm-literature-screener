# Usage guide

This guide walks through a screening project from installation to the report. For the
details of each input format see [file formats](file-formats.md); for how the threshold is
set see [calibration](calibration.md); and before relying on the results, read the
[limitations](limitations.md).

## 1. Install

The package needs Python 3.10 or later.

```bash
git clone https://github.com/Hedmm/llm-literature-screener.git
cd llm-literature-screener
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # command line and the mock backend
pip install -e ".[hf]"             # + PyTorch and transformers, to run models locally
pip install -e ".[ui]"             # + Streamlit, for the web app
pip install -e ".[dev]"            # + tests, ruff and mypy (includes the web app)
```

`pip install -e ".[hf]"` installs the default PyTorch build. For a GPU, install the PyTorch
build that matches your CUDA version first, following the instructions on pytorch.org.

## 2. Prepare three inputs

**Records.** Export the search results from your databases or reference manager: RIS, PubMed
(`.nbib`), CSV, TSV, Excel, or a Rayyan or Covidence CSV export. Several files can be
screened together; duplicates across them are removed.

**Protocol.** A YAML file with the review question and the eligibility criteria. Only
`question` is required:

```yaml
title: Exercise therapy for chronic low back pain
question: Does supervised exercise therapy reduce pain and disability in adults with chronic low back pain?
population: Adults with non-specific low back pain lasting 12 weeks or longer.
intervention: Supervised exercise therapy.
comparator: Usual care, no treatment, a waiting list or another conservative treatment.
outcomes: [Pain intensity, Back-specific disability]
study_designs: [Randomized controlled trials]
include:
  - Trials that report at least one of the outcomes after treatment
exclude:
  - Children or adolescents
  - Acute low back pain (less than 12 weeks)
languages: [English]
years: 2000-2024
```

The fields are rendered into the prompt in a fixed order. `exposure`, `comparison`,
`inclusion_criteria` and `exclusion_criteria` are accepted as alternative names, and a
misspelled field name is reported as an error rather than silently ignored.

**Known decisions (labels).** The threshold is calibrated on records your team has already
screened. Screen a random sample of the records by hand and save the decisions as a CSV with
an id and a label column. Inclusions are usually rare, so the sample has to be large enough
to contain at least 10 of them; a random sample also keeps the recall estimate honest
(see [calibration](calibration.md)).

```csv
id,label
EX01,include
EX07,exclude
```

The id is the record id the tool uses (see [file formats](file-formats.md)); a `doi`,
`pmid` or `title` column works too. The more known inclusions, the more reliable the
threshold: the tool warns when there are fewer than 10.

## 3. Screen

```bash
lls screen --records refs.ris --criteria protocol.yaml --labels seed.csv \
  --backend hf --model Qwen/Qwen2.5-7B-Instruct --target-recall 0.95 --out results/
```

The run removes duplicates, scores every record, sets the threshold at the highest score
that keeps 95% of the known inclusions, and marks each record:

| Decision | Meaning |
|---|---|
| include | The score is at or above the include threshold (by default 0, where P(include) = 0.5). |
| uncertain | Above the calibrated threshold, but the model leans towards "No". People still screen it. |
| exclude | Below the calibrated threshold: marked as ineligible by the tool. |

Useful options (see `lls screen --help` for all of them):

| Option | Effect |
|---|---|
| `--records` (repeatable) | Combine several exports; record ids get the file name as prefix. |
| `--threshold -8` | Use a fixed exclusion threshold instead of calibrating. |
| `--upper-threshold 2` | Move the boundary between include and uncertain. |
| `--reasons` | Write a one-line reason for every include and uncertain record. |
| `--no-dedup` | Keep duplicates. |
| `--column title='Article Title'` | Map spreadsheet columns to fields. |
| `--format medline` | Set the format when it cannot be detected. |
| `--max-abstract-words 350` | Cut abstracts at this many words (the validated setting). |

### Outputs

| File | Content |
|---|---|
| `screening_results.csv` | Every record ranked by score: id, title, authors, year, journal, doi, pmid, human label, score, p_include, decision, reason. |
| `<input>_screened.<ext>` | Each input file in its own format with score, p_include, decision and reason added. |
| `prisma.md` | PRISMA 2020 counts for identification and screening, and a methods paragraph. |
| `report.html` | Self-contained report: summary, decisions, recall curve, calibration, PRISMA, uncertain records. |
| `calibration.json` | Thresholds and the recall reached on the known inclusions. |
| `duplicates.csv` | Each removed duplicate, the record it matched and why. |
| `criteria.yaml` | The protocol as used, for the record. |
| `run.json` | Settings, model, counts and warnings of the run. |

## 4. Choose a backend

**`hf`: a local model with Hugging Face transformers.** This is the validated method. It
uses bf16 on a GPU and float32 on a CPU. In the case study, Qwen2.5-7B-Instruct peaked at
17.4 GB of GPU memory with the default budget of 24,000 padded tokens per batch; lower
`--token-budget` to use less. On a CPU only very small models are practical.

```bash
lls screen ... --backend hf --model Qwen/Qwen2.5-7B-Instruct --device cuda --token-budget 24000
```

**`openai`: any OpenAI-compatible server.** The scorer requests one token with
`logprobs`, so the score is the same logit difference. Start a server, then point the tool
at it:

```bash
# vLLM
vllm serve Qwen/Qwen2.5-7B-Instruct --max-logprobs 20
lls screen ... --backend openai --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1

# llama.cpp server
llama-server -m qwen2.5-7b-instruct-q8_0.gguf --port 8080
lls screen ... --backend openai --model qwen2.5-7b-instruct --base-url http://localhost:8080/v1

# Ollama
ollama serve
lls screen ... --backend openai --model qwen2.5:7b-instruct --base-url http://localhost:11434/v1
```

If the server returns no log-probabilities, the tool falls back to the generated Yes or No
and warns that records with the same answer are tied, which makes ranking and calibration
coarse. Set `LLS_API_KEY` (or `--api-key`) if the server needs a key, and `--concurrency`
to control parallel requests.

**`mock`: no model.** Scores by keyword overlap with the protocol. It exists for tests and
demos and must not be used to screen a real review.

## 5. Recalibrate without running the model again

Scores are saved, so the threshold can change without new model runs. After the team labels
more records:

```bash
lls calibrate results/ --labels more_labels.csv --target-recall 0.95
lls report results/ --labels more_labels.csv --target-recall 0.95 --out results/report_v2.html
```

`lls calibrate` prints the threshold, the recall it reaches with a 95% confidence interval,
and what other targets would cost.

## 6. Evaluate on a fully labeled review

To measure how well a model and protocol work before using them, run them on a review whose
decisions are all known, with a threshold set on a different review:

```bash
lls screen --records dev_review.csv --criteria dev.yaml --labels dev_labels.csv --out dev/
lls evaluate --records test_review.csv --criteria test.yaml --dev-scores dev/ --out evaluation/
```

`lls evaluate` reports ROC AUC, WSS@95, recall with a Wilson interval, workload saved,
precision and Cohen's kappa. Without `--dev-scores` or `--threshold`, the threshold is set
on the evaluated records themselves and the recall it shows is optimistic.

## 7. The web app

```bash
pip install -e ".[ui]"
lls ui
```

The app opens in the browser. Upload the records, paste or upload the protocol, optionally
upload known decisions, choose a backend and its settings in the sidebar, and run. A
progress bar follows the scoring; the results, the report and a zip of every output can be
downloaded at the end. A checkbox loads the bundled synthetic example.

## 8. Docker

The CPU image contains the command line, the web app and the transformers backend with the
CPU build of PyTorch:

```bash
docker build -t llm-literature-screener .
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" llm-literature-screener \
  screen --records refs.ris --criteria protocol.yaml --labels seed.csv \
  --backend openai --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://host.docker.internal:8000/v1 --out results/
docker run --rm -p 8501:8501 llm-literature-screener ui --headless
```

## 9. Python API

```python
from pathlib import Path

from llm_literature_screener.backends import create_scorer
from llm_literature_screener.criteria import Criteria
from llm_literature_screener.io import apply_labels, read_labels, read_records
from llm_literature_screener.pipeline import ScreeningOptions, run_screening, write_outputs

records = read_records(Path("refs.ris"))
apply_labels(records, read_labels(Path("seed.csv")))
criteria = Criteria.from_yaml(Path("protocol.yaml"))
scorer = create_scorer("hf", "Qwen/Qwen2.5-7B-Instruct")

run = run_screening(records, criteria, scorer, ScreeningOptions(target_recall=0.95))
print(run.calibration.threshold, run.prisma.to_screen)
write_outputs(run, Path("results"))
```
