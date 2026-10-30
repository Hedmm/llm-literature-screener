# SYNERGY benchmark

`synergy.py` runs the full pipeline on the three medical reviews of the
[validated case study](https://github.com/Hedmm/ml-case-studies/tree/main/llm-screening-systematic-reviews),
with the reviewers' own decisions as the reference:

| Role | Review | Topic |
|---|---|---|
| development | Appenzeller-Herzog_2020 | Treatments for Wilson disease |
| test | Wolters_2018 | Coronary heart disease, heart failure and dementia |
| test | van_Dis_2020 | Cognitive behavioral therapy for anxiety-related disorders |

The threshold is calibrated on the development review only (the highest score that keeps 95%
of its final inclusions) and applied unchanged to the two test reviews, so no number on the
test reviews is tuned on them. The review criteria in `protocols/` are those of the case study.

## Run it

On one GPU with the case study's model (needs `pip install -e ".[hf]"`):

```bash
python benchmarks/synergy.py
```

Through an OpenAI-compatible server, for example vLLM serving `Qwen/Qwen2.5-7B-Instruct`:

```bash
python benchmarks/synergy.py --backend openai --base-url http://localhost:8000/v1
```

Without a GPU, evaluate the scores published with the case study, to check this package's
evaluation code against the case study's own numbers:

```bash
python benchmarks/synergy.py --from-case-study
```

A quick check of the pipeline on a CPU. The mock backend counts keywords, so its numbers say
nothing about language models:

```bash
python benchmarks/synergy.py --backend mock --limit 300
```

## Data

The script downloads the three CSV files from the
[ASReview systematic-review-datasets](https://github.com/asreview/systematic-review-datasets)
repository (branch `metadata-v1-final`), the source of the SYNERGY collection the case study
used, into `benchmarks/data/`. It checks each file against a pinned SHA-256 checksum and
against the record and inclusion counts of the case study, and stops if either differs. The
datasets keep their own licenses; see their repositories.

## Outputs

Everything goes to `benchmarks/results/<run name>/`:

- `metrics.json`: per review, ROC AUC, WSS@95, records read to find 95% and 100% of the
  inclusions, and at the development threshold the recall with its Wilson 95% interval,
  workload saved, precision and Cohen's kappa. Kappa is computed against the human title and
  abstract decisions where the dataset has them (`kappa_reference: "human"`), otherwise
  against the final inclusions.
- `missed.csv`: final inclusions that scored below the threshold.
- `run_info.json`: model, backend and scoring time per review.
- `<review>/screening_results.csv` and `<review>/report.html`.

## Differences from the case study code

- Among records with exactly the same score, inclusions are counted last when the records
  read to reach a recall level are counted (and so for WSS@95). The case study sorted with an
  unstable sort, which orders ties arbitrarily. Scores from bf16 logits tie often, so WSS@95
  can differ from the case study in the third decimal; threshold-based numbers do not depend
  on ties.
- Positions in a padded batch are counted from each prompt's first real token, so a record's
  score does not depend on which other records share its batch. With rotary position
  embeddings, as in Qwen2.5, this changes nothing beyond floating-point noise.
