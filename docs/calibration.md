# How calibration works

The model does not decide which records to exclude. It gives every record a score, and a
threshold set on decisions your team has already made turns the scores into decisions. This
page explains the score, the threshold rule, the uncertainty that comes with it, and the
metrics used to evaluate a model.

## The score

For each record the model reads the review question, the eligibility criteria, the title and
the abstract, and is asked: "Should this record go to full-text review? Answer Yes or No."
The tool does not let it write an answer. It reads the model's logits for the first answer
token and takes

```
score = logit("Yes") - logit("No")
P(include) = sigmoid(score) = 1 / (1 + exp(-score))
```

P(include) is the probability the model gives to answering Yes rather than No. It is not a
calibrated probability that the study belongs in the review: instruction-tuned models are
often overconfident, and many relevant records get a P(include) far below 0.5. What matters is
the ranking, which is why the tool keeps the unbounded score: two records with P(include) of
0.00001 and 0.0000001 still differ clearly in score (-11.5 and -16.1).

Through an OpenAI-compatible server the score is log P(Yes) - log P(No) from the returned
log-probabilities, which is the same quantity.

## The threshold rule

Let the labeled records contain *n* known inclusions. For a target recall *r* (default 0.95)
the tool keeps the smallest number of them that reaches the target, *k* = ceil(*r* × *n*),
and sets the threshold at the *k*-th highest score among the known inclusions. It is the
highest threshold that keeps at least the target share of the known inclusions:

- records scoring below the threshold are **excluded** (marked as ineligible by the tool);
- the other records go to people.

For example, with 20 known inclusions and a 95% target, the tool keeps 19 of them: the
threshold is the 19th highest inclusion score. Ties at the threshold are kept, so the
recall reached can exceed the target.

This is the rule of the validated case study, where the threshold was set on one review and
then applied unchanged to two others.

## The uncertainty: Wilson intervals

The recall on the known inclusions is an estimate of the recall on all inclusions, and with
few known inclusions it is uncertain. The tool reports a Wilson score interval, which stays
within 0 and 1 and behaves well near 100%, where most recall estimates lie:

```
p = k / n,  z = 1.96 for 95%
centre = (p + z²/(2n)) / (1 + z²/n)
half   = z × sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)
interval = centre ± half
```

How wide it is depends mainly on *n*. These values come from this package's
`wilson_interval`:

| Known inclusions | Kept for a 95% target | 95% interval for the recall |
|---|---|---|
| 5 | 5 (100%) | 56.6% to 100% |
| 10 | 10 (100%) | 72.2% to 100% |
| 20 | 19 (95.0%) | 76.4% to 99.1% |
| 50 | 48 (96.0%) | 86.5% to 98.9% |
| 100 | 95 (95.0%) | 88.8% to 97.8% |

With fewer than 10 known inclusions the threshold rests on a handful of records, so the tool
warns. The remedy is to label more records. The test suite checks the interval against
published values (Newcombe, *Statistics in Medicine* 1998).

## Where the known inclusions come from matters

The interval assumes that the known inclusions are like the inclusions the tool has not seen.
That holds when the labeled records are a random sample of the search results. If the
labels come from somewhere else (the first records screened, a few studies everyone knew,
inclusions of an earlier version of the review), they may be easier to find than the rest,
and the recall on all inclusions can be lower than the estimate.

## The uncertain band

Records at or above the threshold are split in two:

- **include**: the score is at or above an upper threshold, by default 0, where the model
  answers Yes more likely than No;
- **uncertain**: between the two thresholds. The model leans towards No, but the score is
  too high to exclude the record safely.

Both groups go to people. The split helps to plan the work: the uncertain records are where
human judgement is most needed. Change the boundary with `--upper-threshold`. If the
calibrated threshold is above the upper threshold, the band is empty.

## Choosing the target

A higher target keeps more inclusions and leaves more records to read. `lls calibrate`
shows the trade-off on your own labels:

```bash
lls calibrate results/ --labels seed.csv
```

It prints, for targets from 80% to 100%, the threshold, the recall with its interval, the
records left to screen and the workload saved.

## A fixed threshold

`--threshold` sets the exclusion threshold directly, for example a threshold calibrated on a
previous review with the same model and prompt. When labels exist the tool still reports the
recall the threshold reaches on them.

## Evaluation metrics

`lls evaluate` and the benchmark compare the scores with complete human decisions:

| Metric | Definition |
|---|---|
| Recall | Inclusions at or above the threshold / all inclusions, with a Wilson interval. |
| Precision | Inclusions at or above the threshold / records at or above it. |
| Workload saved | Records below the threshold / all records: the share nobody has to read. |
| WSS@95 | (N - n₉₅) / N - 0.05, where n₉₅ is the number of records read in score order until 95% of the inclusions are found. Reading in random order saves nothing. |
| ROC AUC | The chance that a random inclusion scores higher than a random exclusion; ties count one half. |
| Recall curve | The share of inclusions found against the share of records read in score order. |
| Cohen's kappa | Agreement beyond chance between the tool's flags and human decisions. |

Ties in score are common with bf16 models. For the records-read counts, WSS@95 and the recall
curve, the tool counts inclusions last among records with the same score, so these numbers
never look better than reading in score order would be.

Kappa is low by design. The tool flags far more records than people include, because the
threshold is set for recall, so the agreement it measures is not the agreement that matters.
Recall and workload saved are the numbers to look at.

## Validating out of sample

The honest way to know what a model and protocol do on new reviews is the case study's
design: set the threshold on one fully labeled review and measure recall and workload on
another, without changing anything. `lls evaluate --dev-scores` does exactly this, and
`benchmarks/synergy.py` repeats the case study.
