# Limitations

Read this before using the tool in a review.

## A pre-screen; people make the decisions

The tool ranks records and marks those below a calibrated threshold as ineligible. It does
not decide which studies belong in the review. Every record above the threshold, include or
uncertain, still has to be screened by people, and the full-text stage is untouched. Report
the tool's role in the methods section: `lls screen` writes a methods paragraph, and PRISMA
2020 has a box for records marked as ineligible by automation tools.

## Precision is low by design

The threshold is set for recall: it keeps nearly every known inclusion and accepts many
irrelevant records in exchange. In the
[case study](https://github.com/Hedmm/ml-case-studies/tree/main/llm-screening-systematic-reviews),
1 to 2% of the flagged records were final inclusions, and agreement with the human title
and abstract decisions was low (Cohen's kappa 0.06 to 0.21). This is the intended trade-off.
The saving is in the records nobody has to read, not in the accuracy of the flags.

## Recall is an estimate, and only as good as the labels

- **It is estimated on the known inclusions.** The tool reports the recall reached on the
  labeled records with a Wilson confidence interval. With few known inclusions the interval
  is wide, and the tool warns below 10.
- **The labels should be a random sample.** Inclusions found by other routes (the first
  records screened, studies everyone knew, an earlier version of the review) may be easier
  to find than the rest, and then the true recall can be lower than the estimate.
- **Some inclusions will be missed.** A 95% target means about one in twenty known
  inclusions may fall below the threshold, by design. The case study lists the five final
  inclusions its threshold missed so that you can see what kind of record slips through.
  Complementary searches, such as checking reference lists and citations, remain good
  practice.

## The method was validated on three reviews and one model

The case study used Qwen2.5-7B-Instruct on three medical reviews. Other models, prompts,
languages, topics and record sets may behave differently, and the threshold depends on all
of them. When anything changes, calibrate on labels from the review at hand, or validate on a
fully labeled review first (`lls evaluate`, `benchmarks/synergy.py`).

## The scores are not probabilities of eligibility

P(include) is the model's probability of answering "Yes" rather than "No". Models are often
overconfident, so many relevant records get a P(include) close to zero. Use the scores to
rank and the calibrated threshold to decide; do not read P(include) as the chance that a
study is eligible.

## Records with little text

Records without an abstract give the model only the title. The prompt asks the model to
answer Yes when the abstract is missing or too short, so these records tend to be kept, which
costs reading time rather than recall. Abstracts are cut at 350 words. Non-English titles and
abstracts are passed to the model as they are; how well it handles them depends on the
model.

## Servers without log-probabilities

Through an OpenAI-compatible server that returns no log-probabilities, the score is only the
generated answer (+1 for Yes, -1 for No). All records with the same answer tie, so the
ranking, the recall curve and the threshold become coarse. The tool warns when this happens.
Use the transformers backend, or a server that returns `logprobs` (for example vLLM), for
proper scores.

## The mock backend is not a model

The `mock` backend scores by keyword overlap with the protocol. It exists so that the
pipeline can be tested and demonstrated without a model. Its numbers, including those in
the example report, say nothing about how well a language model screens.

## Duplicate detection is conservative

Records are merged on an identical DOI, an identical PMID, or the same normalized title with
the same year; titles shorter than four words never match. Duplicates with typos in the title
or a different year are not found. The duplicate report lists every merge so it can be
checked.

## Data protection

With the transformers backend, records never leave your machine. With the openai backend,
titles and abstracts are sent to the server you configure; for a hosted service, check that
this is allowed for your data.

## No warranty

The software is provided under the MIT license, without warranty of any kind. It is a
research tool, not a medical device.
