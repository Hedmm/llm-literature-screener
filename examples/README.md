# Example data (synthetic)

Every record in this folder is **synthetic**. The records were written for this example and
do not describe real studies: the authors, journals and results are invented, and the DOIs
use the `10.5555` test prefix. Each RIS record carries the note
`SYNTHETIC RECORD: written for the llm-literature-screener example.`

| File | What it is |
|---|---|
| `records.ris` | 26 records for a review of exercise therapy for chronic low back pain. Record EX23 is a copy of EX01 from "another database", so duplicate removal has something to find. EX17 has no abstract. |
| `protocol.yaml` | The review question and eligibility criteria. |
| `labels.csv` | Decisions the review team has "already made" for 15 records (7 inclusions, 8 exclusions), used to calibrate the threshold. |

Run the example with the mock backend, which needs no model:

```bash
lls screen --records examples/records.ris --criteria examples/protocol.yaml \
  --labels examples/labels.csv --backend mock --reasons --out results/
```

The mock backend scores records by keyword overlap with the protocol. Its numbers show how
the tool works; they say nothing about how well a language model screens. With only 7 known
inclusions, the run also shows the warning the tool gives when calibration rests on fewer
than 10 inclusions.
