# Supported file formats

Every reader returns the same record fields: id, title, abstract, authors, year, journal,
DOI, PMID, all source fields, and the existing human decision when the file has one. Every
writer writes the same format back with four fields added: `score`, `p_include`, `decision`
and `reason`. They are prefixed with `lls_` so they never clash with your own fields, and a
second run replaces them instead of adding more.

The format is detected from the file extension and, for `.csv` and `.txt` files, from the
content. Pass `--format` (one of `ris`, `medline`, `csv`, `tsv`, `excel`, `rayyan`,
`covidence`) when detection is not possible.

| Format | Extensions | Record id | Human decision | Added output |
|---|---|---|---|---|
| RIS | `.ris`, `.txt` | `ID` tag, else position | ASReview notes in `N1` | `N1` notes |
| PubMed MEDLINE | `.nbib`, `.medline`, `.txt` | `PMID` | none | `GN` notes |
| CSV, TSV | `.csv`, `.tsv`, `.tab`, `.txt` | id column, else row number | label column | columns |
| Excel | `.xlsx`, `.xlsm` | id column, else row number | label column | columns |
| Rayyan export | `.csv` | `key` | `RAYYAN-INCLUSION` in `notes` | columns |
| Covidence export | `.csv` | `Covidence #` | none (use a labels file) | columns |

Files may be UTF-8 (with or without a byte order mark) or Windows-1252. Ids must be unique
within a file; a repeated id gets a suffix (`5`, `5-2`) and a warning.

## RIS

Exported by EndNote, Zotero, Mendeley, Rayyan, Covidence and most databases. A record starts
with `TY  -` and ends with `ER  -`; lines without a tag continue the previous field.

| Field | Tags, in order of preference |
|---|---|
| title | `TI`, `T1`, `CT`, `BT` |
| abstract | `AB`, `N2` |
| authors | every `AU` and `A1` |
| year | `PY`, `Y1`, `DA`, `Y2` |
| journal | `T2`, `JF`, `JO`, `JA`, `J2`, `J1` |
| DOI | `DO`, `DI`, then a DOI found in `UR`, `L3`, `L2`, `M3` or `N1` |
| PMID | `AN` when it is a number, or `PMID: 123` in any field |

A note `N1  - ASReview_relevant` or `ASReview_irrelevant` sets the human decision. The output
is written as notes; this is record EX17 of the bundled example, screened with the mock
backend:

```
N1  - lls_score: -0.3739
N1  - lls_p_include: 0.40759898128209027
N1  - lls_decision: uncertain
N1  - lls_reason: Mentions 4 protocol terms (back, chronic, exercise, pain).
```

## PubMed MEDLINE (.nbib)

The format PubMed exports with "Send to: Citation manager" or "Save: PubMed format". Each line
starts with a four-character tag and `- `; continuation lines start with spaces.

| Field | Tags |
|---|---|
| id, PMID | `PMID` |
| title | `TI` (or `BTI`, `TT`) |
| abstract | `AB` |
| authors | `FAU` (full names), else `AU` |
| year | `DP`, `DEP`, `EDAT` |
| journal | `JT`, else `TA` |
| DOI | `LID` or `AID` ending in `[doi]`, else `doi:` in `SO` |

The output is written as general notes (`GN  - lls_decision: include`), and long lines are
wrapped at 82 characters as PubMed does.

## CSV, TSV and Excel

The first row holds the column names. Columns are matched to fields by name, ignoring case:

| Field | Column names tried |
|---|---|
| id | `id`, `record_id`, `record id`, `recordid`, `ref id`, `refid` |
| title | `title`, `article title`, `primary title`, `document title`, `ti` |
| abstract | `abstract`, `abstract note`, `ab` |
| authors | `authors`, `author`, `author(s)`, `author names`, `au` |
| year | `year`, `publication year`, `publication_year`, `published year`, `pubyear`, `py`, `date` |
| journal | `journal`, `journal title`, `source title`, `publication title`, `secondary title`, `source`, `so` |
| DOI | `doi`, `di` |
| PMID | `pmid`, `pubmed_id`, `pubmed id`, `pm` |
| label | `label`, `label_included`, `included`, `include`, `decision`, `screening decision`, `final decision` |

Map other names with `--column FIELD=COLUMN`, once per field:

```bash
lls screen --records refs.xlsx --column title='Article Title' --column abstract=Summary ...
```

The delimiter of a `.csv` file is detected from the header (comma, semicolon or tab), and an
Excel `sep=` first line is understood. Authors in one cell are split at semicolons, ` and `
or `|`. For Excel files the first worksheet is read unless `--sheet` names another. The old
`.xls` format is not supported; save it as `.xlsx` or CSV.

Label values `include`, `included`, `yes`, `y`, `1`, `true` mean include; `exclude`,
`excluded`, `no`, `n`, `0`, `false` mean exclude; anything else (for example `maybe`) means
the record has no known decision.

## Rayyan

Export from Rayyan as CSV. The columns are read like any CSV, with `key` as the id and
`pubmed_id` as the PMID. Rayyan stores the reviewers' decisions in the `notes` column, for
example `RAYYAN-INCLUSION: {"Alice"=>"Included", "Bob"=>"Excluded"}`. A record counts as a
known inclusion when at least one reviewer included it, so that no inclusion is lost, and as
a known exclusion when every reviewer who decided excluded it. "Maybe" leaves it unknown.

## Covidence

Export references from Covidence as CSV. `Covidence #` becomes the id, `Published Year` the
year, and `Accession Number` the PMID when it is a number. Covidence exports one list at a
time and has no decision column, so give the decisions in a labels file.

## Labels file

A CSV, TSV or Excel file with a label column (the names in the table above) and at least one
of `id` (also `record_id`, `key`, `Covidence #`), `doi`, `pmid` or `title`. Each row is
matched to a record by id, then DOI, then PMID, then normalized title, so decisions exported
from another tool still find their records. Rows with an unrecognized label are skipped with
a warning, and the number of labels that matched no record is reported.

With several records files, the record ids in the results are prefixed with the file name
(`pubmed.nbib:12345`); use these ids, or DOIs, PMIDs or titles, in the labels file.

## Results file

`screening_results.csv` has one row per screened record, ranked by score:

`rank, id, title, authors, year, journal, doi, pmid, label, score, p_include, decision, reason`

`lls calibrate`, `lls evaluate` and `lls report` read it, and they also accept any CSV with
`id` and `score` columns (and `label`, where needed), so scores from other tools can be
calibrated and reported the same way.
