# USASpending FY22-FY25 federal assistance to 501(c)(3) pipeline

A Python implementation of `Methodology_FederalAssistance_501c3_FY22-FY25.docx`.
Pulls federal financial-assistance prime award transactions and summaries
from USASpending.gov for FY22-FY25, joins to IRS BMF for 501(c)(3)
verification, classifies into Core / Educational / Hospital / International
panels under a configurable priority hierarchy, and produces the 14
exhibits described in the methodology.

## Layout

```
.
├── pipeline.py              # orchestrator CLI
├── requirements.txt
├── reference_lists/         # version-controlled YAML rule lists
├── raw/
│   ├── usaspending/         # bulk download zips
│   ├── irs_bmf/             # eo1..eo4 regional CSVs
│   ├── sam/                 # SAM entity extract (operator-supplied)
│   └── reference/           # optional EIN lists (AHA, HRSA, IPEDS)
├── interim/                 # parsed parquet (transactions, BMF, SAM)
├── processed/               # recipient_match, classified txn, analytic tables
├── exhibits/                # CSVs (1 per panel x exhibit) and qa/
├── manifests/               # per-run manifest JSON (Section 9)
└── src/                     # modules
    ├── config.py
    ├── refdata.py           # YAML loader + version hash
    ├── manifest.py          # run manifest writer
    ├── acquire/
    │   ├── usaspending.py   # Step 2
    │   ├── irs_bmf.py       # Step 3
    │   └── sam.py           # Step 3b
    ├── match/
    │   ├── normalize.py
    │   └── recipient_match.py   # Step 4 / Section 5
    ├── classify/categorize.py   # Step 5 / Section 4
    ├── analytic/tables.py       # Step 6
    ├── aggregate/exhibits.py    # Step 7
    └── qa/checks.py             # Step 8
```

## Install

```
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Run

End-to-end (network required for the acquire/bmf steps):

```
python pipeline.py --all
```

Step by step:

```
python pipeline.py --init                  # Step 1
python pipeline.py --acquire               # Step 2: USASpending bulk download
python pipeline.py --bmf                   # Step 3: IRS BMF
python pipeline.py --sam                   # Step 3b: SAM extract parse (operator places file under raw/sam first)
python pipeline.py --match                 # Step 4
python pipeline.py --classify              # Step 5
python pipeline.py --tables                # Step 6
python pipeline.py --exhibits              # Step 7
python pipeline.py --qa                    # Step 8
```

Override the classification hierarchy:

```
python pipeline.py --classify --priority hospital educational international core
```

Switch deflator:

```
python pipeline.py --tables --deflator GDP
```

## Operator-supplied inputs (optional but recommended)

The methodology references several proprietary or login-gated lists. The
pipeline degrades gracefully when these are absent; supply them to tighten
classification:

- `raw/sam/<extract>.csv` - SAM Entity Public extract (UEI -> EIN)
- `raw/reference/aha_eins.txt` - AHA Annual Survey hospital EINs (one per line)
- `raw/reference/hrsa_uds_eins.txt` - HRSA UDS health-center EINs
- `raw/reference/ipeds_eins.txt` - IPEDS UnitID-EIN crosswalk (EINs only)
- `raw/reference/nces_school_district_names.txt` - school-district names (NCES CCD)
- `reference_lists/manual_match_overrides.yaml` - Tier 4 manual UEI->EIN overrides
- `reference_lists/spending_explorer_ref.yaml` - reconciliation totals from
  the public USAspending Spending Explorer dashboard

## Reproducibility

Every run writes `manifests/run-<UTC>.json` with the contents enumerated in
Section 9 of the methodology: USASpending payload, BMF release date and
SHA-256 per regional file, SAM extract metadata, match-tier counts, the
SHA-256 of the rule YAMLs, and the SHA-256 of every exhibit produced.

## Tests

```
python -m pytest tests/
```

A small smoke test under `tests/test_normalize.py` verifies the name
normalizer and the abbreviation expansion.
