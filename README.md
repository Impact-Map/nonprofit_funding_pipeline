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

## Phase 1 lightweight mode

The pipeline supports two methodologies:

- **Full (default)** — IRS BMF cross-reference, 5-tier match, NTEE/IPEDS/AHA/HRSA-driven
  carve-outs. See `Methodology_FederalAssistance_501c3_FY22-FY25.docx`.
- **Phase 1 lightweight** — USAspending data only. Recipient identification
  via the `business_types_code='M'` rule with mutually-exclusive co-tag
  exclusions. Heuristic name-based carve-outs for Educational and Hospital.
  See `Methodology_FederalAssistance_501c3_Lightweight_FY22-FY25.docx`.

To run Phase 1 lightweight end-to-end:

```
python3 pipeline.py --lightweight --all
```

Or stage by stage:

```
python3 pipeline.py --lightweight --acquire    # same data acquisition path
python3 pipeline.py --lightweight --match      # M-with-exclusions filter (no BMF)
python3 pipeline.py --lightweight --classify   # heuristic panel carve-outs
python3 pipeline.py --lightweight --tables
python3 pipeline.py --lightweight --exhibits   # writes under exhibits/lightweight/
python3 pipeline.py --lightweight --qa
```

Lightweight outputs go under `exhibits/lightweight/` so they don't collide
with the full-pipeline outputs. Both can coexist in the same project tree;
when both have run, the lightweight pipeline produces a reconciliation
exhibit (`exhibit_15_reconciliation_*.csv`) comparing Topline numbers
side-by-side.

Lightweight wall time on FY22–FY25: ~15–30 min (no Tier 3 fuzzy match).

## Acquisition sources

The pipeline can pull USASpending data three ways. Pick one with `--acquire-source`:

| Source | Speed | Freshness | Network |
|---|---|---|---|
| `archive` (default) | ~10 min for FY22-FY25 | ~30-day snapshot lag | Required |
| `bulk_download` | 30 min - 2 hr per FY | Live | Required |
| `manual` | as fast as `unzip` | Whatever the local files are | **None** |

### Manual / offline mode

When USAspending's API and S3 endpoints are unreachable (outage, restricted
network, etc.), download zips by hand from
[usaspending.gov/download_center](https://www.usaspending.gov/download_center)
or copy them off another machine, drop them into:

```
raw/usaspending/manual/
  fy2022/  one or more zips covering FY22
  fy2023/  one or more zips covering FY23
  fy2024/  ...
  fy2025/  ...
```

Then run:

```
python3 pipeline.py --acquire --acquire-source manual
```

Both zip schemas are accepted in the same FY directory:

- **Custom Award Downloads** zips (`All_PrimeTransactions_*.zip`) - DAIMS / API column codes.
- **Award Data Archive** zips (`FY{YYYY}_{agency}_Assistance_Full_*.zip`) - Public-Profile column names; renamed automatically.

Multiple zips per FY are concatenated; sub-award zips are skipped; the
assistance award_type filter (02-11) is re-applied. Each zip is hashed and
recorded in `raw/usaspending/manual/manifest.json` and the run manifest, so
the reproducibility contract from Section 9 still holds.

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
