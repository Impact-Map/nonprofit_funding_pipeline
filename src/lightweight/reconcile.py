"""Reconciliation exhibit: lightweight Topline vs. BMF-backed Topline by FY.

Best-effort: produces output only if the full pipeline's BMF-backed analytic
table (`processed/assistance_txn_501c3.parquet`) exists in the same project
tree. Otherwise emits a placeholder CSV explaining no comparison is available.

Per methodology Section 13.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .. import config

LOG = logging.getLogger(__name__)


def produce(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.EXHIBITS
    out_dir.mkdir(parents=True, exist_ok=True)

    light_path = config.PROCESSED / "assistance_txn_501c3_lightweight.parquet"
    bmf_path = config.PROCESSED / "assistance_txn_501c3.parquet"

    out_path = out_dir / "exhibit_15_reconciliation_lightweight_vs_bmf.csv"

    if not light_path.exists():
        msg = pd.DataFrame([{
            "note": "Lightweight analytic table missing; run --lightweight pipeline first."
        }])
        msg.to_csv(out_path, index=False)
        return out_path

    if not bmf_path.exists():
        msg = pd.DataFrame([{
            "note": ("BMF-backed analytic table not found at "
                     f"{bmf_path}. Reconciliation requires the full pipeline "
                     "to have run in this project tree. Re-run the full "
                     "pipeline to populate it; this exhibit will then "
                     "regenerate with side-by-side comparison.")
        }])
        msg.to_csv(out_path, index=False)
        return out_path

    light = pd.read_parquet(light_path)
    bmf = pd.read_parquet(bmf_path)

    def _summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
        df = df.copy()
        df["federal_action_obligation"] = pd.to_numeric(
            df["federal_action_obligation"], errors="coerce"
        ).fillna(0.0)
        df["federal_action_obligation_real"] = pd.to_numeric(
            df["federal_action_obligation_real"], errors="coerce"
        ).fillna(0.0)
        rep = df.groupby("fy", dropna=False).agg(
            nominal=("federal_action_obligation", "sum"),
            real_fy25=("federal_action_obligation_real", "sum"),
            unique_recipients=("recipient_uei", lambda s: s.nunique()),
            transactions=("transaction_id", "count"),
        ).reset_index()
        rep.columns = ["fy"] + [f"{c}_{label}" for c in rep.columns[1:]]
        return rep

    summary = _summary(light, "lightweight").merge(
        _summary(bmf, "bmf"), on="fy", how="outer"
    )
    summary["delta_nominal"] = summary["nominal_lightweight"] - summary["nominal_bmf"]
    summary["delta_pct"] = summary["delta_nominal"] / summary["nominal_bmf"]

    # Per-panel
    rows = []
    for panel in ("core", "educational", "hospital", "international"):
        l_p = light[light["recipient_category"] == panel]
        b_p = bmf[bmf["recipient_category"] == panel]
        for fy in sorted(set(l_p["fy"].dropna()) | set(b_p["fy"].dropna())):
            l_obl = pd.to_numeric(
                l_p.loc[l_p["fy"] == fy, "federal_action_obligation"], errors="coerce"
            ).fillna(0).sum()
            b_obl = pd.to_numeric(
                b_p.loc[b_p["fy"] == fy, "federal_action_obligation"], errors="coerce"
            ).fillna(0).sum()
            rows.append({
                "fy": int(fy), "panel": panel,
                "nominal_lightweight": l_obl, "nominal_bmf": b_obl,
                "delta_nominal": l_obl - b_obl,
                "delta_pct": (l_obl - b_obl) / b_obl if b_obl else float("nan"),
            })
    panel_summary = pd.DataFrame(rows)

    # Recipient-set overlap
    light_ueis = set(light["recipient_uei"].dropna().unique())
    bmf_ueis = set(bmf["recipient_uei"].dropna().unique())
    overlap = pd.DataFrame([{
        "lightweight_only_recipients": len(light_ueis - bmf_ueis),
        "bmf_only_recipients": len(bmf_ueis - light_ueis),
        "both_recipients": len(light_ueis & bmf_ueis),
        "lightweight_total": len(light_ueis),
        "bmf_total": len(bmf_ueis),
    }])

    # Write three sheets as separate CSVs (CSV doesn't support sheets natively).
    summary.to_csv(out_dir / "exhibit_15_reconciliation_by_fy.csv", index=False)
    panel_summary.to_csv(out_dir / "exhibit_15_reconciliation_by_panel.csv", index=False)
    overlap.to_csv(out_dir / "exhibit_15_reconciliation_recipient_overlap.csv", index=False)
    out_path = out_dir / "exhibit_15_reconciliation_by_fy.csv"
    LOG.info("Reconciliation exhibits written to %s", out_dir)
    return out_path
