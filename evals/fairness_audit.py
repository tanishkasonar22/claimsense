"""Fairness audit (booklet Phase 3.4, study guide F8).

    python -m evals.fairness_audit                       # audit historical adjuster decisions
    python -m evals.fairness_audit --decisions path.csv  # audit any decisions file

The synthetic history contains a PLANTED bias: valid claims in region R3 were wrongly
denied more often. This script measures it, then shows a simple mitigation.

Metrics per group (region, age_band, gender):
  approval_rate          share of claims approved                      (demographic parity)
  false_denial_rate      share of VALID claims that were denied         (equal opportunity)
  fraud_flag_rate        share of claims sent to fraud investigation

Uses Fairlearn's MetricFrame if installed (pip install fairlearn); otherwise pandas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from claimsense.config import get_settings

GROUPS = ["region", "age_band", "gender"]


def load(decisions: str | None) -> pd.DataFrame:
    if decisions:
        return pd.read_csv(decisions)
    df = pd.read_parquet(get_settings().synthetic_dir / "tabular" / "claims_history.parquet")
    return df.rename(columns={"historical_decision": "decision"})


def audit(df: pd.DataFrame, decision_col: str = "decision") -> dict:
    d = df.copy()
    d["approved"] = (d[decision_col] == "approve").astype(int)
    d["denied"] = d[decision_col].isin(["deny", "deny_fraud"]).astype(int)
    d["flagged"] = d[decision_col].isin(["investigate", "deny_fraud"]).astype(int)
    valid = d[d["correct_decision"] == "approve"]
    out = {}
    for g in GROUPS:
        table = pd.DataFrame({
            "claims": d.groupby(g).size(),
            "approval_rate": d.groupby(g)["approved"].mean(),
            "false_denial_rate": valid.groupby(g)["denied"].mean(),
            "fraud_flag_rate": d.groupby(g)["flagged"].mean(),
        }).round(4)
        gaps = {m: round(float(table[m].max() - table[m].min()), 4)
                for m in ("approval_rate", "false_denial_rate", "fraud_flag_rate")}
        out[g] = {"table": table.reset_index().to_dict("records"), "gaps": gaps}
    return out


def mitigate_by_review(df: pd.DataFrame, decision_col: str = "decision") -> pd.DataFrame:
    """A simple, honest mitigation: denials of claims the model considers covered and
    low-risk go to a second human review. Here we simulate that review catching the
    wrongful denials, then re-audit. In the live system this is a routing rule, not a
    change to anyone's outcome by group."""
    d = df.copy()
    second_review = (d[decision_col] == "deny") & (d["correct_covered"]) & (~d["is_fraud"])
    d.loc[second_review, decision_col] = "approve"
    return d


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions")
    args = ap.parse_args(argv)
    df = load(args.decisions)
    before = audit(df)
    after = audit(mitigate_by_review(df))
    print("False-denial rate of VALID claims by region (before -> after second-review mitigation).")
    print("NOTE: 'after' assumes the second reviewer is always right, so it is an UPPER BOUND.")
    print("      In Phase 3.4, route by your fraud model's score instead and report that number.")
    b = {r["region"]: r["false_denial_rate"] for r in before["region"]["table"]}
    a = {r["region"]: r["false_denial_rate"] for r in after["region"]["table"]}
    for reg in sorted(b):
        print(f"  {reg}: {b[reg]:.1%} -> {a[reg]:.1%}")
    for g in GROUPS:
        print(f"{g:<9} gaps before {before[g]['gaps']}  after {after[g]['gaps']}")
    try:
        from fairlearn.metrics import MetricFrame, selection_rate
        mf = MetricFrame(metrics={"approval_rate": selection_rate},
                         y_true=(df["correct_decision"] == "approve").astype(int),
                         y_pred=(df["decision"] == "approve").astype(int), sensitive_features=df["region"])
        print("\nFairlearn MetricFrame (approval rate by region):")
        print(mf.by_group.round(4).to_string())
    except ImportError:
        pass
    result = {"before": before, "after": after}
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "fairness_latest.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
