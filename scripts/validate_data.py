"""Sanity-check the generated data and print a short report.

    python scripts/validate_data.py

Checks that the planted signals exist (fraud patterns, regional bias), that labels are
consistent, and that gold claims never leak into fine-tuning data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "data" / "synthetic"
problems: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        problems.append(msg)


h = pd.read_parquet(S / "tabular" / "claims_history.parquet")
print(f"\nHistory: {len(h):,} claims, {h['customer_id'].nunique():,} customers")
print(f"  date range {h['incident_date'].min()} to {h['incident_date'].max()}")
print(f"  fraud rate {h['is_fraud'].mean():.1%}")

pat = h.loc[h["is_fraud"], "fraud_patterns"].str.split(";").explode().value_counts()
print("  fraud patterns:", dict(pat))

print("\nCoverage outcomes (correct labels):")
print(h["coverage_reason"].value_counts().to_string())

print("\nCorrect vs historical decisions:")
print(pd.crosstab(h["correct_decision"], h["historical_decision"]).to_string())

valid = h[h["correct_decision"] == "approve"]
denial = valid.groupby("region")["historical_decision"].apply(lambda x: (x == "deny").mean())
print("\nWrongful denial rate of VALID claims by region (planted bias in R3):")
print(denial.map("{:.1%}".format).to_string())

print("\nSignal checks")
early = h["days_since_policy_start"] <= 25
check(h.loc[early, "is_fraud"].mean() > 2 * h.loc[~early, "is_fraud"].mean(), "early claims are more often fraud")
infl = h["claim_amount"] / h["estimate_total"] > 1.5
check(h.loc[infl, "is_fraud"].mean() > 0.9, "claim amount far above estimate is almost always fraud")
mism = h["stated_damaged_parts"] != h["true_damaged_parts"]
check(h.loc[mism, "is_fraud"].all(), "stated/true part mismatch only occurs in fraud")
check(denial["R3"] > 2 * denial.drop("R3").mean(), "regional bias is present for the fairness audit")
check(not h.loc[~h["correct_covered"], "correct_decision"].ne("deny").any(), "uncovered claims are always 'deny'")
check((h["incident_date"] >= h["policy_start"]).all(), "incidents happen after policy start")

gold = [json.loads(l) for l in open(S / "gold" / "gold_claims.jsonl")]
sft_ids = {json.loads(l)["id"] for f in ("sft_train.jsonl", "sft_val.jsonl") for l in open(S / "finetune" / f)}
gold_ids = {g["claim_id"] for g in gold}
print(f"\nGold: {len(gold)} claims, fraud {sum(g['expected']['is_fraud'] for g in gold) / len(gold):.0%}")
check(not (gold_ids & sft_ids), "no gold claim appears in fine-tuning data")
missing = [d for g in gold for d in g["inputs"]["documents"] if not (S / d).exists()]
check(not missing, f"all gold documents exist ({len(missing)} missing)")

attacks = [json.loads(l) for l in open(S / "redteam" / "attacks.jsonl")]
print(f"\nRed team: {len(attacks)} items")
print(pd.Series([a["category"] for a in attacks]).value_counts().to_string())

print("\n" + ("All checks passed." if not problems else f"{len(problems)} problem(s): {problems}"))
sys.exit(1 if problems else 0)
