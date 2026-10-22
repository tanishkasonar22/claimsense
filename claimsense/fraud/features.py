"""Fraud features. ONE definition used by both training (training/fraud/train_fraud.py)
and serving (scorer.py), so the model never sees features computed two different ways
(a common cause of "it worked in the notebook but not in production").

Leakage rule: a feature may only use information available when the claim is filed.
History statistics are therefore computed from claims BEFORE the claim's report date.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from ..config import get_settings

FEATURES = [
    "days_since_policy_start",
    "amount_to_estimate_ratio",
    "parts_inconsistent",
    "date_mismatch_with_police",
    "customer_prior_claims",
    "shop_prior_claims",
    "shop_prior_fraud_rate",
    "claim_to_vehicle_value",
    "police_report_filed",
]

EXPLAIN = {
    "days_since_policy_start": "days between policy start and the incident",
    "amount_to_estimate_ratio": "claimed amount divided by the repair estimate",
    "parts_inconsistent": "parts described in the voice note differ from the estimate",
    "date_mismatch_with_police": "incident date in the voice note differs from the police report",
    "customer_prior_claims": "claims this customer filed before",
    "shop_prior_claims": "claims involving this repair shop before",
    "shop_prior_fraud_rate": "share of this shop's past claims confirmed as fraud",
    "claim_to_vehicle_value": "claimed amount divided by the vehicle's value",
    "police_report_filed": "a police report was filed",
}


@dataclass
class HistoryStats:
    """Point-in-time lookups over the claims history."""
    history: pd.DataFrame
    customers: pd.DataFrame

    def customer(self, customer_id: str) -> dict:
        row = self.customers.loc[self.customers["customer_id"] == customer_id]
        return row.iloc[0].to_dict() if len(row) else {}

    def before(self, date: dt.date) -> pd.DataFrame:
        return self.history[self.history["report_date"] < pd.Timestamp(date)]

    def customer_prior(self, customer_id: str, date: dt.date) -> int:
        h = self.before(date)
        return int((h["customer_id"] == customer_id).sum())

    def shop_prior(self, shop: str | None, date: dt.date) -> tuple[int, float]:
        if not shop:
            return 0, 0.0
        h = self.before(date)
        s = h[h["repair_shop"] == shop]
        if len(s) == 0:
            return 0, 0.0
        # "Confirmed fraud" = the historical investigation outcome, not the hidden truth label.
        confirmed = s["historical_decision"].isin(["deny_fraud"]).mean()
        return int(len(s)), float(confirmed)


@lru_cache
def load_history() -> HistoryStats:
    s = get_settings()
    tab = s.synthetic_dir / "tabular"
    hist = pd.read_parquet(tab / "claims_history.parquet")
    hist["report_date"] = pd.to_datetime(hist["report_date"])
    cust = pd.read_csv(tab / "customers.csv")
    return HistoryStats(hist, cust)


def claim_features(*, customer_id: str, report_date: dt.date, incident_date: dt.date | None,
                   police_date: dt.date | None, claim_amount: float | None, estimate_total: float | None,
                   parts_consistent: bool | None, repair_shop: str | None, police_report_filed: bool,
                   stats: HistoryStats | None = None) -> dict:
    stats = stats or load_history()
    cust = stats.customer(customer_id)
    start = pd.to_datetime(cust.get("policy_start")).date() if cust.get("policy_start") else None
    when = police_date or incident_date
    shop_n, shop_rate = stats.shop_prior(repair_shop, report_date)
    value = cust.get("vehicle_value") or None
    return {
        "days_since_policy_start": (when - start).days if (when and start) else None,
        "amount_to_estimate_ratio": (claim_amount / estimate_total) if (claim_amount and estimate_total) else None,
        "parts_inconsistent": None if parts_consistent is None else int(not parts_consistent),
        "date_mismatch_with_police": int(incident_date != police_date) if (incident_date and police_date) else 0,
        "customer_prior_claims": stats.customer_prior(customer_id, report_date),
        "shop_prior_claims": shop_n,
        "shop_prior_fraud_rate": shop_rate,
        "claim_to_vehicle_value": (claim_amount / value) if (claim_amount and value) else None,
        "police_report_filed": int(bool(police_report_filed)),
    }
