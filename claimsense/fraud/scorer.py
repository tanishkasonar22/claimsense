"""Fraud scoring with reasons.

  Rules baseline   A hand-weighted logistic score. Transparent; the row to beat.
  LightGBM model   Trained by training/fraud/train_fraud.py. Reasons come from LightGBM's
                   per-feature contributions (pred_contrib=True), which are SHAP values
                   for tree models, so you don't need the shap package at serving time.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from ..schemas import FraudAssessment, FraudReason
from .features import EXPLAIN, FEATURES

# (feature, test, weight) - each rule adds its weight to the log-odds when it fires.
RULES = [
    ("days_since_policy_start", lambda v: v is not None and v <= 25, 1.6),
    ("amount_to_estimate_ratio", lambda v: v is not None and v > 1.5, 2.8),
    ("parts_inconsistent", lambda v: v == 1, 2.8),
    ("date_mismatch_with_police", lambda v: v == 1, 1.5),
    ("shop_prior_fraud_rate", lambda v: v is not None and v >= 0.2, 1.8),
    ("customer_prior_claims", lambda v: v is not None and v >= 3, 0.6),
]
BIAS = -2.6


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def score_rules(features: dict) -> FraudAssessment:
    logit, reasons = BIAS, []
    for name, test, w in RULES:
        v = features.get(name)
        if test(v):
            logit += w
            reasons.append(FraudReason(feature=name, value=_round(v), contribution=w, explanation=EXPLAIN[name]))
    reasons.sort(key=lambda r: -r.contribution)
    return FraudAssessment(score=round(_sigmoid(logit), 4), reasons=reasons[:3], model="rules")


@lru_cache
def _booster(path: str):
    import lightgbm as lgb
    return lgb.Booster(model_file=path)


def score_model(features: dict, model_path: str) -> FraudAssessment:
    import numpy as np
    meta_path = Path(model_path).with_suffix(".json")
    feats = json.loads(meta_path.read_text())["features"] if meta_path.exists() else FEATURES
    x = np.array([[np.nan if features.get(f) is None else float(features[f]) for f in feats]])
    booster = _booster(model_path)
    score = float(booster.predict(x)[0])
    contrib = booster.predict(x, pred_contrib=True)[0][:-1]   # last value is the bias term
    top = sorted(zip(feats, contrib), key=lambda t: -t[1])[:3]
    reasons = [FraudReason(feature=f, value=_round(features.get(f)), contribution=round(float(c), 3),
                           explanation=EXPLAIN.get(f, f)) for f, c in top if c > 0]
    return FraudAssessment(score=round(score, 4), reasons=reasons, model=f"lightgbm:{Path(model_path).name}")


def score_claim(features: dict) -> FraudAssessment:
    path = get_settings().fraud_model_path
    if path and Path(path).exists():
        try:
            return score_model(features, path)
        except ImportError:
            pass
    return score_rules(features)


def _round(v):
    return round(v, 3) if isinstance(v, float) else v
