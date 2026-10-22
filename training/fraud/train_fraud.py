"""Train the fraud model (booklet Phase 2.4, study guide E8).

    python -m training.fraud.train_fraud                  # LightGBM, time-based split
    python -m training.fraud.train_fraud --tune           # + Ray Tune hyperparameter search

Point-in-time features: every history statistic (prior claims, shop fraud rate) uses only
claims reported BEFORE the claim being scored. Computing them over the whole table would leak
the future into training and make the model look better than it is.

Writes models/fraud_lgbm.txt (+ .json with feature list and metrics). The API picks it up
automatically through FRAUD_MODEL_PATH.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from claimsense.config import get_settings
from claimsense.fraud.features import FEATURES
from claimsense.fraud.scorer import score_rules
from evals.metrics import pr_auc, recall_at_budget

SPLIT_DATE = "2025-07-01"   # train on claims reported before, test on claims after


def build_features(hist: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    d = hist.copy()
    d["report_date"] = pd.to_datetime(d["report_date"])
    d = d.sort_values(["report_date", "claim_id"]).reset_index(drop=True)
    d["customer_prior_claims"] = d.groupby("customer_id").cumcount()
    d["shop_prior_claims"] = d.groupby("repair_shop").cumcount()
    confirmed = (d["historical_decision"] == "deny_fraud").astype(float)
    # mean of the shop's PREVIOUS outcomes (shift(1) excludes the current claim)
    d["shop_prior_fraud_rate"] = (confirmed.groupby(d["repair_shop"])
                                  .transform(lambda s: s.shift(1).expanding().mean())).fillna(0.0)
    d["days_since_policy_start"] = d["days_since_policy_start"].astype(float)
    d["amount_to_estimate_ratio"] = d["claim_amount"] / d["estimate_total"]
    d["parts_inconsistent"] = (d["stated_damaged_parts"] != d["true_damaged_parts"]).astype(int)
    d["date_mismatch_with_police"] = 0     # history has no voice notes; the live system fills this
    d["claim_to_vehicle_value"] = d["claim_amount"] / d["vehicle_value"]
    d["police_report_filed"] = d["police_report_filed"].astype(int)
    d["label"] = d["is_fraud"].astype(int)
    return d


def evaluate(y: np.ndarray, scores: np.ndarray) -> dict:
    y, scores = [int(v) for v in y], [float(v) for v in scores]
    return {"pr_auc": round(pr_auc(y, scores), 4),
            "recall_at_top_10pct": round(recall_at_budget(y, scores, 0.10), 4),
            "base_rate": round(sum(y) / max(1, len(y)), 4)}


def train_lgbm(train: pd.DataFrame, valid: pd.DataFrame, params: dict | None = None):
    import lightgbm as lgb
    p = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40,
         "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "verbose": -1,
         "scale_pos_weight": float((train.label == 0).sum() / max(1, (train.label == 1).sum()))}
    p.update(params or {})
    dtrain = lgb.Dataset(train[FEATURES], train["label"])
    dvalid = lgb.Dataset(valid[FEATURES], valid["label"])
    return lgb.train(p, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def tune(train: pd.DataFrame, valid: pd.DataFrame) -> dict:   # pragma: no cover - needs ray
    from ray import tune as rt

    def objective(config):
        m = train_lgbm(train, valid, config)
        rt.report({"pr_auc": pr_auc(list(valid.label), list(m.predict(valid[FEATURES])))})

    space = {"num_leaves": rt.choice([15, 31, 63]), "learning_rate": rt.loguniform(0.01, 0.2),
             "min_data_in_leaf": rt.choice([20, 40, 80]), "feature_fraction": rt.uniform(0.6, 1.0)}
    res = rt.Tuner(objective, param_space=space,
                   tune_config=rt.TuneConfig(metric="pr_auc", mode="max", num_samples=20)).fit()
    return res.get_best_result().config


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args(argv)
    s = get_settings()
    tab = s.synthetic_dir / "tabular"
    d = build_features(pd.read_parquet(tab / "claims_history.parquet"), pd.read_csv(tab / "customers.csv"))
    train_all, test = d[d.report_date < SPLIT_DATE], d[d.report_date >= SPLIT_DATE]
    cut = int(len(train_all) * 0.85)              # last 15% of training period = validation
    train, valid = train_all.iloc[:cut], train_all.iloc[cut:]
    print(f"train {len(train):,}  valid {len(valid):,}  test {len(test):,}  (split at {SPLIT_DATE})")

    rules = test.apply(lambda r: score_rules({f: r[f] for f in FEATURES}).score, axis=1).to_numpy()
    results = {"rules_baseline": evaluate(test.label.to_numpy(), rules)}

    params = tune(train, valid) if args.tune else None
    model = train_lgbm(train, valid, params)
    preds = model.predict(test[FEATURES])
    results["lightgbm"] = evaluate(test.label.to_numpy(), preds)
    results["feature_importance_gain"] = dict(sorted(
        zip(FEATURES, model.feature_importance("gain").round(1).tolist()), key=lambda t: -t[1]))

    out = s.model_dir
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "fraud_lgbm.txt"))
    (out / "fraud_lgbm.json").write_text(json.dumps({"features": FEATURES, "split_date": SPLIT_DATE,
                                                    "params": params, "results": results}, indent=2))
    print(json.dumps(results, indent=2))
    if s.mlflow_tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        mlflow.set_experiment("claimsense-fraud")
        with mlflow.start_run(run_name="lightgbm"):
            mlflow.log_params(params or {"default": True})
            mlflow.log_metrics({f"{k}.{m}": v for k, r in results.items() if k != "feature_importance_gain" for m, v in r.items()})
            mlflow.log_artifact(str(out / "fraud_lgbm.txt"))
    return results


if __name__ == "__main__":
    main()
