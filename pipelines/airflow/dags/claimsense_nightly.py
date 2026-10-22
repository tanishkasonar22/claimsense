"""Nightly ClaimSense pipeline (booklet Phase 3.2, study guide E4).

    fraud features (Spark) -> dbt run -> dbt test -> fairness audit -> full eval -> drift check

Mount the repo into the Airflow containers at /opt/claimsense (see docker-compose.yml, profile "pipelines").
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

REPO = "/opt/claimsense"
DEFAULT = {"retries": 2, "retry_delay": timedelta(minutes=5)}


@dag(schedule="0 2 * * *", start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULT,
     tags=["claimsense"], doc_md=__doc__)
def claimsense_nightly():
    spark_features = BashOperator(task_id="spark_fraud_features", cwd=REPO,
                                  bash_command="python -m training.fraud.spark_features")
    dbt_run = BashOperator(task_id="dbt_run", cwd=f"{REPO}/pipelines/dbt",
                           bash_command="dbt run --profiles-dir .")
    dbt_test = BashOperator(task_id="dbt_test", cwd=f"{REPO}/pipelines/dbt",
                            bash_command="dbt test --profiles-dir .")
    fairness = BashOperator(task_id="fairness_audit", cwd=REPO, bash_command="python -m evals.fairness_audit")
    full_eval = BashOperator(task_id="full_gold_eval", cwd=REPO, bash_command="python -m evals.run_gold --quiet")

    @task
    def drift_check() -> dict:
        """Compare this week's claim mix with the training data. Large shifts = retrain soon."""
        import pandas as pd
        hist = pd.read_parquet(f"{REPO}/data/synthetic/tabular/claims_history.parquet")
        hist["report_date"] = pd.to_datetime(hist["report_date"])
        recent = hist[hist["report_date"] >= hist["report_date"].max() - pd.Timedelta(days=7)]
        base = hist[hist["report_date"] < hist["report_date"].max() - pd.Timedelta(days=90)]

        def dist(df, col):
            return df[col].value_counts(normalize=True)

        report = {}
        for col in ("incident_type", "product_code", "region"):
            a, b = dist(base, col), dist(recent, col)
            idx = a.index.union(b.index)
            report[col] = round(float(0.5 * (a.reindex(idx, fill_value=0) - b.reindex(idx, fill_value=0)).abs().sum()), 4)  # total variation
        report["alert"] = any(v > 0.15 for v in report.values() if isinstance(v, float))
        Path(f"{REPO}/evals/results").mkdir(parents=True, exist_ok=True)
        Path(f"{REPO}/evals/results/drift_latest.json").write_text(json.dumps(report, indent=2))
        return report

    spark_features >> dbt_run >> dbt_test >> fairness >> full_eval >> drift_check()


claimsense_nightly()

