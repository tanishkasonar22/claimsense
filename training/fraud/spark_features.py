"""Fraud features at scale with PySpark (booklet Phase 2.4, study guide E3).

    python generate_all.py --claims 1000000 --customers 50000    # make Spark worthwhile
    python -m training.fraud.spark_features                       # writes data/features/fraud_features.parquet

Same features as training/fraud/train_fraud.py, computed with window functions so they
run on millions of rows across all your CPU cores (local[*]) or a cluster.
Needs Java 17 and: pip install pyspark
"""
from __future__ import annotations

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from claimsense.config import get_settings


def main() -> None:
    s = get_settings()
    spark = (SparkSession.builder.master("local[*]").appName("claimsense-fraud-features")
             .config("spark.sql.shuffle.partitions", "16").getOrCreate())
    src = str(s.synthetic_dir / "tabular" / "claims_history.parquet")
    claims = spark.read.parquet(src).withColumn("report_ts", F.col("report_date").cast("timestamp"))

    by_customer = Window.partitionBy("customer_id").orderBy("report_ts", "claim_id")
    by_shop = Window.partitionBy("repair_shop").orderBy("report_ts", "claim_id")
    prior_shop = by_shop.rowsBetween(Window.unboundedPreceding, -1)       # strictly earlier claims
    day = F.col("report_ts").cast("long") / 86400
    shop_90d = Window.partitionBy("repair_shop").orderBy(day).rangeBetween(-90, -1)

    feats = (claims
             .withColumn("customer_prior_claims", F.row_number().over(by_customer) - 1)
             .withColumn("shop_prior_claims", F.row_number().over(by_shop) - 1)
             .withColumn("shop_prior_fraud_rate",
                         F.coalesce(F.avg((F.col("historical_decision") == "deny_fraud").cast("double")).over(prior_shop), F.lit(0.0)))
             .withColumn("shop_claims_last_90d", F.count("*").over(shop_90d))
             .withColumn("amount_to_estimate_ratio", F.col("claim_amount") / F.col("estimate_total"))
             .withColumn("parts_inconsistent", (F.col("stated_damaged_parts") != F.col("true_damaged_parts")).cast("int"))
             .withColumn("claim_to_vehicle_value", F.col("claim_amount") / F.col("vehicle_value"))
             .withColumn("police_report_filed", F.col("police_report_filed").cast("int"))
             .withColumn("date_mismatch_with_police", F.lit(0))
             .withColumn("label", F.col("is_fraud").cast("int")))

    cols = ["claim_id", "customer_id", "report_date", "days_since_policy_start", "amount_to_estimate_ratio",
            "parts_inconsistent", "date_mismatch_with_police", "customer_prior_claims", "shop_prior_claims",
            "shop_prior_fraud_rate", "shop_claims_last_90d", "claim_to_vehicle_value", "police_report_filed", "label"]
    out = s.data_dir / "features" / "fraud_features.parquet"
    feats.select(*cols).write.mode("overwrite").parquet(str(out))
    print(f"wrote {feats.count():,} rows to {out}")
    spark.stop()


if __name__ == "__main__":
    main()
