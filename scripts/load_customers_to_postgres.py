"""Load the synthetic customers table into Postgres so dbt can join claims to groups.

    python scripts/load_customers_to_postgres.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claimsense.config import get_settings  # noqa: E402


def main() -> None:
    from sqlalchemy import create_engine
    s = get_settings()
    if not s.database_url.startswith("postgresql"):
        sys.exit("DATABASE_URL must point at Postgres (full mode).")
    df = pd.read_csv(s.synthetic_dir / "tabular" / "customers.csv")
    df.to_sql("customers", create_engine(s.database_url), if_exists="replace", index=False)
    print(f"loaded {len(df):,} customers")


if __name__ == "__main__":
    main()
