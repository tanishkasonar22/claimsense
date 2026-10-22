"""Generate every synthetic ClaimSense dataset in one run.

    python generate_all.py                 # default: 2,000 customers, 20,000 history claims
    python generate_all.py --claims 1000000 --customers 50000   # Spark-scale history (Phase 2.4)

Output goes to data/synthetic/. The same --seed always gives the same data.
"""
from __future__ import annotations

import argparse
import json
import time

from generators import claims, documents, evidence, finetune, policies, population, redteam
from generators.common import SYNTH, Settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--customers", type=int, default=2_000)
    ap.add_argument("--claims", type=int, default=20_000)
    ap.add_argument("--gold", type=int, default=200)
    ap.add_argument("--sft", type=int, default=2_500)
    args = ap.parse_args()

    s = Settings(seed=args.seed, n_customers=args.customers, n_history_claims=args.claims,
                 n_gold_candidates=args.gold)
    t0, summary = time.time(), {}

    summary |= policies.generate()
    customers, rings, st = population.generate(s); summary |= st
    history, st = claims.generate_history(s, customers, rings); summary |= st
    customers_by_id = {c["customer_id"]: c for c in customers.to_dict("records")}

    gold = claims.generate_gold_candidates(s, customers, rings)
    evidence.enrich(s, gold, customers_by_id)
    summary |= documents.generate(s, gold)
    summary |= evidence.generate(s, gold)
    summary |= redteam.generate(s, gold)
    summary |= finetune.generate(s, history, customers_by_id, n=args.sft)

    summary["seconds"] = round(time.time() - t0, 1)
    summary["settings"] = s.__dict__
    (SYNTH / "manifest.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
