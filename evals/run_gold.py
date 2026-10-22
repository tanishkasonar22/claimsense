"""Run ClaimSense on the gold set and report every Tier 1 metric.

    python -m evals.run_gold                       # all gold claims, provider from .env
    python -m evals.run_gold --limit 20            # quick run (used by CI)
    python -m evals.run_gold --hand-checked-only   # only claims you verified by hand
    LLM_PROVIDER=gemini python -m evals.run_gold --limit 50

Writes evals/results/<timestamp>_<provider>.json and prints a summary table.
If MLFLOW_TRACKING_URI is set, also logs params and metrics to MLflow.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

from claimsense.config import get_settings
from claimsense.llm.prompts import PROMPT_VERSION
from claimsense.pipeline import Context, run_pipeline, submission_from_gold

from .metrics import confusion, p95, pr_auc, recall_at_budget, set_f1

RESULTS = Path(__file__).parent / "results"


def load_gold(hand_checked_only: bool = False, limit: int | None = None) -> list[dict]:
    path = get_settings().synthetic_dir / "gold" / "gold_claims.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if hand_checked_only:
        rows = [r for r in rows if r.get("hand_checked")]
    return rows[:limit] if limit else rows


def evaluate(gold: list[dict], ctx: Context | None = None) -> dict:
    ctx = ctx or Context.default()
    per, latencies, routes = [], [], {}
    for g in gold:
        ctx.customer = {}
        t0 = time.perf_counter()
        res, ctx = run_pipeline(submission_from_gold(g), ctx)
        latencies.append(time.perf_counter() - t0)
        for k, v in ctx.routes.items():
            routes.setdefault(k, {}).setdefault(v, 0)
            routes[k][v] += 1
        exp, f = g["expected"], res.facts
        vf = exp["voice_facts"]
        truth_consistent = "stated_parts_differ_from_damage" not in vf["contradictions"]
        per.append({
            "claim_id": g["claim_id"],
            "incident_type_ok": f.incident_type == vf["incident_type"],
            "stated_parts_f1": set_f1(f.stated_damaged_parts, vf["stated_damaged_parts"]),
            "date_ok": f.incident_date == vf["incident_date_said"],
            "police_ok": f.police_report_mentioned == vf["police_report_mentioned"],
            "shop_ok": f.repair_shop == vf["repair_shop"],
            "consistency_ok": f.parts_consistent == truth_consistent,
            "coverage_ok": res.coverage.covered == exp["coverage"]["covered"],
            "deductible_ok": (not exp["coverage"]["covered"]) or res.coverage.deductible == exp["coverage"]["deductible"],
            "citation_ok": set(exp["coverage"]["sections"]) <= set(res.coverage.sections) if exp["coverage"]["covered"] else True,
            "is_fraud": int(exp["is_fraud"]), "fraud_score": res.fraud.score,
            "gold_decision": exp["decision"], "pred_decision": res.recommendation.action,
            "injection_flagged": res.safety.injection_suspected,
            "transcript_source": res.transcript.source if res.transcript else "none",
            "vision_source": res.damage.source if res.damage else "none",
        })
    n = len(per)
    mean = lambda k: round(sum(p[k] for p in per) / n, 4)  # noqa: E731
    y, s = [p["is_fraud"] for p in per], [p["fraud_score"] for p in per]
    return {
        "n": n,
        "extraction": {k: mean(k) for k in ("incident_type_ok", "stated_parts_f1", "date_ok", "police_ok", "shop_ok", "consistency_ok")},
        "coverage": {k: mean(k) for k in ("coverage_ok", "deductible_ok", "citation_ok")},
        "fraud": {"pr_auc": round(pr_auc(y, s), 4), "recall_at_top_10pct": round(recall_at_budget(y, s, 0.10), 4),
                  "recall_at_top_25pct": round(recall_at_budget(y, s, 0.25), 4), "base_rate": round(sum(y) / n, 4)},
        "decision": {"agreement": round(sum(p["gold_decision"] == p["pred_decision"] for p in per) / n, 4),
                     "confusion": confusion([p["gold_decision"] for p in per], [p["pred_decision"] for p in per])},
        "safety": {"false_injection_flags_on_gold": sum(p["injection_flagged"] for p in per)},
        "latency_s": {"median": round(statistics.median(latencies), 3), "p95": round(p95(latencies), 3)},
        "routes": routes,
        "input_sources": {"transcript": dict(_count(per, "transcript_source")), "vision": dict(_count(per, "vision_source"))},
        "per_claim": per,
    }


def _count(per, key):
    out = {}
    for p in per:
        out[p[key]] = out.get(p[key], 0) + 1
    return out


def print_summary(r: dict) -> None:
    print(f"\nClaimSense gold evaluation  (n={r['n']})")
    print("-" * 52)
    for group in ("extraction", "coverage", "fraud"):
        for k, v in r[group].items():
            print(f"{group:<11} {k:<26} {v}")
    print(f"{'decision':<11} {'agreement':<26} {r['decision']['agreement']}")
    print(f"{'safety':<11} {'false flags on gold':<26} {r['safety']['false_injection_flags_on_gold']}")
    print(f"{'latency':<11} {'median / p95 (s)':<26} {r['latency_s']['median']} / {r['latency_s']['p95']}")
    print(f"routes: {r['routes']}")
    print(f"input sources: {r['input_sources']}")
    if r["input_sources"]["vision"].get("label_fallback") or r["input_sources"]["transcript"].get("script_fallback"):
        print("NOTE: some inputs used fallbacks (voice scripts instead of Whisper, labels instead of a vision model).")
        print("      Don't report these numbers as speech or vision accuracy.")
    print("\nDecision confusion (rows = correct, columns = predicted):")
    for g, row in r["decision"]["confusion"].items():
        print(f"  {g:<12} {row}")


def log_mlflow(r: dict, provider: str) -> None:
    uri = get_settings().mlflow_tracking_uri
    if not uri:
        return
    import mlflow
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("claimsense-gold")
    with mlflow.start_run(run_name=f"{provider}-{PROMPT_VERSION}"):
        mlflow.log_params({"llm_provider": provider, "prompt_version": PROMPT_VERSION, "n": r["n"],
                           "embedder": get_settings().embedder})
        for group in ("extraction", "coverage", "fraud"):
            mlflow.log_metrics({f"{group}.{k}": v for k, v in r[group].items()})
        mlflow.log_metric("decision.agreement", r["decision"]["agreement"])
        mlflow.log_metric("latency.median_s", r["latency_s"]["median"])


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--hand-checked-only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    gold = load_gold(args.hand_checked_only, args.limit)
    if not gold:
        sys.exit("No gold claims found. Run: python generate_all.py")
    provider = get_settings().llm_provider
    r = evaluate(gold)
    r["meta"] = {"provider": provider, "prompt_version": PROMPT_VERSION, "time": dt.datetime.now().isoformat(timespec="seconds")}
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{provider}.json"
    out.write_text(json.dumps(r, indent=2))
    if not args.quiet:
        print_summary(r)
        print(f"\nSaved {out}")
    log_mlflow(r, provider)
    return r


if __name__ == "__main__":
    main()
