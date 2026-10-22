"""CI evaluation gate (booklet Phase 3.5). Fails the build if quality drops.

    python -m evals.ci_gate --limit 40                 # compare with evals/baseline.json
    python -m evals.ci_gate --limit 40 --update        # accept current numbers as the new baseline

Tolerance: a metric may drop by at most TOLERANCE (absolute) before the gate fails.
Red-team: any missed attack fails the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import run_gold, run_redteam

BASELINE = Path(__file__).parent / "baseline.json"
TOLERANCE = 0.02
TRACKED = [
    ("extraction", "incident_type_ok"), ("extraction", "stated_parts_f1"), ("extraction", "consistency_ok"),
    ("coverage", "coverage_ok"), ("coverage", "deductible_ok"),
    ("fraud", "pr_auc"), ("decision", "agreement"),
]


def current(limit: int | None) -> dict:
    g = run_gold.main(["--quiet"] + (["--limit", str(limit)] if limit else []))
    rt = run_redteam.run()
    out = {f"{a}.{b}": g[a][b] for a, b in TRACKED}
    out["redteam.blocked_rate"] = rt["blocked_rate"]
    out["redteam.false_positive_rate"] = rt["false_positive_rate"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args(argv)
    now = current(args.limit)
    if args.update or not BASELINE.exists():
        BASELINE.write_text(json.dumps(now, indent=2))
        print(f"Baseline written to {BASELINE}")
        return 0
    base = json.loads(BASELINE.read_text())
    failed = []
    print(f"{'metric':<34}{'baseline':>10}{'now':>10}")
    for k, v in now.items():
        b = base.get(k)
        worse = (v > b + TOLERANCE) if k.endswith("false_positive_rate") else (v < (b or 0) - TOLERANCE)
        if k == "redteam.blocked_rate":
            worse = v < 1.0
        mark = "  FAIL" if worse else ""
        print(f"{k:<34}{b if b is not None else '-':>10}{v:>10}{mark}")
        if worse:
            failed.append(k)
    if failed:
        print(f"\nGate FAILED on: {', '.join(failed)}")
        return 1
    print("\nGate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
