"""Build the DPO preference dataset (booklet Phase 2.5, study guide B5).

    python -m training.dpo.build_pairs --out data/synthetic/finetune/dpo_pairs.jsonl

Sources:
  1. Adjuster edits from the dashboard (preferences table): edited summary = chosen,
     agent's original = rejected. These are the real, valuable pairs.
  2. Optional hand-written pairs in data/synthetic/finetune/dpo_handwritten.jsonl
     ({"prompt": ..., "chosen": ..., "rejected": ...} per line) to reach 300-500 pairs.
Gold claims are excluded so evaluation stays clean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from claimsense.config import get_settings
from claimsense.llm.prompts import SUMMARY_SYSTEM
from claimsense.storage import get_repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(get_settings().synthetic_dir / "finetune" / "dpo_pairs.jsonl"))
    args = ap.parse_args()
    rows = []
    for p in get_repo().preferences():
        if p.claim_id.startswith("G"):        # demo submissions of gold claims: keep them out
            continue
        rows.append({"prompt": [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": p.prompt}],
                     "chosen": [{"role": "assistant", "content": p.chosen}],
                     "rejected": [{"role": "assistant", "content": p.rejected}]})
    hand = get_settings().synthetic_dir / "finetune" / "dpo_handwritten.jsonl"
    if hand.exists():
        for line in hand.open():
            r = json.loads(line)
            rows.append({"prompt": [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": r["prompt"]}],
                         "chosen": [{"role": "assistant", "content": r["chosen"]}],
                         "rejected": [{"role": "assistant", "content": r["rejected"]}]})
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} preference pairs to {args.out}")


if __name__ == "__main__":
    main()
