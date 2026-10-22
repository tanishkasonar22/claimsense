"""Attach real CarDD photos to the gold claims (run after you download CarDD).

    python scripts/assemble_gold_with_cardd.py --cardd data/public/cardd

For each gold claim's photo requests (damage type + severity), pick a matching image from
CarDD's TEST split only, so no gold photo is ever used to train the vision model.
Fills inputs.photos in data/synthetic/gold/gold_claims.jsonl (a backup is written first).

Expects the COCO-format folder from the CarDD download (CarDD_COCO/annotations/*.json).
Check your download's folder names and adjust --split-file if they differ.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from claimsense.config import get_settings  # noqa: E402
from claimsense.vision import CARDD_TO_DAMAGE, severity_from_area  # noqa: E402


def load_test_images(cardd: Path, split_file: str) -> dict[tuple[str, str], list[str]]:
    direct = cardd / "annotations" / split_file          # fast path: no folder listing over Drive
    ann_path = direct if direct.exists() else next(cardd.rglob(split_file), None)
    if ann_path is None:
        sys.exit(f"Could not find {split_file} under {cardd}. Pass --split-file with the test annotation name.")
    coco = json.loads(ann_path.read_text())
    cats = {c["id"]: CARDD_TO_DAMAGE.get(c["name"].lower()) for c in coco["categories"]}
    imgs = {i["id"]: i for i in coco["images"]}
    img_dir = next((p for p in ann_path.parent.parent.iterdir() if p.is_dir() and "test" in p.name.lower()), None)
    pool: dict[tuple[str, str], list[str]] = defaultdict(list)
    for a in coco["annotations"]:
        dtype = cats.get(a["category_id"])
        if not dtype:
            continue
        im = imgs[a["image_id"]]
        sev = severity_from_area(a.get("area", 0) / (im["width"] * im["height"]))
        # Store "cardd://<path inside the CarDD folder>" so the same gold file works wherever
        # CarDD lives (Google Drive in Colab, Drive for desktop on a Mac): set CARDD_DIR there.
        full = (img_dir / im["file_name"]) if img_dir else Path(im["file_name"])
        try:
            rel = "cardd://" + full.relative_to(cardd).as_posix()
        except ValueError:
            rel = str(full)
        pool[(dtype, sev)].append(rel)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cardd", default=str(get_settings().cardd_dir))
    ap.add_argument("--split-file", default="instances_test2017.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pool = load_test_images(Path(args.cardd), args.split_file)
    print({k: len(v) for k, v in sorted(pool.items())})

    gold_path = get_settings().synthetic_dir / "gold" / "gold_claims.jsonl"
    shutil.copy(gold_path, gold_path.with_suffix(".jsonl.bak"))
    rows = [json.loads(line) for line in gold_path.open()]
    filled = 0
    for g in rows:
        photos = []
        for req in g["inputs"]["photo_requests"]:
            options = pool.get((req["damage_type"], req["severity"])) or \
                [p for (t, _), ps in pool.items() if t == req["damage_type"] for p in ps]
            if options:
                photos.append(rng.choice(options))
        g["inputs"]["photos"] = photos
        filled += bool(photos)
    gold_path.write_text("".join(json.dumps(g) + "\n" for g in rows))
    print(f"attached photos to {filled}/{len(rows)} gold claims (backup at {gold_path.with_suffix('.jsonl.bak')})")


if __name__ == "__main__":
    main()
