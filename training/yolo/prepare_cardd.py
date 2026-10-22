"""Convert CarDD (COCO format) to YOLO segmentation format (booklet Phase 2.1).

    python -m training.yolo.prepare_cardd --cardd data/public/cardd --out data/public/cardd_yolo

Writes images/{train,val,test}, labels/{train,val,test} and car_damage.yaml.
Each label line: <class> x1 y1 x2 y2 ... (polygon points normalized to 0-1).
Keep CarDD's own split, and never move test images into training.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CLASSES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
SPLITS = {"train": "instances_train2017.json", "val": "instances_val2017.json", "test": "instances_test2017.json"}


def convert_split(cardd: Path, out: Path, split: str, ann_name: str) -> int:
    ann_path = next(cardd.rglob(ann_name), None)
    if ann_path is None:
        print(f"  skip {split}: {ann_name} not found")
        return 0
    coco = json.loads(ann_path.read_text())
    cat_to_idx = {}
    for c in coco["categories"]:
        name = c["name"].lower().replace(" ", "_")
        if name in CLASSES:
            cat_to_idx[c["id"]] = CLASSES.index(name)
    images = {i["id"]: i for i in coco["images"]}
    # CarDD_COCO/ has annotations/ next to train2017/, val2017/, test2017/
    src_dir = next(p for p in ann_path.parent.parent.iterdir()
                   if p.is_dir() and p.name != "annotations" and p.name.lower().startswith(split))
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    lines: dict[int, list[str]] = {}
    for a in coco["annotations"]:
        if a["category_id"] not in cat_to_idx or not a.get("segmentation") or not isinstance(a["segmentation"], list):
            continue
        im = images[a["image_id"]]
        w, h = im["width"], im["height"]
        for poly in a["segmentation"]:
            pts = " ".join(f"{(x / w if i % 2 == 0 else x / h):.5f}" for i, x in enumerate(poly))
            lines.setdefault(a["image_id"], []).append(f"{cat_to_idx[a['category_id']]} {pts}")
    for img_id, im in images.items():
        src = src_dir / im["file_name"]
        if not src.exists():
            continue
        shutil.copy(src, out / "images" / split / src.name)
        (out / "labels" / split / f"{src.stem}.txt").write_text("\n".join(lines.get(img_id, [])))
    return len(images)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cardd", default="data/public/cardd")
    ap.add_argument("--out", default="data/public/cardd_yolo")
    args = ap.parse_args()
    cardd, out = Path(args.cardd), Path(args.out)
    for split, ann in SPLITS.items():
        print(f"{split}: {convert_split(cardd, out, split, ann)} images")
    yaml = (f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (out / "car_damage.yaml").write_text(yaml)
    print(f"wrote {out / 'car_damage.yaml'}")


if __name__ == "__main__":
    main()
