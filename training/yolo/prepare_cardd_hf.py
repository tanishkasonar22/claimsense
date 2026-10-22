"""Convert the Hugging Face CarDD mirror (mouen5/CarDD, FiftyOne format) into the official
COCO layout, so prepare_cardd.py and assemble_gold_with_cardd.py work unchanged.

Run in Colab (recommended, keeps the 2.2 GB off your laptop):

    pip install fiftyone huggingface_hub
    from google.colab import drive; drive.mount('/content/drive')
    !hf download mouen5/CarDD --repo-type=dataset --local-dir /content/cardd_hf
    !python -m training.yolo.prepare_cardd_hf --src /content/cardd_hf \
        --out /content/drive/MyDrive/claimsense/data/public/cardd

Output (same as the official download):
    <out>/CarDD_COCO/annotations/instances_{train,val,test}2017.json
    <out>/CarDD_COCO/{train,val,test}2017/*.jpg

About splits: the mirror's schema has no split field. This script uses the sample tags if
they contain train/val/test. If they don't, it makes a deterministic 70/20/10 split by file
name and prints a warning: your numbers will then NOT be comparable with the CarDD paper,
and you should say so in your README. (CarDD removed near-duplicate images before release,
so a fresh random split doesn't leak duplicates.)

Label fields in the mirror: `segmentations` (masks) and `detections` (boxes); classes
"dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat".
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

SPLIT_TAGS = {"train": "train", "training": "train", "val": "val", "validation": "val", "valid": "val", "test": "test"}


def split_of(sample) -> str | None:
    for t in sample.tags or []:
        if t.lower() in SPLIT_TAGS:
            return SPLIT_TAGS[t.lower()]
    return None


def hash_split(filepath: str) -> str:
    h = int(hashlib.md5(Path(filepath).name.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 90 else "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder from `hf download mouen5/CarDD`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-field", default="segmentations")
    args = ap.parse_args()

    import fiftyone as fo

    ds = fo.Dataset.from_dir(dataset_dir=args.src, dataset_type=fo.types.FiftyOneDataset, name="cardd_hf",
                             overwrite=True)
    print(ds)
    tagged = [split_of(s) for s in ds]
    if all(tagged):
        print("Using the split from sample tags:", Counter(tagged))
    else:
        print("WARNING: no split tags found. Making a deterministic 70/20/10 split by file name.\n"
              "         Results won't be comparable with the CarDD paper's official test set.")
    for s, t in zip(ds, tagged):
        s["split"] = t or hash_split(s.filepath)
        s.save()

    coco = Path(args.out) / "CarDD_COCO"
    classes = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
    for split in ("train", "val", "test"):
        view = ds.match(fo.ViewField("split") == split)
        view.export(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=str(coco / f"{split}2017"),
            labels_path=str(coco / "annotations" / f"instances_{split}2017.json"),
            label_field=args.label_field,
            classes=classes,
            tolerance=2,          # write masks as polygons (prepare_cardd.py reads polygons)
            export_media="copy",
        )
        print(f"{split}: {len(view)} images")
    print(f"Done. Next: python -m training.yolo.prepare_cardd --cardd {args.out}")


if __name__ == "__main__":
    main()
