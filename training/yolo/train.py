"""Train the damage segmentation model (booklet Phase 2.1). Run in Colab or Kaggle with a GPU.

    # Colab cell 1: get the code and data
    !git clone https://github.com/<you>/claimsense && cd claimsense && pip install ultralytics wandb
    from google.colab import drive; drive.mount('/content/drive')
    # Colab cell 2:
    !python -m training.yolo.train --data /content/drive/MyDrive/claimsense/cardd_yolo/car_damage.yaml \
        --project /content/drive/MyDrive/claimsense/runs --model yolo11n-seg.pt --epochs 60

Saving `--project` to Google Drive means a Colab disconnect never loses your checkpoints;
resume with --resume. Compare at least two model sizes and two augmentation settings, and
report mAP per class on the TEST split only at the very end.
Check the Ultralytics docs for current model names (yolo11n-seg.pt may have a newer version).
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/damage")
    ap.add_argument("--name", default=None)
    ap.add_argument("--aug", choices=["default", "strong"], default="default")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    name = args.name or f"{args.model.split('.')[0]}_{args.aug}"
    model = YOLO(args.model)
    aug = {} if args.aug == "default" else {"degrees": 10, "hsv_v": 0.5, "mixup": 0.1, "copy_paste": 0.2, "scale": 0.6}
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name=name, patience=15, resume=args.resume, seed=0, **aug)

    best = YOLO(f"{args.project}/{name}/weights/best.pt")
    val = best.val(data=args.data, split="val")
    report = {"model": args.model, "aug": args.aug,
              "val_mask_map50": float(val.seg.map50), "val_box_map50": float(val.box.map50),
              "per_class_mask_map50": {best.names[i]: float(m) for i, m in enumerate(val.seg.maps)}}
    print(json.dumps(report, indent=2))
    print("When you've picked your final model, run once on the test split:\n"
          f"  YOLO('{args.project}/{name}/weights/best.pt').val(data='{args.data}', split='test')")


if __name__ == "__main__":
    main()
