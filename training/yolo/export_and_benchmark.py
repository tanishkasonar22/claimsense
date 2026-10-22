"""Export the damage model and benchmark each format (booklet Phase 2.3, study guide C3).

    python -m training.yolo.export_and_benchmark --weights best.pt --data car_damage.yaml --formats torch onnx
    # on a Colab GPU, add TensorRT:
    python -m training.yolo.export_and_benchmark --weights best.pt --data car_damage.yaml --formats torch onnx engine-fp16 engine-int8

For each format: median latency per image, throughput, file size and mask mAP@0.5 on the
validation split. INT8 TensorRT needs calibration data (Ultralytics uses --data for it).
Fills the table in booklet Phase 2.3.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def bench(model, images: list[str], warmup: int = 5, runs: int = 50) -> dict:
    for p in images[:warmup]:
        model.predict(p, verbose=False)
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        model.predict(images[i % len(images)], verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times)
    return {"median_ms": round(med, 1), "p95_ms": round(sorted(times)[int(0.95 * (len(times) - 1))], 1),
            "images_per_s": round(1000 / med, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--formats", nargs="+", default=["torch", "onnx"])
    ap.add_argument("--images", default=None, help="folder of images to time (default: val images)")
    args = ap.parse_args()
    from ultralytics import YOLO

    img_dir = Path(args.images) if args.images else Path(args.data).parent / "images" / "val"
    images = [str(p) for p in sorted(img_dir.glob("*.jpg"))[:100]]
    results = {}
    for fmt in args.formats:
        if fmt == "torch":
            path = args.weights
        elif fmt == "onnx":
            path = YOLO(args.weights).export(format="onnx", dynamic=False, simplify=True)
        elif fmt == "engine-fp16":
            path = YOLO(args.weights).export(format="engine", half=True)
        elif fmt == "engine-int8":
            path = YOLO(args.weights).export(format="engine", int8=True, data=args.data)
        else:
            raise SystemExit(f"unknown format {fmt}")
        model = YOLO(path, task="segment")
        r = bench(model, images)
        val = model.val(data=args.data, split="val", verbose=False)
        r["mask_map50"] = round(float(val.seg.map50), 4)
        r["size_mb"] = round(Path(path).stat().st_size / 1e6, 1)
        results[fmt] = r
        print(fmt, r)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/vision_benchmark.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
