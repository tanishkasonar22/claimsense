"""Process a claims backlog in parallel with Ray (booklet Phase 3.3, study guide E6).

Scenario: a hailstorm creates thousands of claims overnight.

    python -m training.batch.ray_backlog --mode pipeline --copies 25     # 200 gold claims x 25 = 5,000 claims
    python -m training.batch.ray_backlog --mode vision --images data/public/cardd_yolo/images/test

Compares a single-process loop with Ray across all cores and prints claims (or images) per second.
Needs: pip install "ray[data]"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def gold_submissions(copies: int) -> list[dict]:
    from claimsense.config import get_settings
    path = get_settings().synthetic_dir / "gold" / "gold_claims.jsonl"
    rows = [json.loads(line) for line in path.open()]
    return [r | {"claim_id": f"{r['claim_id']}-{i}"} for i in range(copies) for r in rows]


def process_one(g: dict) -> str:
    from claimsense.pipeline import Context, run_pipeline, submission_from_gold
    res, _ = run_pipeline(submission_from_gold(g), Context(llm=None))
    return res.recommendation.action


def run_pipeline_mode(copies: int, limit_single: int) -> None:
    import ray
    claims = gold_submissions(copies)
    t0 = time.perf_counter()
    for g in claims[:limit_single]:
        process_one(g)
    single = limit_single / (time.perf_counter() - t0)
    print(f"single process: {single:.1f} claims/s (measured on {limit_single})")

    ray.init(ignore_reinit_error=True)
    remote = ray.remote(process_one)
    t0 = time.perf_counter()
    actions = ray.get([remote.remote(g) for g in claims])
    rate = len(claims) / (time.perf_counter() - t0)
    print(f"ray ({ray.available_resources().get('CPU', 0):.0f} CPUs): {rate:.1f} claims/s for {len(claims):,} claims "
          f"-> {rate / single:.1f}x speedup")
    print({a: actions.count(a) for a in set(actions)})


class YoloActor:
    """Loads the model ONCE per worker, then processes batches (the reason to use an actor)."""

    def __init__(self, weights: str):
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def __call__(self, batch: dict) -> dict:
        res = self.model.predict(list(batch["path"]), verbose=False)
        batch["detections"] = [len(r.boxes) for r in res]
        return batch


def run_vision_mode(images: str, weights: str, workers: int) -> None:
    import ray
    paths = [str(p) for p in sorted(Path(images).glob("*.jpg"))]
    ray.init(ignore_reinit_error=True)
    ds = ray.data.from_items([{"path": p} for p in paths])
    t0 = time.perf_counter()
    out = ds.map_batches(YoloActor, fn_constructor_args=(weights,), batch_size=16, concurrency=workers).take_all()
    print(f"ray data: {len(out) / (time.perf_counter() - t0):.1f} images/s with {workers} model replicas")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pipeline", "vision"], default="pipeline")
    ap.add_argument("--copies", type=int, default=25)
    ap.add_argument("--images", default="data/public/cardd_yolo/images/test")
    ap.add_argument("--weights", default="models/damage.onnx")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "pipeline":
        run_pipeline_mode(args.copies, limit_single=200)
    else:
        run_vision_mode(args.images, args.weights, args.workers)


if __name__ == "__main__":
    main()
