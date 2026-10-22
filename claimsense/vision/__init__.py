"""Damage photos -> DamageReport (booklet Phase 2.1, study guide C1-C3).

Detector choice:
  VISION_MODEL_PATH set   -> your trained YOLO model (.pt, or .onnx for fast CPU inference)
  not set                 -> label fallback: uses the photo requests attached to synthetic
                             gold claims, so the pipeline runs before you train a model.
                             Marked source="label_fallback"; never report it as model accuracy.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from ..schemas import DamageItem, DamageReport
from .quality import photo_quality

# CarDD class names -> ClaimSense damage types. Check your dataset's YAML for the exact names.
CARDD_TO_DAMAGE = {
    "dent": "dent", "scratch": "scratch", "crack": "crack",
    "glass shatter": "glass_shatter", "glass_shatter": "glass_shatter",
    "lamp broken": "lamp_broken", "lamp_broken": "lamp_broken",
    "tire flat": "tire_flat", "tire_flat": "tire_flat",
}


def severity_from_area(area_share: float) -> str:
    """Share of the image covered by the damage mask/box -> severity.
    Tune these thresholds on your validation set (Phase 2.1 step 5)."""
    if area_share < 0.03:
        return "minor"
    if area_share < 0.12:
        return "moderate"
    return "severe"


@lru_cache
def _yolo(path: str):
    from ultralytics import YOLO  # pip install ultralytics
    return YOLO(path)


def resolve_photo(p: str) -> Path:
    """'cardd://CarDD_COCO/test2017/x.jpg' -> file inside CARDD_DIR; relative paths -> data_dir."""
    s = get_settings()
    if p.startswith("cardd://"):
        return s.cardd_dir / p.removeprefix("cardd://")
    return Path(p) if Path(p).is_absolute() else s.data_dir / p


def detect(photos: list[str], photo_requests: list[dict] | None = None, conf: float = 0.35) -> DamageReport:
    s = get_settings()
    usable, rejected = [], []
    # Photos live on Google Drive; opening them without a model to run only adds network time.
    if s.vision_model_path:
        for p in photos:
            path = resolve_photo(p)
            q = photo_quality(path)
            (usable if q.get("usable", False) else rejected).append(str(path))

    if s.vision_model_path and usable:
        model = _yolo(s.vision_model_path)
        items: list[DamageItem] = []
        for path in usable:
            res = model.predict(path, conf=conf, verbose=False)[0]
            h, w = res.orig_shape
            names = res.names
            for i, box in enumerate(res.boxes):
                cls_name = names[int(box.cls)]
                dtype = CARDD_TO_DAMAGE.get(cls_name.lower())
                if not dtype:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                if res.masks is not None:
                    area = float(res.masks.data[i].sum()) / float(res.masks.data[i].numel())
                else:
                    area = (x2 - x1) * (y2 - y1) / (w * h)
                # CarDD labels damage type only. Which PART is damaged needs a second model
                # (a car-parts segmenter); until you add one, the part is "unknown".
                items.append(DamageItem(part="unknown", damage_type=dtype, severity=severity_from_area(area),
                                        confidence=round(float(box.conf), 3), box=[x1, y1, x2, y2], photo=path))
        return DamageReport(items=items, source="model", rejected_photos=rejected)

    if photo_requests:
        items = [DamageItem(part=r["part"], damage_type=r["damage_type"], severity=r["severity"], confidence=1.0)
                 for r in photo_requests]
        return DamageReport(items=items, source="label_fallback", rejected_photos=rejected)
    return DamageReport(source="none", rejected_photos=rejected)
