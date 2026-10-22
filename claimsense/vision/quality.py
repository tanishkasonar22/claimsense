"""Photo quality check: reject photos too blurry or too dark to assess.

Uses only Pillow + NumPy. Blur = variance of the Laplacian (low variance = few sharp edges).
Tune the thresholds by looking at the scores of photos you consider good and bad.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BLUR_MIN = 60.0
BRIGHTNESS_MIN = 40.0
BRIGHTNESS_MAX = 230.0


def laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def photo_quality(path: str | Path, max_side: int = 800) -> dict:
    from PIL import Image
    p = Path(path)
    if not p.exists():
        return {"usable": False, "reason": "missing"}
    img = Image.open(p).convert("L")
    img.thumbnail((max_side, max_side))
    gray = np.asarray(img)
    blur = laplacian_variance(gray)
    bright = float(gray.mean())
    reasons = []
    if blur < BLUR_MIN:
        reasons.append("too blurry")
    if bright < BRIGHTNESS_MIN:
        reasons.append("too dark")
    if bright > BRIGHTNESS_MAX:
        reasons.append("overexposed")
    return {"usable": not reasons, "blur_score": round(blur, 1), "brightness": round(bright, 1),
            "reason": ", ".join(reasons) or "ok"}
