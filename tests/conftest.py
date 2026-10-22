import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LLM_PROVIDER", "rules")
os.environ.setdefault("FRAUD_MODEL_PATH", "")          # tests use the rules scorer, not a trained model
# Tests never read Google Drive: no vision model (so photos aren't opened) and a local, empty CarDD dir.
os.environ["VISION_MODEL_PATH"] = ""
os.environ["CARDD_DIR"] = str(ROOT / "tests" / "no_cardd")


def _gold_path() -> Path:
    from claimsense.config import get_settings
    return get_settings().synthetic_dir / "gold" / "gold_claims.jsonl"


@pytest.fixture(scope="session")
def gold() -> list[dict]:
    p = _gold_path()
    if not p.exists():
        pytest.skip("synthetic data missing: run `python generate_all.py`")
    return [json.loads(line) for line in p.open()]
