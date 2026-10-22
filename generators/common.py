"""Shared settings and helpers for every ClaimSense data generator.

Everything here is synthetic. Nothing describes a real person, policy or claim.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from faker import Faker

ROOT = Path(__file__).resolve().parent.parent
# Same setting the services use, so data can live on Google Drive or another disk.
DATA = Path(os.environ.get("CLAIMSENSE_DATA_DIR", ROOT / "data"))
SYNTH = DATA / "synthetic"
PUBLIC = DATA / "public"


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    n_customers: int = 2_000
    n_history_claims: int = 20_000
    n_gold_candidates: int = 200
    fraud_rate: float = 0.08            # share of history claims that are fraud
    n_fraud_rings: int = 6
    biased_region: str = "R3"           # region with planted unfair denials (for the fairness audit)
    biased_denial_rate: float = 0.14    # wrongful denial rate of VALID claims in the biased region
    base_denial_rate: float = 0.04      # wrongful denial rate of VALID claims elsewhere


REGIONS = {
    "R1": "North Metro",
    "R2": "East Suburbs",
    "R3": "South Valley",
    "R4": "West Coast",
    "R5": "Central Rural",
}

# CarDD's six damage categories (see the CarDD paper). The vision model is trained on these.
DAMAGE_TYPES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]

PARTS = [
    "front_bumper", "rear_bumper", "hood", "trunk", "front_left_door", "front_right_door",
    "rear_left_door", "rear_right_door", "left_fender", "right_fender", "windshield",
    "rear_window", "left_headlamp", "right_headlamp", "left_taillamp", "right_taillamp",
    "side_mirror", "roof", "front_left_tire", "rear_right_tire",
]

# Which damage types make physical sense on which parts.
PART_DAMAGE = {
    "windshield": ["crack", "glass_shatter"],
    "rear_window": ["crack", "glass_shatter"],
    "left_headlamp": ["lamp_broken", "crack"],
    "right_headlamp": ["lamp_broken", "crack"],
    "left_taillamp": ["lamp_broken", "crack"],
    "right_taillamp": ["lamp_broken", "crack"],
    "front_left_tire": ["tire_flat"],
    "rear_right_tire": ["tire_flat"],
    "side_mirror": ["crack", "glass_shatter", "scratch"],
}
BODY_DAMAGE = ["dent", "scratch", "crack"]

INCIDENT_TYPES = {
    # incident: (likely parts, which policy coverage applies)
    "rear_end_collision": (["rear_bumper", "trunk", "left_taillamp", "right_taillamp"], "collision"),
    "front_collision": (["front_bumper", "hood", "left_headlamp", "right_headlamp", "left_fender", "right_fender"], "collision"),
    "side_swipe": (["front_left_door", "rear_left_door", "front_right_door", "rear_right_door", "side_mirror"], "collision"),
    "parking_lot_scrape": (["front_bumper", "rear_bumper", "front_left_door", "rear_right_door"], "collision"),
    "hail": (["hood", "roof", "trunk", "windshield"], "comprehensive"),
    "vandalism": (["front_left_door", "hood", "windshield", "side_mirror", "rear_window"], "comprehensive"),
    "falling_object": (["roof", "hood", "windshield"], "comprehensive"),
    "pothole": (["front_left_tire", "rear_right_tire", "front_bumper"], "collision"),
    "animal_strike": (["front_bumper", "hood", "left_headlamp", "right_headlamp"], "comprehensive"),
}

VEHICLES = [
    ("Toyota", "Camry", 27_000), ("Toyota", "RAV4", 31_000), ("Honda", "Civic", 25_000),
    ("Honda", "CR-V", 32_000), ("Ford", "F-150", 45_000), ("Ford", "Escape", 29_000),
    ("Chevrolet", "Malibu", 26_000), ("Chevrolet", "Equinox", 30_000), ("Nissan", "Altima", 26_500),
    ("Hyundai", "Elantra", 22_000), ("Kia", "Sportage", 28_000), ("Subaru", "Outback", 33_000),
    ("Tesla", "Model 3", 42_000), ("BMW", "3 Series", 46_000), ("Jeep", "Grand Cherokee", 41_000),
]

LABOR_RATE = 62.0  # USD per hour, synthetic

# Rough synthetic repair costs (parts USD, labor hours) by damage type.
REPAIR_COST = {
    "dent": (180, 3.0), "scratch": (90, 2.0), "crack": (260, 2.5),
    "glass_shatter": (420, 2.0), "lamp_broken": (310, 1.0), "tire_flat": (190, 0.7),
}
SEVERITY_MULT = {"minor": 0.7, "moderate": 1.0, "severe": 1.9}


def make_rng(settings: Settings, salt: str = "") -> random.Random:
    return random.Random(f"{settings.seed}-{salt}")


def make_faker(settings: Settings, salt: str = "") -> Faker:
    fk = Faker("en_US")
    fk.seed_instance(f"{settings.seed}-{salt}")
    return fk


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
