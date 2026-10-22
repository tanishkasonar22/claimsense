"""Synthetic customers, their vehicles and their policies, plus planted fraud rings."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .common import REGIONS, SYNTH, VEHICLES, Settings, make_faker, make_rng

PRODUCT_MIX = {"CS-BASIC": 0.12, "CS-COLL": 0.15, "CS-STD": 0.40, "CS-PLUS": 0.18, "CS-ECON": 0.15}
REGION_MIX = {"R1": 0.26, "R2": 0.22, "R3": 0.18, "R4": 0.20, "R5": 0.14}
AGE_BANDS = {"18-25": 0.14, "26-40": 0.34, "41-60": 0.34, "61+": 0.18}

FIRST_START = dt.date(2022, 7, 1)
LAST_START = dt.date(2026, 3, 31)


def _pick(rng, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def make_customers(s: Settings) -> pd.DataFrame:
    rng, fk = make_rng(s, "customers"), make_faker(s, "customers")
    rows = []
    span = (LAST_START - FIRST_START).days
    for i in range(s.n_customers):
        region = _pick(rng, REGION_MIX)
        make, model, base_value = rng.choice(VEHICLES)
        year = rng.randint(2012, 2026)
        value = round(base_value * max(0.25, 1 - 0.07 * (2026 - year)) * rng.uniform(0.9, 1.1), -1)
        start = FIRST_START + dt.timedelta(days=rng.randint(0, span))
        rows.append({
            "customer_id": f"C{i:05d}",
            "name": fk.name(),
            "age_band": _pick(rng, AGE_BANDS),
            "gender": rng.choice(["female", "male", "nonbinary"]) if rng.random() < 0.03 else rng.choice(["female", "male"]),
            "region": region,
            "region_name": REGIONS[region],
            "city": fk.city(),
            "address": fk.street_address(),
            "phone": fk.numerify("555-###-####"),
            "email": f"customer{i:05d}@example.com",
            "license_years": rng.randint(1, 45),
            "vehicle_make": make, "vehicle_model": model, "vehicle_year": year,
            "vehicle_value": value,
            "vin": fk.bothify("1??#?##?#?#######").upper(),
            "policy_id": f"P{i:05d}",
            "product_code": _pick(rng, PRODUCT_MIX),
            "policy_start": start,
            "fraud_ring": None,
        })
    return pd.DataFrame(rows)


def make_fraud_rings(s: Settings, customers: pd.DataFrame) -> list[dict]:
    """A ring = a group of customers who share phones, an address and a 'friendly' repair shop."""
    rng, fk = make_rng(s, "rings"), make_faker(s, "rings")
    rings = []
    ids = customers["customer_id"].tolist()
    rng.shuffle(ids)
    cursor = 0
    for r in range(s.n_fraud_rings):
        size = rng.randint(5, 12)
        members = ids[cursor:cursor + size]
        cursor += size
        rings.append({
            "ring_id": f"RING{r}",
            "members": members,
            "shared_phones": [fk.numerify("555-###-####") for _ in range(2)],
            "shared_address": fk.street_address(),
            "repair_shop": f"{fk.last_name()} {rng.choice(['Auto Body', 'Collision Center', 'Motors', 'Auto Repair'])}",
        })
        customers.loc[customers["customer_id"].isin(members), "fraud_ring"] = f"RING{r}"
    return rings


def generate(s: Settings) -> tuple[pd.DataFrame, list[dict], dict]:
    customers = make_customers(s)
    rings = make_fraud_rings(s, customers)
    out = SYNTH / "tabular"
    out.mkdir(parents=True, exist_ok=True)
    customers.to_csv(out / "customers.csv", index=False)
    pd.DataFrame([{**r, "members": ";".join(r["members"]), "shared_phones": ";".join(r["shared_phones"])} for r in rings]) \
        .to_csv(out / "fraud_rings.csv", index=False)
    return customers, rings, {"customers": len(customers), "fraud_rings": len(rings)}
