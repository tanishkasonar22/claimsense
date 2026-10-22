"""Synthetic claims: the 20,000-claim history and the richer 'gold candidate' claims.

Every claim carries ground-truth labels:
  - is_fraud / fraud_patterns : the truth (fraud model target)
  - correct_coverage          : what the policy actually says (from policies.coverage_decision)
  - correct_decision          : what SHOULD happen (approve / deny / investigate)
  - historical_decision       : what the (synthetic) human adjusters actually decided.
                                This contains a planted regional bias so the fairness
                                audit in Phase 3.4 has something real to find.
"""
from __future__ import annotations

import datetime as dt
import random

import pandas as pd

from .common import (BODY_DAMAGE, INCIDENT_TYPES, LABOR_RATE, PART_DAMAGE, REPAIR_COST,
                     SEVERITY_MULT, SYNTH, Settings, make_faker, make_rng)
from .policies import coverage_decision

HISTORY_END = dt.date(2026, 6, 30)
FRAUD_PATTERNS = ["ring", "early_claim", "inflated_amount", "story_mismatch", "repeat_claimant"]

STREETS = ["Main St", "Oak Ave", "5th St", "Elm Dr", "Highway 9", "Market St", "Pine Rd",
           "Lakeview Blvd", "Cedar Ln", "Station Rd"]

NARRATIVES = {
    "rear_end_collision": [
        "I was stopped at a red light on {street} when the car behind me didn't stop and hit my {parts}.",
        "Traffic slowed down on {street} and another driver rear-ended me. The {parts} got damaged.",
    ],
    "front_collision": [
        "The car in front of me braked suddenly on {street} and I couldn't stop in time. My {parts} hit their bumper.",
        "Someone pulled out of a driveway on {street} and I hit them. The {parts} are damaged.",
    ],
    "side_swipe": [
        "A truck changed lanes into me on {street} and scraped along my {parts}.",
        "While merging on {street}, another car side-swiped me and damaged the {parts}.",
    ],
    "parking_lot_scrape": [
        "I came back to my car in the grocery store parking lot and found damage on the {parts}. Nobody left a note.",
        "While parking at work I misjudged a concrete pillar and scraped the {parts}.",
    ],
    "hail": [
        "A hailstorm came through overnight and my car was parked outside. There are dents on the {parts}.",
        "I got caught in a hailstorm driving on {street}. The {parts} got hit hard.",
    ],
    "vandalism": [
        "Someone vandalized my car while it was parked on {street} overnight. The {parts} are damaged.",
        "I found my car keyed and broken into outside my apartment. Damage to the {parts}.",
    ],
    "falling_object": [
        "A tree branch fell on my car during a windstorm on {street}. It damaged the {parts}.",
        "Debris fell off a construction site on {street} onto my {parts}.",
    ],
    "pothole": [
        "I hit a deep pothole on {street} and heard a bang. The {parts} got damaged.",
        "There was a big pothole on {street} that I couldn't avoid. Now the {parts} is damaged.",
    ],
    "animal_strike": [
        "A deer ran out in front of me on {street} at night and I hit it. The {parts} are damaged.",
        "I hit a deer on {street}. It damaged my {parts}.",
    ],
}


def human(parts: list[str]) -> str:
    names = [p.replace("_", " ") for p in parts]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def damage_for(part: str, rng: random.Random, incident: str) -> str:
    if incident == "hail" and part in ("hood", "roof", "trunk"):
        return "dent"
    options = PART_DAMAGE.get(part, BODY_DAMAGE)
    return rng.choice(options)


def estimate_lines(damage: list[dict], rng: random.Random) -> list[dict]:
    lines = []
    for d in damage:
        parts_cost, hours = REPAIR_COST[d["damage_type"]]
        mult = SEVERITY_MULT[d["severity"]]
        op = "replace" if d["severity"] == "severe" or d["damage_type"] in ("glass_shatter", "lamp_broken", "tire_flat") else "repair"
        p_cost = round(parts_cost * mult * rng.uniform(0.85, 1.2) * (1.6 if op == "replace" else 0.5), 2)
        h = round(hours * mult * rng.uniform(0.8, 1.25), 1)
        lines.append({"part": d["part"], "operation": op, "damage_type": d["damage_type"],
                      "parts_cost": p_cost, "labor_hours": h, "labor_cost": round(h * LABOR_RATE, 2)})
    return lines


def build_claim(i: int, cust: dict, rings_by_id: dict, s: Settings, rng: random.Random, fk,
                prior_claims: int, force_fraud: bool | None = None, id_prefix: str = "H") -> dict:
    start: dt.date = cust["policy_start"]
    ring = rings_by_id.get(cust["fraud_ring"]) if cust["fraud_ring"] else None

    # --- is this claim fraudulent, and how? ---
    if force_fraud is not None:
        is_fraud = force_fraud
    else:
        base = s.fraud_rate * (5.0 if ring else 0.85)
        is_fraud = rng.random() < base
    patterns: list[str] = []
    if is_fraud:
        candidates = ["early_claim", "inflated_amount", "story_mismatch"]
        if ring:
            candidates.append("ring")
        if prior_claims >= 2:
            candidates.append("repeat_claimant")
        k = rng.choice([1, 1, 2, 2, 3])
        patterns = rng.sample(candidates, min(k, len(candidates)))
        if ring and "ring" not in patterns and rng.random() < 0.7:
            patterns.append("ring")

    # --- incident date ---
    latest = (HISTORY_END - start).days
    if latest < 5:
        latest = 5
    if "early_claim" in patterns:
        days = rng.randint(2, min(25, latest))
    else:
        days = rng.randint(min(15, latest), latest)
        if not is_fraud and rng.random() < 0.04:       # some honest early claims, so the signal is noisy
            days = rng.randint(2, min(25, latest))
    incident_date = start + dt.timedelta(days=days)
    report_delay = rng.choice([0, 0, 1, 1, 2, 3, 5, 8, 14]) if not is_fraud else rng.choice([0, 1, 3, 9, 21, 35])

    # --- what happened, and what got damaged ---
    incident = rng.choice(list(INCIDENT_TYPES))
    likely_parts, coverage_kind = INCIDENT_TYPES[incident]
    n_parts = rng.choice([1, 1, 2, 2, 3])
    true_parts = rng.sample(likely_parts, min(n_parts, len(likely_parts)))
    severity = rng.choices(["minor", "moderate", "severe"], weights=[0.45, 0.4, 0.15])[0]
    damage = [{"part": p, "damage_type": damage_for(p, rng, incident), "severity": severity} for p in true_parts]

    # What the customer SAYS was damaged. For story_mismatch fraud, the narrative names
    # different parts than the photos / estimate show (a cross-modal inconsistency).
    stated_parts = list(true_parts)
    if "story_mismatch" in patterns:
        other = [p for parts, _ in INCIDENT_TYPES.values() for p in parts if p not in true_parts]
        stated_parts = rng.sample(sorted(set(other)), min(len(true_parts), 2))

    narrative = rng.choice(NARRATIVES[incident]).format(street=rng.choice(STREETS), parts=human(stated_parts))

    # --- money ---
    lines = estimate_lines(damage, rng)
    estimate_total = round(sum(l["parts_cost"] + l["labor_cost"] for l in lines) * 1.0825, 2)
    claim_amount = estimate_total
    if "inflated_amount" in patterns:
        claim_amount = round(estimate_total * rng.uniform(1.7, 3.2), 2)
    elif rng.random() < 0.15:
        claim_amount = round(estimate_total * rng.uniform(1.0, 1.25), 2)  # honest rounding up / extras

    # --- who and where ---
    phone = rng.choice(ring["shared_phones"]) if ring and "ring" in patterns else cust["phone"]
    if ring and "ring" in patterns:
        shop = ring["repair_shop"]
    else:
        shop = f"{fk.last_name()} {rng.choice(['Auto Body', 'Collision', 'Car Care', 'Auto Repair', 'Body Shop'])}"

    rideshare = rng.random() < 0.02
    racing = rng.random() < 0.003
    intentional = is_fraud and rng.random() < 0.08
    police_report = incident == "vandalism" or (coverage_kind == "collision" and incident != "parking_lot_scrape" and rng.random() < 0.6)

    cov = coverage_decision(cust["product_code"], incident, coverage_kind,
                            [d["damage_type"] for d in damage], true_parts, rideshare, intentional, racing)

    # --- correct decision vs. what historical adjusters did ---
    if not cov["covered"]:
        correct = "deny"
    elif is_fraud:
        correct = "investigate"
    else:
        correct = "approve"

    hist = correct
    if correct == "investigate":
        hist = rng.choices(["deny_fraud", "investigate", "approve"], weights=[0.45, 0.25, 0.30])[0]
    elif correct == "approve":
        wrongful = s.biased_denial_rate if cust["region"] == s.biased_region else s.base_denial_rate
        if rng.random() < wrongful:
            hist = "deny"

    deductible = cov["deductible"] or 0
    payout = 0.0
    if hist == "approve":
        payout = round(max(0.0, min(claim_amount, cust["vehicle_value"]) - deductible), 2)

    return {
        "claim_id": f"{id_prefix}{i:06d}",
        "customer_id": cust["customer_id"], "policy_id": cust["policy_id"],
        "product_code": cust["product_code"], "region": cust["region"],
        "age_band": cust["age_band"], "gender": cust["gender"],
        "policy_start": start, "incident_date": incident_date,
        "report_date": incident_date + dt.timedelta(days=report_delay),
        "days_since_policy_start": days, "report_delay_days": report_delay,
        "incident_type": incident, "coverage_kind": coverage_kind,
        "true_damaged_parts": ";".join(true_parts),
        "stated_damaged_parts": ";".join(stated_parts),
        "damage_types": ";".join(d["damage_type"] for d in damage),
        "severity": severity,
        "damage": damage,
        "estimate_lines": lines,
        "vehicle_make": cust["vehicle_make"], "vehicle_model": cust["vehicle_model"],
        "vehicle_year": cust["vehicle_year"], "vehicle_value": cust["vehicle_value"], "vin": cust["vin"],
        "estimate_total": estimate_total, "claim_amount": claim_amount,
        "repair_shop": shop, "contact_phone": phone, "address": cust["address"], "city": cust["city"],
        "police_report_filed": police_report, "rideshare_use": rideshare, "racing": racing,
        "prior_claims_count": prior_claims,
        "narrative": narrative,
        "is_fraud": is_fraud, "fraud_patterns": ";".join(sorted(patterns)),
        "correct_covered": cov["covered"], "correct_deductible": cov["deductible"],
        "correct_sections": ";".join(cov["sections"]), "coverage_reason": cov["reason"],
        "correct_decision": correct, "historical_decision": hist, "historical_payout": payout,
    }


def generate_history(s: Settings, customers: pd.DataFrame, rings: list[dict]) -> tuple[pd.DataFrame, dict]:
    rng, fk = make_rng(s, "history"), make_faker(s, "history")
    rings_by_id = {r["ring_id"]: r for r in rings}
    custs = customers.to_dict("records")
    weights = [3.0 if c["fraud_ring"] else (1.6 if c["license_years"] < 4 else 1.0) for c in custs]
    counts: dict[str, int] = {}
    rows = []
    for i in range(s.n_history_claims):
        c = rng.choices(custs, weights=weights)[0]
        prior = counts.get(c["customer_id"], 0)
        rows.append(build_claim(i, c, rings_by_id, s, rng, fk, prior))
        counts[c["customer_id"]] = prior + 1
    df = pd.DataFrame(rows).sort_values("incident_date").reset_index(drop=True)

    out = SYNTH / "tabular"
    flat = df.drop(columns=["damage", "estimate_lines"])
    flat.to_csv(out / "claims_history.csv", index=False)
    flat.to_parquet(out / "claims_history.parquet", index=False)
    df[["claim_id", "damage", "estimate_lines"]].to_json(out / "claims_history_details.jsonl", orient="records", lines=True)
    stats = {
        "history_claims": len(df),
        "fraud_rate": round(df["is_fraud"].mean(), 4),
        "covered_rate": round(df["correct_covered"].mean(), 4),
    }
    return df, stats


def generate_gold_candidates(s: Settings, customers: pd.DataFrame, rings: list[dict]) -> list[dict]:
    """New (not-yet-decided) claims that get full evidence: documents, voice scripts, photos.

    ~25% fraud so the evaluation has enough positive cases. These IDs must never appear
    in fine-tuning data.
    """
    rng, fk = make_rng(s, "gold"), make_faker(s, "gold")
    rings_by_id = {r["ring_id"]: r for r in rings}
    custs = customers.to_dict("records")
    claims = []
    for i in range(s.n_gold_candidates):
        fraud = rng.random() < 0.25
        pool = [c for c in custs if c["fraud_ring"]] if fraud and rng.random() < 0.5 else custs
        c = rng.choice(pool)
        claims.append(build_claim(i, c, rings_by_id, s, rng, fk, prior_claims=rng.choice([0, 0, 1, 2, 3]),
                                  force_fraud=fraud, id_prefix="G"))
    return claims
