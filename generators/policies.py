"""Synthetic auto insurance policy products.

Each product is written two ways from ONE spec, so they can never disagree:
  1. A numbered policy document in Markdown (what the RAG system reads).
  2. Machine-readable rules (what the gold-label generator uses to compute the correct answer).

The wording is original and only loosely follows the usual structure of US personal auto
policies (definitions, liability, damage to your auto, exclusions, duties, conditions).
"""
from __future__ import annotations

import json

from .common import SYNTH

PRODUCTS = [
    {
        "code": "CS-BASIC", "name": "ClaimSense Basic Liability",
        "collision": False, "comprehensive": False,
        "collision_deductible": None, "comprehensive_deductible": None,
        "glass_zero_deductible": False, "tire_coverage": False, "rental_per_day": 0,
        "hail_excluded": False, "rideshare_allowed": False,
    },
    {
        "code": "CS-COLL", "name": "ClaimSense Collision",
        "collision": True, "comprehensive": False,
        "collision_deductible": 1000, "comprehensive_deductible": None,
        "glass_zero_deductible": False, "tire_coverage": False, "rental_per_day": 0,
        "hail_excluded": False, "rideshare_allowed": False,
    },
    {
        "code": "CS-STD", "name": "ClaimSense Standard",
        "collision": True, "comprehensive": True,
        "collision_deductible": 500, "comprehensive_deductible": 250,
        "glass_zero_deductible": False, "tire_coverage": False, "rental_per_day": 30,
        "hail_excluded": False, "rideshare_allowed": False,
    },
    {
        "code": "CS-PLUS", "name": "ClaimSense Plus",
        "collision": True, "comprehensive": True,
        "collision_deductible": 250, "comprehensive_deductible": 100,
        "glass_zero_deductible": True, "tire_coverage": True, "rental_per_day": 45,
        "hail_excluded": False, "rideshare_allowed": True,
    },
    {
        "code": "CS-ECON", "name": "ClaimSense Economy",
        "collision": True, "comprehensive": True,
        "collision_deductible": 1500, "comprehensive_deductible": 1000,
        "glass_zero_deductible": False, "tire_coverage": False, "rental_per_day": 0,
        "hail_excluded": True, "rideshare_allowed": False,
    },
]
PRODUCT_BY_CODE = {p["code"]: p for p in PRODUCTS}


def _money(x):
    return f"${x:,.0f}"


def render_policy(p: dict) -> str:
    """Write the policy document. Section numbers are referenced by the gold labels."""
    L = []
    L.append(f"# {p['name']} Auto Policy (form {p['code']})\n")
    L.append("*Synthetic policy written for the ClaimSense portfolio project. Not a real insurance contract.*\n")

    L.append("## Part 1. Definitions\n")
    L.append("**§1.1 \"You\" and \"your\"** mean the named insured shown on the declarations page and a spouse living in the same household.\n")
    L.append("**§1.2 \"We,\" \"us\" and \"our\"** mean the company providing this insurance.\n")
    L.append("**§1.3 \"Your covered auto\"** means the vehicle listed on the declarations page.\n")
    L.append("**§1.4 \"Collision\"** means the upset of your covered auto or its impact with another vehicle or object, including a pothole.\n")
    L.append("**§1.5 \"Other than collision\" (comprehensive) loss** includes loss caused by theft, vandalism, hail, falling objects, fire, flood, windstorm, and contact with a bird or animal.\n")
    L.append("**§1.6 \"Ridesharing\"** means using your covered auto to carry people or goods for a fee through a transportation network platform.\n")

    L.append("## Part 2. Liability Coverage\n")
    L.append("**§2.1** We pay damages for bodily injury or property damage to others for which you are legally responsible because of an auto accident, up to the limits on the declarations page.\n")
    L.append("**§2.2** Liability coverage does not pay for damage to your covered auto.\n")

    L.append("## Part 3. Coverage for Damage to Your Auto\n")
    if p["collision"]:
        L.append(f"**§3.1 Collision.** We pay for direct and accidental loss to your covered auto caused by collision, minus a deductible of {_money(p['collision_deductible'])}.\n")
    else:
        L.append("**§3.1 Collision.** This policy does not include collision coverage. Loss to your covered auto caused by collision is not covered.\n")
    if p["comprehensive"]:
        L.append(f"**§3.2 Other than collision.** We pay for direct and accidental loss to your covered auto not caused by collision, minus a deductible of {_money(p['comprehensive_deductible'])}.\n")
    else:
        L.append("**§3.2 Other than collision.** This policy does not include comprehensive coverage. Loss not caused by collision is not covered.\n")
    if p["glass_zero_deductible"]:
        L.append("**§3.3 Glass.** If the only damage is to glass (windshield, windows, or lamp lenses), no deductible applies.\n")
    else:
        L.append("**§3.3 Glass.** Glass damage is paid under §3.1 or §3.2, and the deductible of that section applies.\n")
    if p["tire_coverage"]:
        L.append("**§3.4 Tires.** Damage to tires is covered when caused by a road hazard such as a pothole, minus the collision deductible.\n")
    else:
        L.append("**§3.4 Tires.** Damage to tires alone is not covered unless it happens at the same time as other covered damage to your covered auto.\n")
    if p["rental_per_day"]:
        L.append(f"**§3.5 Rental reimbursement.** While your covered auto is being repaired for a covered loss, we pay up to {_money(p['rental_per_day'])} per day for a rental vehicle, for up to 30 days.\n")
    else:
        L.append("**§3.5 Rental reimbursement.** This policy does not include rental reimbursement.\n")
    L.append("**§3.6 Limit of liability.** Our limit for loss to your covered auto is the lesser of the actual cash value of the auto or the reasonable cost to repair it with parts of like kind and quality.\n")

    L.append("## Part 4. Exclusions\n")
    L.append("We do not provide coverage under Part 3 for loss:\n")
    L.append("**§4.1** caused intentionally by you or at your direction.\n")
    L.append("**§4.2** while your covered auto is used in any race, speed contest, or timed driving event.\n")
    if p["rideshare_allowed"]:
        L.append("**§4.3** while your covered auto is used for ridesharing, **except** that this exclusion does not apply under this policy, because ridesharing use is included.\n")
    else:
        L.append("**§4.3** while your covered auto is used for ridesharing or to carry people or goods for a fee.\n")
    L.append("**§4.4** due to wear and tear, freezing, mechanical or electrical breakdown, or road damage to tires, except as provided in §3.4.\n")
    if p["hail_excluded"]:
        L.append("**§4.5** caused by hail, regardless of any other cause contributing to the loss.\n")
    else:
        L.append("**§4.5** (Reserved. This policy does not exclude hail.)\n")
    L.append("**§4.6** to any vehicle while operated by a person who is not permitted to use it.\n")

    L.append("## Part 5. Your Duties After a Loss\n")
    L.append("**§5.1** You must tell us promptly how, when and where the loss happened. Notice should be given within 30 days.\n")
    L.append("**§5.2** You must report theft and vandalism to the police.\n")
    L.append("**§5.3** You must let us inspect and photograph the damaged auto before repairs, and provide repair estimates when asked.\n")
    L.append("**§5.4** You must cooperate with our investigation and answer questions truthfully.\n")

    L.append("## Part 6. General Conditions\n")
    L.append("**§6.1 Fraud.** We do not provide coverage for anyone who has made fraudulent statements or engaged in fraudulent conduct in connection with any loss for which coverage is sought.\n")
    L.append("**§6.2 Policy period.** Coverage applies only to losses that occur during the policy period shown on the declarations page.\n")
    L.append("**§6.3 Appraisal.** If we and you do not agree on the amount of loss, either may demand an appraisal.\n")
    return "\n".join(L)


GLASS_PARTS = {"windshield", "rear_window", "left_headlamp", "right_headlamp",
               "left_taillamp", "right_taillamp"}


def coverage_decision(product_code: str, incident_type: str, coverage_kind: str,
                      damage_types: list[str], parts: list[str], rideshare_use: bool,
                      intentional: bool, racing: bool) -> dict:
    """The correct coverage answer for a claim, with the policy sections that justify it."""
    p = PRODUCT_BY_CODE[product_code]
    cited: list[str] = []

    if intentional:
        return {"covered": False, "deductible": None, "sections": ["§4.1"], "reason": "intentional damage excluded"}
    if racing:
        return {"covered": False, "deductible": None, "sections": ["§4.2"], "reason": "racing excluded"}
    if rideshare_use and not p["rideshare_allowed"]:
        return {"covered": False, "deductible": None, "sections": ["§4.3"], "reason": "ridesharing use excluded"}
    if incident_type == "hail" and p["hail_excluded"]:
        return {"covered": False, "deductible": None, "sections": ["§4.5"], "reason": "hail excluded"}

    tires_only = set(damage_types) == {"tire_flat"}
    if tires_only:
        if p["tire_coverage"]:
            return {"covered": True, "deductible": p["collision_deductible"], "sections": ["§3.4"], "reason": "road hazard tire damage covered"}
        return {"covered": False, "deductible": None, "sections": ["§3.4", "§4.4"], "reason": "tire-only damage not covered"}

    if coverage_kind == "collision":
        if not p["collision"]:
            return {"covered": False, "deductible": None, "sections": ["§3.1"], "reason": "no collision coverage"}
        deductible, cited = p["collision_deductible"], ["§3.1"]
    else:
        if not p["comprehensive"]:
            return {"covered": False, "deductible": None, "sections": ["§3.2"], "reason": "no comprehensive coverage"}
        deductible, cited = p["comprehensive_deductible"], ["§3.2"]

    glass_only = bool(parts) and set(parts) <= GLASS_PARTS
    if glass_only and p["glass_zero_deductible"]:
        deductible, cited = 0, cited + ["§3.3"]

    return {"covered": True, "deductible": deductible, "sections": cited, "reason": "covered loss"}


def generate() -> dict:
    out = SYNTH / "policies"
    out.mkdir(parents=True, exist_ok=True)
    for p in PRODUCTS:
        (out / f"{p['code']}.md").write_text(render_policy(p), encoding="utf-8")
    (out / "products.json").write_text(json.dumps(PRODUCTS, indent=2), encoding="utf-8")
    return {"policy_products": len(PRODUCTS)}
