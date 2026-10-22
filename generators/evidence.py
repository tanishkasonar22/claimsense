"""Evidence for gold candidate claims: voice-note scripts, photo requests and the gold labels.

Voice notes: we write the SCRIPT (exact words). You then turn each script into audio by
recording it yourself or with a text-to-speech model (see README). Because the exact words
are known, you can measure Whisper's word error rate (WER) directly.

Photos: we write a PHOTO REQUEST per damaged part (part + damage type + severity).
After you download CarDD, scripts/assemble_gold_with_cardd.py picks matching real images.
"""
from __future__ import annotations

import datetime as dt
import json
import random

from .common import SYNTH, Settings, make_faker, make_rng, write_jsonl
from .claims import STREETS, human

FILLERS = ["um,", "uh,", "so", "like,", "you know,", "basically"]
OPENERS = [
    "Hi, I'm calling about a claim.", "Hello, this is about my car.", "Hi there.",
    "Okay, so, I need to report some damage.", "Hey, I want to file a claim.",
]
TIME_WORDS = ["in the morning", "around lunchtime", "in the afternoon", "in the evening", "late at night"]


def _say_date(d: dt.date) -> str:
    return d.strftime("%B ") + str(d.day)


def voice_script(c: dict, rng: random.Random) -> tuple[str, dict]:
    incident_date = dt.date.fromisoformat(str(c["incident_date"]))
    said_date = incident_date
    contradictions = []
    # Fraudulent story mismatches sometimes also include a date that doesn't match the police report.
    if "story_mismatch" in c["fraud_patterns"] and c["police_report_filed"] and rng.random() < 0.5:
        said_date = incident_date - dt.timedelta(days=rng.choice([2, 3, 5]))
        contradictions.append("date_differs_from_police_report")
    if c["stated_damaged_parts"] != c["true_damaged_parts"]:
        contradictions.append("stated_parts_differ_from_damage")

    parts_spoken = human(c["stated_damaged_parts"].split(";"))
    bits = [rng.choice(OPENERS),
            f"My name is {c['customer_name']} and my policy number is {c['policy_id']}.",
            f"So {rng.choice(FILLERS)} on {_say_date(said_date)} {rng.choice(TIME_WORDS)},",
            c["narrative"][0].lower() + c["narrative"][1:]]
    if c["police_report_filed"]:
        bits.append("I did call the police and they came out and made a report.")
    else:
        bits.append(f"I didn't call the police, {rng.choice(FILLERS)} it didn't seem necessary.")
    if c["incident_type"] in ("rear_end_collision", "front_collision", "side_swipe"):
        bits.append("I got the other driver's information.")
    bits.append(f"The {parts_spoken} {'is' if ',' not in parts_spoken and ' and ' not in parts_spoken else 'are'} damaged and I already got an estimate from {c['repair_shop']}.")
    if "inflated_amount" in c["fraud_patterns"]:
        bits.append(f"It's going to be around {round(c['claim_amount'], -2):,.0f} dollars, which seems about right to me.")
    bits.append(rng.choice(["Please let me know what you need from me.", "Thanks.", "Can you call me back? Thank you."]))
    script = " ".join(bits)

    facts = {
        "incident_date_said": said_date.isoformat(),
        "incident_type": c["incident_type"],
        "stated_damaged_parts": c["stated_damaged_parts"].split(";"),
        "police_report_mentioned": c["police_report_filed"],
        "repair_shop": c["repair_shop"],
        "contradictions": contradictions,
    }
    return script, facts


def officer_narrative(c: dict, rng: random.Random) -> str:
    parts = human(c["true_damaged_parts"].split(";"))
    kind = c["incident_type"].replace("_", " ")
    return (f"Officer responded to a reported {kind} at the listed location. "
            f"Observed damage to the insured vehicle's {parts}. "
            f"{'Both drivers exchanged information. ' if c['incident_type'] in ('rear_end_collision', 'front_collision', 'side_swipe') else ''}"
            f"No injuries reported. {rng.choice(['Photos taken at scene.', 'Statements recorded.', 'Vehicle was drivable.'])}")


def enrich(s: Settings, gold: list[dict], customers_by_id: dict) -> None:
    """Add the fields the document generator needs (call before documents.generate)."""
    rng = make_rng(s, "enrich")
    for c in gold:
        cust = customers_by_id[c["customer_id"]]
        c["customer_name"] = cust["name"]
        c["narrative_location"] = f"{rng.randint(100, 9800)} {rng.choice(STREETS)}, {cust['city']}"
        c["officer_narrative"] = officer_narrative(c, rng)


def generate(s: Settings, gold: list[dict]) -> dict:
    rng = make_rng(s, "voice")
    out = SYNTH / "voice"
    out.mkdir(parents=True, exist_ok=True)
    voice_rows, gold_rows = [], []
    for c in gold:
        script, facts = voice_script(c, rng)
        style = rng.choices(["calm", "fast", "noisy_background", "accented_tts"], weights=[0.45, 0.2, 0.2, 0.15])[0]
        voice_rows.append({"claim_id": c["claim_id"], "audio_file": f"voice/{c['claim_id']}.wav",
                           "script": script, "suggested_style": style, "word_count": len(script.split())})
        photo_requests = [{"part": d["part"], "damage_type": d["damage_type"], "severity": d["severity"]}
                          for d in c["damage"]]
        gold_rows.append({
            "claim_id": c["claim_id"],
            "customer_id": c["customer_id"], "policy_id": c["policy_id"], "product_code": c["product_code"],
            "inputs": {
                "voice_note": f"voice/{c['claim_id']}.wav",
                "documents": c.get("documents", []),
                "photos": [],                       # filled by scripts/assemble_gold_with_cardd.py
                "photo_requests": photo_requests,
            },
            "expected": {
                "voice_facts": facts,
                "damage": c["damage"],
                "coverage": {"covered": c["correct_covered"], "deductible": c["correct_deductible"],
                             "sections": c["correct_sections"].split(";"), "reason": c["coverage_reason"]},
                "is_fraud": c["is_fraud"],
                "fraud_patterns": [p for p in c["fraud_patterns"].split(";") if p],
                "decision": c["correct_decision"],
            },
            "metadata": {"incident_date": str(c["incident_date"]), "claim_amount": c["claim_amount"],
                         "estimate_total": c["estimate_total"], "region": c["region"],
                         "age_band": c["age_band"], "gender": c["gender"]},
            "hand_checked": False,                  # flip to true as you verify each claim by hand
        })
    write_jsonl(out / "voice_scripts.jsonl", voice_rows)
    write_jsonl(SYNTH / "gold" / "gold_claims.jsonl", gold_rows)
    (SYNTH / "gold" / "README.md").write_text(
        "# Gold set\n\nEach line of `gold_claims.jsonl` is one complete claim with its correct answers.\n"
        "Set `hand_checked` to true after you verify a claim yourself. Aim for 150 verified claims.\n"
        "Never use these claims for training or fine-tuning.\n")
    return {"voice_scripts": len(voice_rows), "gold_claims": len(gold_rows)}
