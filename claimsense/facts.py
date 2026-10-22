"""Extract ClaimFacts from the voice transcript + documents.

Two ways, same output schema:
  extract_facts_rules  deterministic baseline (regex + keyword rules)
  extract_facts_llm    Gemini / local / fine-tuned model via the LLMClient interface

extract_facts() tries the LLM when one is configured and falls back to rules if the LLM
fails, recording which route answered so evaluation can report each separately.
"""
from __future__ import annotations

import datetime as dt
import re

from .llm import LLMClient, LLMError
from .llm.prompts import EXTRACT_FACTS_SYSTEM, facts_user_prompt
from .schemas import ClaimFacts, EstimateFields, PoliceFields

PARTS = [
    "front_bumper", "rear_bumper", "hood", "trunk", "front_left_door", "front_right_door",
    "rear_left_door", "rear_right_door", "left_fender", "right_fender", "windshield",
    "rear_window", "left_headlamp", "right_headlamp", "left_taillamp", "right_taillamp",
    "side_mirror", "roof", "front_left_tire", "rear_right_tire",
]
# Longest names first so "front left door" wins over "door".
_PART_PATTERNS = sorted(((p, re.compile(r"\b" + p.replace("_", r"[\s_-]") + r"\b", re.I)) for p in PARTS),
                        key=lambda x: -len(x[0]))

# Order matters: more specific phrases first.
INCIDENT_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("hail", re.compile(r"\bhail", re.I)),
    ("pothole", re.compile(r"\bpothole", re.I)),
    ("animal_strike", re.compile(r"\b(deer|animal|elk|moose)\b", re.I)),
    ("vandalism", re.compile(r"\b(vandaliz|keyed|broken into|broke into)", re.I)),
    ("falling_object", re.compile(r"\b(branch|debris|fell|falling)\b", re.I)),
    ("side_swipe", re.compile(r"\b(changed lanes|side-?swiped|scraped along|merging)\b", re.I)),
    ("parking_lot_scrape", re.compile(r"\b(parking lot|parking|pillar|nobody left a note)\b", re.I)),
    ("rear_end_collision", re.compile(r"\b(rear-?ended|behind me|red light)\b", re.I)),
    ("front_collision", re.compile(r"\b(braked suddenly|pulled out|couldn't stop|hit them|hit their)\b", re.I)),
]
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
DATE_SAID = re.compile(r"\bon\s+(" + "|".join(MONTHS) + r")\s+(\d{1,2})\b", re.I)


def find_parts(text: str) -> list[str]:
    """Part names mentioned in the text, in order of appearance, without overlaps."""
    hits: list[tuple[int, int, str]] = []
    for part, pat in _PART_PATTERNS:          # longest names first
        for m in pat.finditer(text or ""):
            if any(m.start() < e and s < m.end() for s, e, _ in hits):
                continue
            hits.append((m.start(), m.end(), part))
    ordered: list[str] = []
    for _, _, part in sorted(hits):
        if part not in ordered:
            ordered.append(part)
    return ordered


def incident_type(text: str) -> str | None:
    for kind, pat in INCIDENT_KEYWORDS:
        if pat.search(text or ""):
            return kind
    return None


def said_date(text: str, reference: str | None) -> str | None:
    """'on December 4' -> ISO date, choosing the latest year that is not after `reference`."""
    m = DATE_SAID.search(text or "")
    if not m:
        return None
    month, day = MONTHS[m.group(1).lower()], int(m.group(2))
    ref = dt.date.fromisoformat(reference) if reference else dt.date.today()
    for year in (ref.year, ref.year - 1):
        try:
            d = dt.date(year, month, day)
        except ValueError:
            return None
        if d <= ref:
            return d.isoformat()
    return None


def extract_facts_rules(transcript: str, estimate: EstimateFields | None,
                        police: PoliceFields | None = None) -> ClaimFacts:
    estimate = estimate or EstimateFields()
    stated = find_parts(transcript)
    est_parts = [li.part for li in estimate.line_items]
    shop = None
    m = re.search(r"estimate from ([^.]+)\.", transcript or "")
    if m:
        shop = m.group(1).strip()
    police_mentioned = None
    if re.search(r"\b(didn't|did not|no)\b.{0,15}\bpolice\b", transcript or "", re.I):
        police_mentioned = False
    elif re.search(r"\bpolice\b", transcript or "", re.I):
        police_mentioned = True
    ref = estimate.estimate_date or (police.incident_date if police else None)
    return ClaimFacts(
        incident_date=said_date(transcript, ref),
        incident_type=incident_type(transcript),
        stated_damaged_parts=stated,
        estimate_parts=est_parts,
        repair_shop=shop or estimate.shop_name,
        estimate_total=estimate.total,
        police_report_mentioned=police_mentioned,
        parts_consistent=(set(stated) == set(est_parts)) if stated and est_parts else None,
    )


def extract_facts_llm(llm: LLMClient, transcript: str, estimate_text: str,
                      police_text: str | None = None) -> ClaimFacts:
    facts = llm.structured(EXTRACT_FACTS_SYSTEM, facts_user_prompt(transcript, estimate_text, police_text), ClaimFacts)
    # Recompute derived fields ourselves rather than trusting the model's arithmetic.
    if facts.stated_damaged_parts and facts.estimate_parts:
        facts.parts_consistent = set(facts.stated_damaged_parts) == set(facts.estimate_parts)
    return facts


def extract_facts(transcript: str, estimate: EstimateFields | None, estimate_text: str,
                  police: PoliceFields | None = None, police_text: str | None = None,
                  llm: LLMClient | None = None) -> tuple[ClaimFacts, str]:
    """Returns (facts, route) where route is 'rules', the LLM name, or 'rules_after_llm_error'."""
    if llm is None:
        return extract_facts_rules(transcript, estimate, police), "rules"
    try:
        return extract_facts_llm(llm, transcript, estimate_text, police_text), llm.name
    except LLMError:
        return extract_facts_rules(transcript, estimate, police), "rules_after_llm_error"
