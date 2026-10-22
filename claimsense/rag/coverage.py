"""Coverage assessment: does this policy cover this loss, and what is the deductible?

Pipeline:
  1. Turn the claim facts into focused questions (query rewriting).
  2. Retrieve policy sections for each question, only from the customer's own policy.
  3. Answer from the retrieved sections ONLY, citing them:
       - rules baseline: reads the retrieved text with simple patterns
       - LLM: Gemini / local model with COVERAGE_SYSTEM prompt
     Both see the same retrieved sections, so a retrieval miss hurts both (as it should).

Things the evidence can't establish (intentional damage, racing, rideshare use) are left
to the fraud check and the human adjuster; the assessment says so instead of guessing.
"""
from __future__ import annotations

import re

from ..llm import LLMClient, LLMError
from ..llm.prompts import COVERAGE_SYSTEM
from ..schemas import Citation, ClaimFacts, CoverageAssessment, DamageReport
from .policy_index import PolicyChunk, get_policy_index

GLASS_PARTS = {"windshield", "rear_window", "left_headlamp", "right_headlamp", "left_taillamp", "right_taillamp"}
# How each incident type is described in the policy definitions (§1.4 collision, §1.5 comprehensive).
INCIDENT_TERMS = {
    "hail": "hail", "vandalism": "vandalism", "falling_object": "falling objects",
    "animal_strike": "animal", "pothole": "pothole",
}


def build_queries(facts: ClaimFacts, damage: DamageReport | None) -> list[str]:
    inc = (facts.incident_type or "accident").replace("_", " ")
    dtypes = sorted({d.damage_type for d in (damage.items if damage else [])})
    q = [
        f"definition of collision and other than collision loss: {inc}",
        f"is {inc} damage to your covered auto covered, deductible",
        f"collision coverage deductible for {inc}",
        f"other than collision comprehensive coverage deductible {inc}",
        f"exclusions: loss caused by {inc}",
    ]
    if any(t in ("glass_shatter", "crack", "lamp_broken") for t in dtypes):
        q.append("glass damage windshield windows lamp deductible")
    if "tire_flat" in dtypes:
        q.append("tires road hazard pothole damage to tires")
    return q


def retrieve(policy_code: str, facts: ClaimFacts, damage: DamageReport | None, k: int = 3) -> list[PolicyChunk]:
    index = get_policy_index()
    seen: dict[str, PolicyChunk] = {}
    for q in build_queries(facts, damage):
        for chunk, _ in index.search(q, policy_code, k=k):
            seen.setdefault(chunk.section, chunk)
    return sorted(seen.values(), key=lambda c: [int(x) for x in c.section[1:].split(".")])


def _money(text: str) -> float | None:
    m = re.search(r"\$([\d,]+)", text)
    return float(m.group(1).replace(",", "")) if m else None


def assess_rules(facts: ClaimFacts, damage: DamageReport | None, sections: list[PolicyChunk]) -> CoverageAssessment:
    by = {c.section: c for c in sections}
    cite = lambda *secs: [Citation(section=s, text=by[s].text) for s in secs if s in by]  # noqa: E731
    inc = facts.incident_type
    if not inc:
        return CoverageAssessment(covered=None, reason="incident type unknown; adjuster must classify the loss")
    term = INCIDENT_TERMS.get(inc)

    # Exclusion: hail (§4.5) when the policy text actually excludes it.
    if inc == "hail" and "§4.5" in by and "caused by hail" in by["§4.5"].text.lower():
        return CoverageAssessment(covered=False, sections=["§4.5"], reason="hail excluded", citations=cite("§4.5"))

    dtypes = {d.damage_type for d in (damage.items if damage else [])}
    parts = set(facts.estimate_parts or facts.stated_damaged_parts)

    # Tires only (§3.4)
    if dtypes and dtypes == {"tire_flat"} and "§3.4" in by:
        t = by["§3.4"].text.lower()
        if "is covered when caused by a road hazard" in t:
            ded = _money(by["§3.1"].text) if "§3.1" in by else None
            return CoverageAssessment(covered=True, deductible=ded, sections=["§3.4"],
                                      reason="road hazard tire damage covered", citations=cite("§3.4", "§3.1"))
        return CoverageAssessment(covered=False, sections=["§3.4", "§4.4"], reason="tire-only damage not covered",
                                  citations=cite("§3.4", "§4.4"))

    # Collision or comprehensive, decided from the policy's own definitions.
    comp_def = by.get("§1.5")
    is_comp = bool(term and term != "pothole" and comp_def and term in comp_def.text.lower())
    if not comp_def and inc in ("hail", "vandalism", "falling_object", "animal_strike"):
        is_comp = True   # definition not retrieved; fall back to the standard meaning
    sec = "§3.2" if is_comp else "§3.1"
    if sec not in by:
        return CoverageAssessment(covered=None, reason=f"{sec} not retrieved; cannot decide from the policy text")
    text = by[sec].text.lower()
    if "does not include" in text:
        kind = "comprehensive" if is_comp else "collision"
        return CoverageAssessment(covered=False, sections=[sec], reason=f"no {kind} coverage", citations=cite(sec))
    deductible = _money(by[sec].text)
    used = [sec]
    if parts and parts <= GLASS_PARTS and "§3.3" in by and "no deductible applies" in by["§3.3"].text.lower():
        deductible, used = 0.0, [sec, "§3.3"]
    return CoverageAssessment(covered=True, deductible=deductible, sections=used, reason="covered loss",
                              citations=cite(*used))


def assess_llm(llm: LLMClient, facts: ClaimFacts, damage: DamageReport | None,
               sections: list[PolicyChunk]) -> CoverageAssessment:
    ctx = "\n".join(f"{c.section} ({c.part}): {c.text}" for c in sections)
    dmg = ", ".join(f"{d.damage_type} on {d.part} ({d.severity})" for d in (damage.items if damage else [])) or "unknown"
    user = (f"<policy_sections>\n{ctx}\n</policy_sections>\n\n"
            f"<claim_facts>\nincident type: {facts.incident_type}\nparts on estimate: {facts.estimate_parts}\n"
            f"damage seen in photos: {dmg}\n</claim_facts>\n\n"
            "Is the loss covered? What deductible applies? Which sections decide it?")
    out = llm.structured(COVERAGE_SYSTEM, user, CoverageAssessment)
    by = {c.section: c for c in sections}
    # Keep only citations that were actually retrieved: the model must not invent sections.
    out.sections = [s for s in out.sections if s in by]
    out.citations = [Citation(section=s, text=by[s].text) for s in out.sections]
    return out


def assess_coverage(policy_code: str, facts: ClaimFacts, damage: DamageReport | None,
                    llm: LLMClient | None = None) -> tuple[CoverageAssessment, list[PolicyChunk], str]:
    sections = retrieve(policy_code, facts, damage)
    if llm is not None:
        try:
            return assess_llm(llm, facts, damage, sections), sections, llm.name
        except LLMError:
            return assess_rules(facts, damage, sections), sections, "rules_after_llm_error"
    return assess_rules(facts, damage, sections), sections, "rules"
