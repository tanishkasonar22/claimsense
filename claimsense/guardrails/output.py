"""Output guardrails: whatever the models say, the agent can only RECOMMEND.

Rules enforced here (unit-tested in tests/test_guardrails.py):
  - needs_human is always True
  - a claim flagged for possible prompt injection is never recommended for approval
  - payout estimate is never negative and never above the claim amount or vehicle value
  - summaries never state that a claim "is approved"
"""
from __future__ import annotations

import re

from ..schemas import EvidenceRef, Recommendation

APPROVED_CLAIM = re.compile(r"\b(claim\s+(is|has\s+been)\s+approved|approved\s+for\s+payment|payment\s+(is|has\s+been)\s+sent)\b", re.I)


def enforce(rec: Recommendation, *, injection_suspected: bool, claim_amount: float | None,
            vehicle_value: float | None) -> Recommendation:
    rec = rec.model_copy(deep=True)
    rec.needs_human = True
    if injection_suspected and rec.action == "approve":
        rec.action = "investigate"
        rec.reasons.append(EvidenceRef(kind="consistency", ref="guardrail",
                                       detail="Possible prompt injection in claim content; approval blocked pending review."))
    if rec.payout_estimate is not None:
        cap = min(x for x in (claim_amount, vehicle_value, rec.payout_estimate) if x is not None)
        rec.payout_estimate = round(max(0.0, cap), 2)
    if rec.action != "approve":
        rec.payout_estimate = None if rec.action == "deny" else rec.payout_estimate
    rec.summary = APPROVED_CLAIM.sub("recommended for approval (pending adjuster decision)", rec.summary)
    return rec
