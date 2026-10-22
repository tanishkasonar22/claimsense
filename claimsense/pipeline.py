"""The claim-processing steps as plain functions.

Each step takes (submission, result, ctx) and fills part of the ClaimResult. Keeping the
steps framework-free means:
  - the LangGraph agent (agent/graph.py) wires them into a graph with a human interrupt
  - the Kafka workers (streaming/) run single steps as independent services
  - the evaluation (evals/run_gold.py) calls run_pipeline() directly, fast and deterministic
  - unit tests need no web server, no database and no API key
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from functools import wraps

from .config import get_settings
from .documents import parse_estimate, parse_police, read_document
from .facts import extract_facts
from .fraud import claim_features, score_claim
from .guardrails import injection, pii
from .guardrails.output import enforce
from .llm import LLMClient, LLMError, get_llm
from .llm.prompts import SUMMARY_SYSTEM
from .rag import assess_coverage
from .schemas import ClaimResult, ClaimSubmission, EvidenceRef, Recommendation
from .speech import transcribe
from .vision import detect


@dataclass
class Context:
    llm: LLMClient | None = None
    routes: dict[str, str] = field(default_factory=dict)   # which model answered each step
    customer: dict = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Context":
        return cls(llm=get_llm())


def timed(step):
    @wraps(step)
    def wrapper(sub, res, ctx):
        t0 = time.perf_counter()
        out = step(sub, res, ctx)
        res.timings_ms[step.__name__] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    return wrapper


# ------------------------------------------------------------------ steps
@timed
def step_speech(sub: ClaimSubmission, res: ClaimResult, ctx: Context) -> None:
    res.transcript = transcribe(sub.voice_note, sub.claim_id)


@timed
def step_documents(sub: ClaimSubmission, res: ClaimResult, ctx: Context) -> None:
    for path in sub.documents:
        doc = read_document(path)
        res.documents.append(doc)
        if doc.doc_type == "repair_estimate":
            res.estimate = parse_estimate(doc.text)
        elif doc.doc_type == "police_report":
            res.police = parse_police(doc.text)


@timed
def step_vision(sub: ClaimSubmission, res: ClaimResult, ctx: Context) -> None:
    res.damage = detect(sub.photos, sub.photo_requests)


@timed
def step_safety_and_facts(sub: ClaimSubmission, res: ClaimResult, ctx: Context) -> None:
    """Scan every piece of customer-supplied text for injection, redact PII, extract facts."""
    texts = {"transcript": res.transcript.text if res.transcript else "", "narrative": sub.narrative or ""}
    texts |= {f"document:{d.path}": d.text for d in res.documents}
    for source, text in texts.items():
        scan = injection.scan(text)
        if scan.suspected:
            res.safety.injection_suspected = True
            res.safety.injection_snippets += [f"[{source}] {s}" for s in scan.snippets][:3]

    transcript = texts["transcript"] or texts["narrative"]
    est_text = next((d.text for d in res.documents if d.doc_type == "repair_estimate"), "")
    pol_text = next((d.text for d in res.documents if d.doc_type == "police_report"), None)
    if ctx.llm is not None:
        # Nothing personal leaves for an external model.
        names = [ctx.customer.get("name", "")]
        r1, r2 = pii.redact(transcript, names), pii.redact(est_text, names)
        r3 = pii.redact(pol_text or "", names)
        res.safety.pii_redacted_count = r1.count + r2.count + r3.count
        transcript, est_text, pol_text = r1.text, r2.text, (r3.text or None)
    res.facts, ctx.routes["facts"] = extract_facts(transcript, res.estimate, est_text, res.police, pol_text, ctx.llm)


@timed
def step_coverage(sub: ClaimSubmission, res: ClaimResult, ctx: Context) -> None:
    res.coverage, _, ctx.routes["coverage"] = assess_coverage(sub.product_code, res.facts, res.damage, ctx.llm)


@timed
def step_consistency(sub: ClaimSubmission, res: ClaimResult, ctx: Context) -> None:
    """Cross-modal checks: does the story agree with the documents and the photos?"""
    f, issues = res.facts, []
    if f and f.parts_consistent is False:
        issues.append(f"Voice note describes {f.stated_damaged_parts} but the estimate lists {f.estimate_parts}.")
    if res.damage and res.damage.source != "none" and f and f.estimate_parts:
        seen = {d.part for d in res.damage.items if d.part != "unknown"}
        if seen and not seen & set(f.estimate_parts):
            issues.append(f"Photos show damage to {sorted(seen)}, none of which are on the estimate.")
    if f and f.incident_date and res.police and res.police.incident_date and f.incident_date != res.police.incident_date:
        issues.append(f"Incident date in voice note ({f.incident_date}) differs from police report ({res.police.incident_date}).")
    if sub.claim_amount and f and f.estimate_total and sub.claim_amount > 1.5 * f.estimate_total:
        issues.append(f"Claimed ${sub.claim_amount:,.0f} is {sub.claim_amount / f.estimate_total:.1f}x the repair estimate.")
    res.consistency_issues = issues


@timed
def step_fraud(sub: ClaimSubmission, res: ClaimResult, ctx: Context) -> None:
    f = res.facts
    to_date = lambda s: dt.date.fromisoformat(s) if s else None  # noqa: E731
    police_date = to_date(res.police.incident_date) if res.police else None
    report_date = to_date(res.estimate.estimate_date) if res.estimate and res.estimate.estimate_date else dt.date.today()
    feats = claim_features(
        customer_id=sub.customer_id, report_date=report_date,
        incident_date=to_date(f.incident_date) if f else None, police_date=police_date,
        claim_amount=sub.claim_amount or (f.estimate_total if f else None),
        estimate_total=f.estimate_total if f else None,
        parts_consistent=f.parts_consistent if f else None,
        repair_shop=f.repair_shop if f else None,
        police_report_filed=res.police is not None,
    )
    res.fraud = score_claim(feats)


@timed
def step_recommend(sub: ClaimSubmission, res: ClaimResult, ctx: Context) -> None:
    s = get_settings()
    cov, fraud, reasons = res.coverage, res.fraud, []
    for c in (cov.citations if cov else []):
        reasons.append(EvidenceRef(kind="policy", ref=c.section, detail=c.text[:200]))
    for issue in res.consistency_issues:
        reasons.append(EvidenceRef(kind="consistency", ref="cross-check", detail=issue))
    if fraud:
        for r in fraud.reasons:
            reasons.append(EvidenceRef(kind="fraud", ref=r.feature, detail=f"{r.explanation}: {r.value}"))

    amount = sub.claim_amount or (res.facts.estimate_total if res.facts else None)
    if res.safety.injection_suspected:
        action = "investigate"
    elif cov is None or cov.covered is None:
        action = "investigate"
    elif cov.covered is False:
        action = "deny"
    elif (fraud and fraud.score >= s.fraud_investigate_threshold) or res.consistency_issues:
        action = "investigate"
    else:
        action = "approve"
    payout = None
    if cov and cov.covered and amount is not None:
        payout = max(0.0, amount - (cov.deductible or 0))

    summary = _template_summary(action, res, amount, payout)
    if ctx.llm is not None:
        try:
            summary = ctx.llm.text(SUMMARY_SYSTEM, _summary_input(action, res, amount, payout)).strip() or summary
        except LLMError:
            pass
    rec = Recommendation(action=action, payout_estimate=payout, summary=summary, reasons=reasons)
    res.recommendation = enforce(rec, injection_suspected=res.safety.injection_suspected,
                                 claim_amount=amount, vehicle_value=ctx.customer.get("vehicle_value"))


def _template_summary(action, res: ClaimResult, amount, payout) -> str:
    f, cov = res.facts, res.coverage
    parts = ", ".join(p.replace("_", " ") for p in (f.estimate_parts if f else [])) or "unknown parts"
    s = [f"{(f.incident_type or 'unknown incident').replace('_', ' ').capitalize()} with damage to {parts}."]
    if cov:
        s.append(f"Coverage: {cov.reason}" + (f" ({', '.join(cov.sections)})." if cov.sections else "."))
    if res.fraud:
        s.append(f"Fraud score {res.fraud.score:.2f} ({res.fraud.model}).")
    if res.consistency_issues:
        s.append("Check: " + " ".join(res.consistency_issues))
    if res.safety.injection_suspected:
        s.append("Warning: claim content contains text that tries to instruct the AI; review it manually.")
    if action == "approve" and payout == 0 and cov and cov.deductible and amount is not None:
        s.append(f"The claim (${amount:,.2f}) is below the ${cov.deductible:,.2f} deductible, so nothing would be paid.")
    s.append(f"Recommendation: {action}" + (f", estimated payout ${payout:,.2f}." if action == "approve" and payout is not None else "."))
    return " ".join(s)


def _summary_input(action, res: ClaimResult, amount, payout) -> str:
    return (f"<claim>\nrecommendation: {action}\nclaimed amount: {amount}\npayout estimate: {payout}\n"
            f"facts: {res.facts.model_dump_json() if res.facts else None}\n"
            f"coverage: {res.coverage.model_dump_json() if res.coverage else None}\n"
            f"fraud score: {res.fraud.score if res.fraud else None}\n"
            f"issues: {res.consistency_issues}\ninjection suspected: {res.safety.injection_suspected}\n</claim>")


STEPS = [step_speech, step_documents, step_vision, step_safety_and_facts, step_coverage,
         step_consistency, step_fraud, step_recommend]


def load_customer(customer_id: str) -> dict:
    from .fraud.features import load_history
    return load_history().customer(customer_id)


def run_pipeline(sub: ClaimSubmission, ctx: Context | None = None) -> tuple[ClaimResult, Context]:
    """Run every step in order (no human interrupt). Used by evaluation and tests."""
    ctx = ctx or Context.default()
    if not ctx.customer:
        ctx.customer = load_customer(sub.customer_id)
    res = ClaimResult(claim_id=sub.claim_id)
    for step in STEPS:
        step(sub, res, ctx)
    return res, ctx


def submission_from_gold(g: dict) -> ClaimSubmission:
    """Build a submission from a gold-set record (data/synthetic/gold/gold_claims.jsonl)."""
    return ClaimSubmission(
        claim_id=g["claim_id"], customer_id=g["customer_id"], policy_id=g["policy_id"],
        product_code=g["product_code"], claim_amount=g["metadata"].get("claim_amount"),
        voice_note=g["inputs"]["voice_note"], documents=g["inputs"]["documents"],
        photos=g["inputs"].get("photos", []), photo_requests=g["inputs"].get("photo_requests", []),
    )
