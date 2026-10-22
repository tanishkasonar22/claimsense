"""Data shapes shared by every service. Services pass these objects to each other.

Field names match the synthetic data generators in generators/, so evaluation can compare
predictions with the gold labels directly.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

IncidentType = Literal[
    "rear_end_collision", "front_collision", "side_swipe", "parking_lot_scrape",
    "hail", "vandalism", "falling_object", "pothole", "animal_strike",
]
DamageType = Literal["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
Severity = Literal["minor", "moderate", "severe"]
RecommendationAction = Literal["approve", "deny", "investigate", "request_info"]


class ClaimStatus(str, Enum):
    submitted = "submitted"
    processing = "processing"
    ready_for_review = "ready_for_review"
    decided = "decided"
    failed = "failed"


# ------------------------------------------------------------------ inputs
class ClaimSubmission(BaseModel):
    """What arrives at the API. File paths point into the file store."""
    claim_id: str
    customer_id: str
    policy_id: str
    product_code: str
    claim_amount: float | None = None
    narrative: str | None = None
    voice_note: str | None = None
    documents: list[str] = Field(default_factory=list)
    photos: list[str] = Field(default_factory=list)
    # Only used by the demo "load a gold claim" path when CarDD photos aren't downloaded yet.
    photo_requests: list[dict] = Field(default_factory=list)


# ------------------------------------------------------------------ evidence produced by services
class Transcript(BaseModel):
    text: str
    language: str | None = None
    source: Literal["whisper", "script_fallback", "none"] = "none"
    segments: list[dict] = Field(default_factory=list)


class DocumentText(BaseModel):
    path: str
    doc_type: Literal["repair_estimate", "police_report", "unknown"] = "unknown"
    text: str
    method: Literal["pdf_text_layer", "ocr", "none"] = "none"
    hidden_text_suspected: bool = False


class EstimateLine(BaseModel):
    part: str
    operation: str | None = None
    parts_cost: float | None = None
    labor_hours: float | None = None
    labor_cost: float | None = None


class EstimateFields(BaseModel):
    shop_name: str | None = None
    estimate_date: str | None = None
    vin: str | None = None
    line_items: list[EstimateLine] = Field(default_factory=list)
    total: float | None = None


class PoliceFields(BaseModel):
    report_number: str | None = None
    incident_date: str | None = None
    location: str | None = None
    narrative: str | None = None


class ClaimFacts(BaseModel):
    """Facts extracted from the voice note + documents. Same keys as the fine-tuning data."""
    incident_date: str | None = None
    incident_type: IncidentType | None = None
    stated_damaged_parts: list[str] = Field(default_factory=list)
    estimate_parts: list[str] = Field(default_factory=list)
    repair_shop: str | None = None
    estimate_total: float | None = None
    police_report_mentioned: bool | None = None
    parts_consistent: bool | None = None


class DamageItem(BaseModel):
    part: str
    damage_type: DamageType
    severity: Severity
    confidence: float = 1.0
    box: list[float] | None = None          # x1, y1, x2, y2
    photo: str | None = None


class DamageReport(BaseModel):
    items: list[DamageItem] = Field(default_factory=list)
    source: Literal["model", "label_fallback", "none"] = "none"
    rejected_photos: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    section: str          # e.g. "§3.1"
    text: str


class CoverageAssessment(BaseModel):
    covered: bool | None            # None = unclear, a human must check
    deductible: float | None = None
    sections: list[str] = Field(default_factory=list)
    reason: str = ""
    citations: list[Citation] = Field(default_factory=list)


class FraudReason(BaseModel):
    feature: str
    value: float | str | bool | None
    contribution: float
    explanation: str


class FraudAssessment(BaseModel):
    score: float                    # 0 to 1
    reasons: list[FraudReason] = Field(default_factory=list)
    model: str = "rules"
    similar_claims: list[dict] = Field(default_factory=list)


class SafetyFlags(BaseModel):
    injection_suspected: bool = False
    injection_snippets: list[str] = Field(default_factory=list)
    pii_redacted_count: int = 0


class EvidenceRef(BaseModel):
    kind: Literal["policy", "transcript", "document", "photo", "fraud", "consistency"]
    ref: str
    detail: str


class Recommendation(BaseModel):
    action: RecommendationAction
    payout_estimate: float | None = None
    summary: str
    reasons: list[EvidenceRef] = Field(default_factory=list)
    needs_human: bool = True        # always true: only a human adjuster makes the final decision


class AdjusterDecision(BaseModel):
    action: Literal["approve", "deny", "investigate", "request_info"]
    adjuster: str = "adjuster"
    note: str = ""
    edited_summary: str | None = None   # becomes DPO training data (chosen) vs the agent's summary (rejected)
    decided_at: datetime | None = None


class ClaimResult(BaseModel):
    """Everything the pipeline learned about one claim."""
    claim_id: str
    transcript: Transcript | None = None
    documents: list[DocumentText] = Field(default_factory=list)
    estimate: EstimateFields | None = None
    police: PoliceFields | None = None
    facts: ClaimFacts | None = None
    damage: DamageReport | None = None
    coverage: CoverageAssessment | None = None
    fraud: FraudAssessment | None = None
    safety: SafetyFlags = Field(default_factory=SafetyFlags)
    consistency_issues: list[str] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
