"""ClaimSense claims API (booklet Phase 1.2, study guide D5).

    uvicorn claimsense.api.main:app --reload --port 8000
    open http://localhost:8000/docs     # interactive API docs

Endpoints
  GET  /health
  POST /claims                       submit a claim (multipart: fields + files)
  POST /demo/gold/{claim_id}         submit a synthetic gold claim (no uploads needed)
  GET  /claims                       list claims (optional ?status=ready_for_review)
  GET  /claims/{claim_id}            one claim with its analysis and audit trail
  POST /claims/{claim_id}/decision   adjuster approves / denies / investigates
  POST /claims/{claim_id}/messages   message from the claimant or adjuster (guarded)
  GET  /files?path=...               serve a claim file to the dashboard
  GET  /stats                        counts for the analytics page
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import get_settings
from ..guardrails.injection import classify_request
from ..schemas import AdjusterDecision, ClaimSubmission
from ..storage import get_file_store, get_repo
from ..storage.files import FileRejected
from .service import get_service

app = FastAPI(title="ClaimSense API", version="0.1.0",
              description="Multimodal claims processing. The AI recommends; an adjuster decides.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])


def _claim_out(c) -> dict:
    return {"claim_id": c.claim_id, "status": c.status, "customer_id": c.customer_id,
            "product_code": c.product_code, "submission": c.submission, "result": c.result,
            "decision": c.decision, "error": c.error,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None}


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "mode": s.mode, "llm_provider": s.llm_provider}


@app.post("/claims", status_code=202)
async def submit_claim(
    background: BackgroundTasks,
    customer_id: str = Form(..., pattern=r"^C\d{5}$"),
    policy_id: str = Form(..., pattern=r"^P\d{5}$"),
    product_code: str = Form(..., pattern=r"^CS-[A-Z]{3,5}$"),
    claim_amount: float | None = Form(None, ge=0),
    narrative: str | None = Form(None, max_length=5000),
    voice_note: UploadFile | None = File(None),
    documents: list[UploadFile] = File(default_factory=list),
    photos: list[UploadFile] = File(default_factory=list),
):
    claim_id = f"CL{uuid.uuid4().hex[:10].upper()}"
    store = get_file_store()
    try:
        voice = store.save(claim_id, voice_note.filename, await voice_note.read(), "voice") if voice_note else None
        docs = [store.save(claim_id, f.filename, await f.read(), "document", i) for i, f in enumerate(documents)]
        pics = [store.save(claim_id, f.filename, await f.read(), "photo", i) for i, f in enumerate(photos)]
    except FileRejected as e:
        raise HTTPException(415, str(e)) from e
    sub = ClaimSubmission(claim_id=claim_id, customer_id=customer_id, policy_id=policy_id, product_code=product_code,
                          claim_amount=claim_amount, narrative=narrative, voice_note=voice, documents=docs, photos=pics)
    _dispatch(sub, background)
    return {"claim_id": claim_id, "status": "submitted"}


def _dispatch(sub: ClaimSubmission, background: BackgroundTasks) -> None:
    """PROCESSING=inline (default): run the LangGraph agent in this process.
    PROCESSING=kafka: publish the claim and let the workers in claimsense/streaming process it."""
    data = sub.model_dump(mode="json")
    get_repo().create(data)
    if os.environ.get("PROCESSING", "inline") == "kafka":
        from ..streaming import kafka_io
        kafka_io.publish(kafka_io.SUBMITTED, sub.claim_id, {"claim_id": sub.claim_id, "submission": data})
    else:
        background.add_task(get_service().process, sub.claim_id)


@app.post("/demo/gold/{gold_id}", status_code=202)
def submit_gold(gold_id: str, background: BackgroundTasks):
    from ..pipeline import submission_from_gold
    path = get_settings().synthetic_dir / "gold" / "gold_claims.jsonl"
    gold = next((json.loads(line) for line in path.open() if f'"claim_id": "{gold_id}"' in line), None)
    if gold is None:
        raise HTTPException(404, f"gold claim {gold_id} not found")
    sub = submission_from_gold(gold)
    sub.claim_id = f"{gold_id}-{uuid.uuid4().hex[:4].upper()}"
    _dispatch(sub, background)
    return {"claim_id": sub.claim_id, "status": "submitted"}


@app.get("/claims")
def list_claims(status: str | None = None, limit: int = Query(100, le=500)):
    return [_claim_out(c) for c in get_repo().list(status, limit)]


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str):
    c = get_repo().get(claim_id)
    if c is None:
        raise HTTPException(404, "claim not found")
    out = _claim_out(c)
    out["audit"] = [{"event": a.event, "actor": a.actor, "payload": a.payload, "at": a.at.isoformat()}
                    for a in get_repo().audit(claim_id)]
    return out


@app.post("/claims/{claim_id}/decision")
def decide(claim_id: str, decision: AdjusterDecision):
    if get_repo().get(claim_id) is None:
        raise HTTPException(404, "claim not found")
    try:
        return get_service().decide(claim_id, decision)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


class Message(BaseModel):
    text: str
    sender: str = "claimant"


@app.post("/claims/{claim_id}/messages")
def message(claim_id: str, msg: Message):
    """Messages are checked before anything else sees them. Blocked requests are logged."""
    blocked = classify_request(msg.text)
    get_repo().update(claim_id, event="message", actor=msg.sender,
                      payload={"text": msg.text[:500], "blocked": blocked})
    if blocked:
        return {"accepted": False, "reason": blocked,
                "reply": "I can't help with that. Your claim will be reviewed by an adjuster as normal."}
    return {"accepted": True, "reply": "Thanks, your message was added to the claim for the adjuster."}


@app.get("/files")
def get_file(path: str):
    s = get_settings()
    if path.startswith("cardd://"):
        from ..vision import resolve_photo
        p = resolve_photo(path).resolve()
    else:
        p = Path(path).resolve() if Path(path).is_absolute() else (s.synthetic_dir / path).resolve()
    allowed = [s.upload_dir.resolve(), s.synthetic_dir.resolve(), s.data_dir.resolve(), s.cardd_dir.resolve()]
    if not any(p.is_relative_to(a) for a in allowed) or not p.is_file():   # blocks ../../etc/passwd
        raise HTTPException(404, "file not found")
    return FileResponse(p)


@app.get("/stats")
def stats():
    claims = get_repo().list(limit=500)
    by_status, by_action, agreed, decided = {}, {}, 0, 0
    for c in claims:
        by_status[c.status] = by_status.get(c.status, 0) + 1
        action = ((c.result or {}).get("recommendation") or {}).get("action")
        if action:
            by_action[action] = by_action.get(action, 0) + 1
        if c.decision and action:
            decided += 1
            agreed += c.decision.get("action") == action
    return {"claims": len(claims), "by_status": by_status, "by_recommendation": by_action,
            "adjuster_agreement": round(agreed / decided, 3) if decided else None}
