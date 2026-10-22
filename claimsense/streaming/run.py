"""Start one Kafka worker.

    python -m claimsense.streaming.run topics         # create topics once
    python -m claimsense.streaming.run speech|documents|vision|orchestrator
"""
from __future__ import annotations

import logging
import sys

from .. import pipeline
from ..schemas import (ClaimResult, ClaimSubmission, DamageReport, DocumentText, EstimateFields, PoliceFields,
                       Transcript)
from . import kafka_io as k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def _sub(event: dict) -> ClaimSubmission:
    return ClaimSubmission.model_validate(event["submission"])


def speech(topic: str, event: dict):
    sub, res = _sub(event), ClaimResult(claim_id=event["claim_id"])
    pipeline.step_speech(sub, res, pipeline.Context())
    return k.SPEECH_DONE, {"claim_id": sub.claim_id, "transcript": res.transcript.model_dump(mode="json")}


def documents(topic: str, event: dict):
    sub, res = _sub(event), ClaimResult(claim_id=event["claim_id"])
    pipeline.step_documents(sub, res, pipeline.Context())
    return k.DOCUMENTS_DONE, {"claim_id": sub.claim_id,
                              "documents": [d.model_dump(mode="json") for d in res.documents],
                              "estimate": res.estimate.model_dump(mode="json") if res.estimate else None,
                              "police": res.police.model_dump(mode="json") if res.police else None}


def vision(topic: str, event: dict):
    sub, res = _sub(event), ClaimResult(claim_id=event["claim_id"])
    pipeline.step_vision(sub, res, pipeline.Context())
    return k.VISION_DONE, {"claim_id": sub.claim_id, "damage": res.damage.model_dump(mode="json")}


class Orchestrator:
    """Waits until speech, documents and vision results for a claim have all arrived, then
    runs the analysis steps and marks the claim ready for the adjuster.
    Partial results live in Redis when REDIS_URL is set, so the orchestrator can restart safely."""

    NEEDED = {k.SUBMITTED, k.SPEECH_DONE, k.DOCUMENTS_DONE, k.VISION_DONE}

    def __init__(self):
        import json
        import os
        self.json = json
        url = os.environ.get("REDIS_URL")
        self.r = __import__("redis").from_url(url) if url else None
        self.mem: dict[str, dict] = {}

    def _get(self, cid: str) -> dict:
        if self.r:
            return {key.decode(): self.json.loads(v) for key, v in self.r.hgetall(f"parts:{cid}").items()}
        return self.mem.setdefault(cid, {})

    def _put(self, cid: str, topic: str, event: dict) -> None:
        if self.r:
            self.r.hset(f"parts:{cid}", topic, self.json.dumps(event))
            self.r.expire(f"parts:{cid}", 3 * 24 * 3600)
        else:
            self.mem.setdefault(cid, {})[topic] = event

    def __call__(self, topic: str, event: dict):
        cid = event["claim_id"]
        self._put(cid, topic, event)
        parts = self._get(cid)
        if not self.NEEDED <= set(parts):
            return None
        sub = _sub(parts[k.SUBMITTED])
        d = parts[k.DOCUMENTS_DONE]
        res = ClaimResult(
            claim_id=cid, transcript=Transcript.model_validate(parts[k.SPEECH_DONE]["transcript"]),
            documents=[DocumentText.model_validate(x) for x in d["documents"]],
            estimate=EstimateFields.model_validate(d["estimate"]) if d["estimate"] else None,
            police=PoliceFields.model_validate(d["police"]) if d["police"] else None,
            damage=DamageReport.model_validate(parts[k.VISION_DONE]["damage"]))
        ctx = pipeline.Context.default()
        ctx.customer = pipeline.load_customer(sub.customer_id)
        for step in (pipeline.step_safety_and_facts, pipeline.step_coverage, pipeline.step_consistency,
                     pipeline.step_fraud, pipeline.step_recommend):
            step(sub, res, ctx)
        from ..storage import get_repo
        get_repo().update(cid, status="ready_for_review", result=res.model_dump(mode="json"),
                          event="analysis_complete", payload={"via": "kafka", "routes": ctx.routes})
        if self.r:
            self.r.delete(f"parts:{cid}")
        else:
            self.mem.pop(cid, None)
        return k.READY, {"claim_id": cid, "action": res.recommendation.action}


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else ""
    if role == "topics":
        k.create_topics()
    elif role == "speech":
        k.run_worker("speech-workers", [k.SUBMITTED], speech)
    elif role == "documents":
        k.run_worker("document-workers", [k.SUBMITTED], documents)
    elif role == "vision":
        k.run_worker("vision-workers", [k.SUBMITTED], vision)
    elif role == "orchestrator":
        k.run_worker("orchestrator", [k.SUBMITTED, k.SPEECH_DONE, k.DOCUMENTS_DONE, k.VISION_DONE], Orchestrator())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
