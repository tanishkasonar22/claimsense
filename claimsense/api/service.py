"""Connects the API to the agent and the database.

process(claim_id)   run the graph until it pauses for the adjuster, save the result
decide(claim_id)    resume the paused graph with the adjuster's decision, save it

The graph's thread_id is the claim_id, so the checkpointer can always find a paused claim.
"""
from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache

from langgraph.types import Command

from ..agent import build_graph
from ..schemas import AdjusterDecision
from ..storage import get_file_store, get_repo
from ..tracing import langgraph_callbacks, trace_metadata

log = logging.getLogger("claimsense.service")


class AgentService:
    def __init__(self):
        self.graph = build_graph()
        self.repo = get_repo()

    def _config(self, claim_id: str) -> dict:
        return {"configurable": {"thread_id": claim_id},
                "callbacks": langgraph_callbacks(claim_id), "metadata": trace_metadata(claim_id)}

    def process(self, claim_id: str) -> None:
        claim = self.repo.get(claim_id)
        if claim is None:
            raise KeyError(claim_id)
        self.repo.update(claim_id, status="processing")
        try:
            sub = dict(claim.submission)
            store = get_file_store()
            for key in ("documents", "photos"):
                sub[key] = [str(store.open_path(p)) for p in sub.get(key, [])]
            if sub.get("voice_note"):
                sub["voice_note"] = str(store.open_path(sub["voice_note"]))
            self.graph.invoke({"submission": sub, "routes": {}}, self._config(claim_id))
            state = self.graph.get_state(self._config(claim_id))
            result = state.values.get("result")
            waiting = "human_review" in (state.next or ())
            self.repo.update(claim_id, status="ready_for_review" if waiting else "failed", result=result,
                             event="analysis_complete",
                             payload={"routes": state.values.get("routes", {}),
                                      "recommendation": (result or {}).get("recommendation", {}).get("action")})
        except Exception as e:  # keep the claim visible and explain what broke
            log.exception("processing failed for %s", claim_id)
            self.repo.update(claim_id, status="failed", error=f"{type(e).__name__}: {e}", event="processing_failed")

    def decide(self, claim_id: str, decision: AdjusterDecision) -> dict:
        state = self.graph.get_state(self._config(claim_id))
        claim = self.repo.get(claim_id)
        decision.decided_at = dt.datetime.now(dt.timezone.utc)
        payload = decision.model_dump(mode="json")
        if "human_review" in (state.next or ()):
            self.graph.invoke(Command(resume=payload), self._config(claim_id))   # resume the paused agent
            result = state.values.get("result") or {}
        elif claim is not None and claim.status == "ready_for_review":
            result = claim.result or {}                     # processed by the Kafka workers (no paused graph)
            try:
                from ..streaming import kafka_io
                kafka_io.publish(kafka_io.DECIDED, claim_id, {"claim_id": claim_id, "decision": payload})
            except Exception:  # noqa: BLE001 - Kafka is optional here; the database is the source of truth
                log.warning("could not publish decision for %s to Kafka", claim_id)
        else:
            raise ValueError("claim is not waiting for a decision")
        rec = result.get("recommendation") or {}
        self.repo.update(claim_id, status="decided", decision=payload, actor=decision.adjuster,
                         event="adjuster_decision",
                         payload={"agent": rec.get("action"), "adjuster": decision.action,
                                  "agreed": rec.get("action") == decision.action})
        if decision.edited_summary and decision.edited_summary.strip() != rec.get("summary", "").strip():
            self.repo.add_preference(claim_id, prompt=str(result.get("facts")),
                                     chosen=decision.edited_summary, rejected=rec.get("summary", ""))
        return payload


@lru_cache
def get_service() -> AgentService:
    return AgentService()
