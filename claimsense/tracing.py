"""Optional Langfuse tracing (booklet Phase 1.6, study guide F5).

Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST (your self-hosted Langfuse),
then pip install langfuse. Every graph node and LLM call then shows up as a trace, grouped by claim.
Without those settings, tracing is simply off.
"""
from __future__ import annotations

import os


def langgraph_callbacks(claim_id: str) -> list:
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse.langchain import CallbackHandler   # langfuse v3
    except ImportError:
        return []
    return [CallbackHandler()]


def trace_metadata(claim_id: str) -> dict:
    # Langfuse reads these keys from the LangGraph config metadata to group traces.
    return {"langfuse_session_id": claim_id, "langfuse_tags": ["claimsense"]}
