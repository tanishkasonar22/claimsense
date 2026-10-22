"""Read-only tools the agent (and any MCP client) can call.

Rules for every tool:
  - read-only: no tool can write, update or delete anything
  - inputs are validated inside the tool (never trust the caller, including the LLM)
  - output contains no other customer's personal data
"""
from __future__ import annotations

import re

from .fraud.features import load_history
from .rag.policy_index import get_policy_index

ID = {
    "customer_id": re.compile(r"^C\d{5}$"),
    "policy_code": re.compile(r"^CS-[A-Z]{3,5}$"),
    "section": re.compile(r"^§\d+\.\d+$"),
}


class ToolInputError(ValueError):
    pass


def _check(name: str, value: str) -> str:
    if not isinstance(value, str) or not ID[name].match(value):
        raise ToolInputError(f"invalid {name}: {value!r}")
    return value


def search_policy(policy_code: str, question: str, k: int = 4) -> list[dict]:
    """Find the policy sections most relevant to a question, from one policy only."""
    _check("policy_code", policy_code)
    k = max(1, min(int(k), 8))
    hits = get_policy_index().search(question[:500], policy_code, k=k)
    return [{"section": c.section, "part": c.part, "text": c.text, "score": round(s, 3)} for c, s in hits]


def get_policy_section(policy_code: str, section: str) -> dict:
    """Return the full text of one policy section, e.g. §3.1."""
    _check("policy_code", policy_code)
    _check("section", section)
    c = get_policy_index().get(policy_code, section)
    if not c:
        raise ToolInputError(f"{section} not found in {policy_code}")
    return {"section": c.section, "part": c.part, "text": c.text}


def get_claim_history(customer_id: str) -> dict:
    """Summary of a customer's past claims (counts and outcomes, no other people's data)."""
    _check("customer_id", customer_id)
    h = load_history().history
    mine = h[h["customer_id"] == customer_id]
    return {
        "customer_id": customer_id,
        "past_claims": int(len(mine)),
        "outcomes": mine["historical_decision"].value_counts().to_dict(),
        "incident_types": mine["incident_type"].value_counts().to_dict(),
        "last_claim_date": str(mine["report_date"].max().date()) if len(mine) else None,
    }


def get_repair_shop_stats(shop_name: str) -> dict:
    """How often a repair shop appears in past claims, and how often those were confirmed fraud."""
    shop_name = shop_name.strip()[:100]
    h = load_history().history
    s = h[h["repair_shop"] == shop_name]
    return {
        "shop": shop_name,
        "past_claims": int(len(s)),
        "distinct_customers": int(s["customer_id"].nunique()),
        "confirmed_fraud_rate": round(float((s["historical_decision"] == "deny_fraud").mean()), 3) if len(s) else 0.0,
    }


TOOLS = {f.__name__: f for f in (search_policy, get_policy_section, get_claim_history, get_repair_shop_stats)}
