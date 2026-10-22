"""Rule-based field extraction from document text.

This is the BASELINE. In Phase 1.3 you compare it with LLM extraction (Gemini) and in
Phase 2.5 with your fine-tuned model. Keep it: a simple baseline makes every later
improvement measurable.
"""
from __future__ import annotations

import re

from ..schemas import EstimateFields, EstimateLine, PoliceFields

MONEY = r"\$?\s*([\d,]+\.\d{2})"
LINE = re.compile(
    r"^\s*([a-z][a-z ]+?)\s+(repair|replace)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", re.I | re.M)


def _f(x: str | None) -> float | None:
    return float(x.replace(",", "")) if x else None


def _search(pattern: str, text: str, flags=re.I) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def parse_estimate(text: str) -> EstimateFields:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    shop = lines[0].strip() if lines else None
    items = [
        EstimateLine(part=m.group(1).strip().lower().replace(" ", "_"), operation=m.group(2).lower(),
                     parts_cost=_f(m.group(3)), labor_hours=_f(m.group(4)), labor_cost=_f(m.group(5)))
        for m in LINE.finditer(text)
    ]
    return EstimateFields(
        shop_name=shop,
        estimate_date=_search(r"Date:\s*(\d{4}-\d{2}-\d{2})", text),
        vin=_search(r"VIN:\s*([A-Z0-9]{17})", text),
        line_items=items,
        # Anchored and case-sensitive so "Parts subtotal:" doesn't match.
        total=_f(_search(r"^\s*TOTAL:\s*" + MONEY, text, re.M)),
    )


def parse_police(text: str) -> PoliceFields:
    narrative = None
    m = re.search(r"Officer narrative\s*(.+)$", text, re.I | re.S)
    if m:
        narrative = " ".join(m.group(1).split())
    return PoliceFields(
        report_number=_search(r"Report #:\s*(\S+)", text),
        incident_date=_search(r"Date:\s*(\d{4}-\d{2}-\d{2})", text),
        location=_search(r"Location:\s*(.+)", text),
        narrative=narrative,
    )
