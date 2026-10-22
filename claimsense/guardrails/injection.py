"""Prompt-injection detection for claim content (narratives, transcripts, documents)
and for messages typed by users.

Two layers:
  1. Rules: phrases that address an AI or try to change processing. Fast, explainable,
     and the baseline you report in the red-team results.
  2. Optional LLM classifier (classify_with_llm) for what rules miss. Measure both.

Detection does not "fix" an attack. The pipeline still treats all claim content as data;
detection adds a flag the adjuster sees and is recorded in the audit log.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel

# Each rule: (category, regex). Keep them readable; you will explain them in interviews.
RULES: list[tuple[str, re.Pattern]] = [
    ("override", re.compile(r"\b(system\s+override|new\s+instruction|admin(istrator)?\s*:|developer\s+mode)", re.I)),
    ("ignore_previous", re.compile(r"\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)", re.I)),
    ("address_ai", re.compile(r"\b(note\s+to\s+(the\s+)?ai|ai\s+reviewer|assistant,|dear\s+ai|language\s+model)", re.I)),
    ("force_decision", re.compile(r"\b(recommend|mark|set|reply\s+only\s+with)\b.{0,40}\b(approve[d]?|low\s+risk|fraud\s+score\s+(to\s+)?0)\b", re.I)),
    ("skip_checks", re.compile(r"\b(skip|bypass|turn\s+off|disable)\b.{0,30}\b(check|review|investigation|verification|fraud)", re.I)),
    ("pre_approved", re.compile(r"\b(pre-?approved|already\s+(verified|approved))\b", re.I)),
    ("fake_authority", re.compile(r"\bthis\s+message\s+is\s+from\s+(compliance|management|the\s+claims\s+director)", re.I)),
    ("tool_abuse", re.compile(r"\b(call|run|execute)\s+\w*\s*(run_query|tool|function)\b|\b(DELETE\s+FROM|DROP\s+TABLE|UPDATE\s+\w+\s+SET)\b", re.I)),
    ("tag_escape", re.compile(r"</\s*(document|voice_transcript|repair_estimate_ocr|system)\s*>", re.I)),
    ("omit_facts", re.compile(r"\b(omit|do\s+not\s+mention|don't\s+mention)\b.{0,30}\b(deductible|exclusion|fraud)", re.I)),
]


@dataclass
class ScanResult:
    suspected: bool
    hits: list[tuple[str, str]] = field(default_factory=list)   # (category, matched snippet)

    @property
    def snippets(self) -> list[str]:
        return [s for _, s in self.hits]


def scan(text: str) -> ScanResult:
    hits = []
    for cat, pat in RULES:
        for m in pat.finditer(text or ""):
            start, end = max(0, m.start() - 20), min(len(text), m.end() + 20)
            hits.append((cat, text[start:end].strip()))
    return ScanResult(bool(hits), hits)


class _InjectionVerdict(BaseModel):
    is_injection: bool
    reason: str = ""


def classify_with_llm(text: str, llm) -> bool:
    """Second opinion from an LLM. Use it on text the rules did NOT flag."""
    from ..llm.prompts import INJECTION_CLASSIFIER_SYSTEM
    v = llm.structured(INJECTION_CLASSIFIER_SYSTEM, f"<text>\n{text}\n</text>", _InjectionVerdict)
    return v.is_injection


# ---------------------------------------------------------------- user messages (dashboard / API)
REQUEST_RULES: list[tuple[str, re.Pattern]] = [
    ("destructive_tool_use", re.compile(r"\b(delete|drop|update|change|reset|remove)\b.{0,40}\b(claims?|database|table|payout|status|flags?)\b", re.I)),
    ("pii_extraction", re.compile(r"\b(phone\s+number|address(es)?|emails?|vins?|policy\s+numbers?|list\s+every\s+customer|give\s+me\s+names)\b", re.I)),
    ("system_prompt_leak", re.compile(r"\b(system\s+prompt|your\s+(rules|instructions)|threshold|tools?\s+you\s+have|quote\s+them)\b", re.I)),
    ("decision_bypass", re.compile(r"\b(don'?t\s+send\s+this\s+to\s+a\s+human|approve\s+it\s+yourself|finalize|mark\s+this\s+as\s+reviewed|auto-?approve|turn\s+off)\b", re.I)),
    ("social_engineering", re.compile(r"\b(legally\s+required|i'?m\s+a\s+\w*\s*developer|five-star|already\s+approved|just\s+approve|skip\s+the\s+checks)\b", re.I)),
    ("data_poisoning", re.compile(r"\b(remember\s+that|save\s+this\s+as\s+a\s+rule|update\s+your\s+knowledge|from\s+now\s+on)\b", re.I)),
    ("off_topic", re.compile(r"\b(poem|stocks?|college\s+essay|fake\s+police\s+report|look\s+bigger)\b", re.I)),
]


def classify_request(text: str) -> str | None:
    """Category of a disallowed user request, or None if it looks like a normal claim message."""
    for cat, pat in REQUEST_RULES:
        if pat.search(text):
            return cat
    inj = scan(text)
    return "injection" if inj.suspected else None
