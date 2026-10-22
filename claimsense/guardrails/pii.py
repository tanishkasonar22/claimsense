"""PII redaction.

Default: fast regex recognizers (no downloads needed).
Optional: Microsoft Presidio for names and addresses (pip install presidio-analyzer
presidio-anonymizer, then python -m spacy download en_core_web_lg). Enable with
PII_ENGINE=presidio.

Redaction replaces values with typed placeholders like <PHONE_1>, and returns a mapping
so the original values can be restored for the human adjuster's view (never for the LLM).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    # Real VINs never contain I, O or Q, but synthetic ones can, so accept any 17 letters/digits
    # that include at least one digit and one letter.
    "VIN": re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{17}\b"),
    "POLICY_ID": re.compile(r"\bP\d{5}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PLATE": re.compile(r"\bplate\s+([A-Z0-9]{2,4}-[A-Z0-9]{3,4})\b", re.IGNORECASE),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}
# Names are hard to catch with regex. We catch the common "My name is X Y" pattern.
NAME_PHRASE = re.compile(r"((?i:my name is|this is|customer:|other party:))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")


@dataclass
class Redaction:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.mapping)


def redact(text: str, known_names: list[str] | None = None) -> Redaction:
    if os.environ.get("PII_ENGINE") == "presidio":
        return _redact_presidio(text)
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}

    def token(kind: str, value: str) -> str:
        for k, v in mapping.items():
            if v == value:
                return k
        counters[kind] = counters.get(kind, 0) + 1
        key = f"<{kind}_{counters[kind]}>"
        mapping[key] = value
        return key

    out = text
    for name in known_names or []:
        if name and name in out:
            out = out.replace(name, token("PERSON", name))

    def name_sub(m: re.Match) -> str:
        return f"{m.group(1)} {token('PERSON', m.group(2))}"

    out = NAME_PHRASE.sub(name_sub, out)
    for kind, pat in PATTERNS.items():
        if kind == "PLATE":
            out = pat.sub(lambda m: f"plate {token('PLATE', m.group(1))}", out)
        else:
            out = pat.sub(lambda m, k=kind: token(k, m.group(0)), out)
    return Redaction(out, mapping)


def restore(text: str, mapping: dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def _redact_presidio(text: str) -> Redaction:  # pragma: no cover - optional dependency
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity="VIN", patterns=[Pattern("vin", PATTERNS["VIN"].pattern, 0.6)]))
    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity="POLICY_ID", patterns=[Pattern("policy", PATTERNS["POLICY_ID"].pattern, 0.6)]))
    results = analyzer.analyze(text=text, language="en")
    mapping: dict[str, str] = {}
    for i, r in enumerate(sorted(results, key=lambda r: r.start)):
        mapping[f"<{r.entity_type}_{i + 1}>"] = text[r.start:r.end]
    anon = AnonymizerEngine().anonymize(text=text, analyzer_results=results)
    return Redaction(anon.text, mapping)
