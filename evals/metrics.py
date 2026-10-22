"""Metric functions. Small and dependency-light so they're easy to test and explain."""
from __future__ import annotations

from collections import Counter


def set_f1(pred: list | set, gold: list | set) -> float:
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def field_accuracy(pred: dict, gold: dict, fields: list[str]) -> dict[str, bool]:
    return {f: pred.get(f) == gold.get(f) for f in fields}


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via edit distance (same definition as the jiwer library)."""
    r, h = reference.lower().split(), hypothesis.lower().split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def pr_auc(y_true: list[int], scores: list[float]) -> float:
    """Average precision (area under the precision-recall curve)."""
    pairs = sorted(zip(scores, y_true), key=lambda t: -t[0])
    pos = sum(y_true)
    if pos == 0:
        return 0.0
    tp, ap = 0, 0.0
    for k, (_, y) in enumerate(pairs, start=1):
        if y:
            tp += 1
            ap += tp / k
    return ap / pos


def recall_at_budget(y_true: list[int], scores: list[float], budget: float = 0.10) -> float:
    """Share of all fraud caught if investigators can only review the top `budget` share of claims."""
    n = max(1, int(round(len(scores) * budget)))
    top = sorted(zip(scores, y_true), key=lambda t: -t[0])[:n]
    pos = sum(y_true)
    return sum(y for _, y in top) / pos if pos else 0.0


def confusion(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(gold) | set(pred))
    c = Counter(zip(gold, pred))
    return {g: {p: c[(g, p)] for p in labels} for g in labels}


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
