"""Deterministic code-set and chapter-set F1 metrics."""

from __future__ import annotations

from dataclasses import dataclass

from clinical_grpo.utils.icd10 import chapter_of, dedupe_codes


@dataclass
class F1Result:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _set_prf(pred: set[str], gold: set[str]) -> F1Result:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return F1Result(p, r, f, tp, fp, fn)


def micro_code_f1(preds: list[list[str]], golds: list[list[str]]) -> F1Result:
    """Micro-averaged F1 over the multiset of (sample, code) pairs."""
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        ps = set(dedupe_codes(p))
        gs = set(dedupe_codes(g))
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return F1Result(prec, rec, f, tp, fp, fn)


def chapter_f1(preds: list[list[str]], golds: list[list[str]]) -> F1Result:
    """Micro-F1 at the chapter level (more forgiving than exact-code F1)."""
    pred_chapters: list[list[str]] = [
        [ch for c in p if (ch := chapter_of(c)) is not None] for p in preds
    ]
    gold_chapters: list[list[str]] = [
        [ch for c in g if (ch := chapter_of(c)) is not None] for g in golds
    ]
    return micro_code_f1(pred_chapters, gold_chapters)
