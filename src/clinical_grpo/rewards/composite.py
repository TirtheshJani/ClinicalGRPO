"""Output parser and the list of TRL-compatible reward functions.

We register THREE reward functions with TRL rather than one composite scalar so
each component is logged independently — diagnosing whether the model is
collapsing on formatting, gaming chapter credit, or actually learning codes is
trivial when each curve is separate.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from clinical_grpo.utils.icd10 import chapter_of, dedupe_codes, is_valid

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_completion(text: str) -> list[str] | None:
    """Extract and normalize predicted ICD-10 codes from a model completion.

    Returns None if the output does not contain a parseable JSON object with a
    `codes` list of strings. Invalid codes are dropped from the returned list
    but do not cause the parse to fail (they're penalized elsewhere).
    """
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    codes = obj.get("codes") if isinstance(obj, dict) else None
    if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
        return None
    return dedupe_codes([c for c in codes if is_valid(c)])


def exact_match_reward(
    completions: list[str],
    gold_codes: list[list[str]],
    **_: object,
) -> list[float]:
    """Recall on exact ICD-10 code matches."""
    out: list[float] = []
    for completion, gold in zip(completions, gold_codes):
        pred = parse_completion(completion)
        if pred is None:
            out.append(0.0)
            continue
        gold_set = set(dedupe_codes(gold))
        if not gold_set:
            out.append(0.0)
            continue
        out.append(len(set(pred) & gold_set) / len(gold_set))
    return out


def chapter_partial_reward(
    completions: list[str],
    gold_codes: list[list[str]],
    chapter_weight: float = 0.3,
    hallucination_penalty: float = 0.1,
    **_: object,
) -> list[float]:
    """Partial credit for predicted codes that share a chapter with any gold code,
    minus a penalty for predictions outside any gold chapter."""
    out: list[float] = []
    for completion, gold in zip(completions, gold_codes):
        pred = parse_completion(completion)
        if pred is None:
            out.append(0.0)
            continue
        gold_norm = dedupe_codes(gold)
        gold_set = set(gold_norm)
        gold_chapters = {chapter_of(c) for c in gold_norm} - {None}

        chapter_only = [
            c for c in pred if c not in gold_set and chapter_of(c) in gold_chapters
        ]
        hallucinations = [
            c for c in pred if c not in gold_set and chapter_of(c) not in gold_chapters
        ]

        chap = chapter_weight * (len(chapter_only) / max(len(gold_norm), 1))
        pen = hallucination_penalty * (len(hallucinations) / max(len(pred), 1))
        out.append(chap - pen)
    return out


def format_validity_reward(
    completions: list[str],
    format_invalid_penalty: float = 0.5,
    **_: object,
) -> list[float]:
    """+0.0 if the completion parses; -format_invalid_penalty otherwise."""
    return [
        0.0 if parse_completion(c) is not None else -format_invalid_penalty
        for c in completions
    ]


def build_reward_funcs(weights: dict[str, float]) -> list[Callable]:
    """Return the list of reward callables in the order TRL should log them.

    `weights` keys: `chapter_weight`, `hallucination_penalty`, `format_invalid_penalty`.
    `exact_weight` is applied by caller multiplying TRL's logged reward; keeping
    exact at 1.0 here means TRL's per-reward log directly equals recall.
    """
    cw = weights.get("chapter_weight", 0.3)
    hp = weights.get("hallucination_penalty", 0.1)
    fp = weights.get("format_invalid_penalty", 0.5)

    def _chapter(completions, gold_codes, **kw):
        return chapter_partial_reward(
            completions, gold_codes,
            chapter_weight=cw, hallucination_penalty=hp, **kw,
        )

    def _format(completions, **kw):
        return format_validity_reward(completions, format_invalid_penalty=fp, **kw)

    _chapter.__name__ = "chapter_partial_reward"
    _format.__name__ = "format_validity_reward"
    return [exact_match_reward, _chapter, _format]
