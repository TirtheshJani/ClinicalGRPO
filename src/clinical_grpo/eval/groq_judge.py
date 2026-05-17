"""Groq Llama-3.3-70B LLM judge.

Used to score predicted code lists against gold for plausibility (catches cases
where the model emits a closely related but not identical code that a human
coder would accept). Caches per-(pred, gold) result to disk so reruns don't
burn the 1000 RPD free-tier budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

JUDGE_MODEL = "llama-3.3-70b-versatile"

_RUBRIC = (
    "You are auditing ICD-10-CM coding. Given a list of GOLD codes (correct) and "
    "PRED codes (predicted by a model), return a JSON object: "
    '{"matched": <int>, "near_matches": <int>, "spurious": <int>, "missed": <int>}. '
    "A near_match is a predicted code in the same family as a gold code (same first 3 "
    "characters, different specificity). Respond with JSON only."
)


@dataclass
class JudgeVerdict:
    matched: int
    near_matches: int
    spurious: int
    missed: int


class RateLimiter:
    """Simple sliding-window limiter for the Groq free tier (1000 RPD by default)."""

    def __init__(self, requests_per_day: int = 1000) -> None:
        self.rpd = requests_per_day
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        now = time.time()
        window_start = now - 86400.0
        self._timestamps = [t for t in self._timestamps if t > window_start]
        if len(self._timestamps) >= self.rpd:
            sleep_for = self._timestamps[0] + 86400.0 - now
            time.sleep(max(sleep_for, 0))
        self._timestamps.append(time.time())


def _cache_key(gold: list[str], pred: list[str]) -> str:
    payload = json.dumps({"gold": sorted(gold), "pred": sorted(pred)})
    return hashlib.sha256(payload.encode()).hexdigest()


class GroqJudge:
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str = "outputs/groq_cache",
        rpd: int = 1000,
    ) -> None:
        from groq import Groq  # imported lazily so tests can mock without the dep

        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(requests_per_day=rpd)

    def score(self, gold: list[str], pred: list[str]) -> JudgeVerdict:
        key = _cache_key(gold, pred)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            return JudgeVerdict(**json.loads(cache_path.read_text()))

        self.limiter.acquire()
        resp = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _RUBRIC},
                {
                    "role": "user",
                    "content": json.dumps({"gold": gold, "pred": pred}),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        verdict_dict = json.loads(resp.choices[0].message.content)
        verdict = JudgeVerdict(**verdict_dict)
        cache_path.write_text(json.dumps(verdict_dict))
        return verdict
