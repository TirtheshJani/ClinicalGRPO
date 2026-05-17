"""Bootstrap 95% CI for micro-code F1. No scipy dependency."""

from __future__ import annotations

import random
from dataclasses import dataclass

from clinical_grpo.eval.f1 import micro_code_f1


@dataclass
class BootstrapResult:
    mean_f1: float
    ci_low: float
    ci_high: float
    n_resamples: int


def bootstrap_f1_ci(
    preds: list[list[str]],
    golds: list[list[str]],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap CI for micro-code F1 over (pred, gold) sample pairs."""
    rng = random.Random(seed)
    n = len(preds)
    scores: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        scores.append(micro_code_f1([preds[i] for i in idx], [golds[i] for i in idx]).f1)
    scores.sort()
    alpha = 1.0 - ci
    lo = int(alpha / 2 * n_resamples)
    hi = int((1 - alpha / 2) * n_resamples) - 1
    return BootstrapResult(
        mean_f1=micro_code_f1(preds, golds).f1,
        ci_low=scores[lo],
        ci_high=scores[hi],
        n_resamples=n_resamples,
    )
