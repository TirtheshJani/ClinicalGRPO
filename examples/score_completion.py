"""Score sample model completions with the GRPO reward functions (CPU only).

This is the exact code path GRPOTrainer calls during training: each completion is
parsed against the {"codes": [...]} output contract, then scored by the three
reward functions. No model, GPU, or data download is needed.

Usage (after `pip install -e .`): python examples/score_completion.py
"""

from __future__ import annotations

from clinical_grpo.rewards.composite import build_reward_funcs, parse_completion

# Gold codes for one hypothetical admission: type 2 diabetes, hypertension, acute kidney injury.
GOLD = ["E11.9", "I10", "N17.9"]

SAMPLES: list[tuple[str, str]] = [
    ("perfect", '{"codes": ["E11.9", "I10", "N17.9"]}'),
    ("right_chapter", '{"codes": ["E11.9", "I12.9"]}'),
    ("hallucinating", '{"codes": ["E11.9", "S72.001A"]}'),
    ("with_think_block", '<think>HTN and DM2 noted.</think>{"codes": ["i10", "e11.9"]}'),
    ("malformed", "Codes: E11.9, I10"),
]

# Same defaults as the `rewards:` block in configs/train.yaml.
WEIGHTS = {"chapter_weight": 0.3, "hallucination_penalty": 0.1, "format_invalid_penalty": 0.5}


def score_examples(gold: list[str], samples: list[tuple[str, str]]) -> list[dict]:
    """Return per-completion reward components and their sum."""
    exact_fn, chapter_fn, format_fn = build_reward_funcs(WEIGHTS)
    completions = [text for _, text in samples]
    golds = [gold] * len(completions)
    exact = exact_fn(completions, gold_codes=golds)
    chapter = chapter_fn(completions, gold_codes=golds)
    fmt = format_fn(completions, gold_codes=golds)
    rows = []
    for (name, text), e, c, f in zip(samples, exact, chapter, fmt):
        rows.append(
            {
                "name": name,
                "parsed": parse_completion(text),
                "exact": e,
                "chapter": c,
                "format": f,
                "total": e + c + f,
            }
        )
    return rows


def main() -> None:
    print(f"gold codes: {GOLD}\n")
    header = f"{'sample':<18}{'exact_match':>12}{'chapter':>10}{'format':>9}{'total':>8}  parsed"
    print(header)
    print("-" * len(header))
    for r in score_examples(GOLD, SAMPLES):
        print(
            f"{r['name']:<18}{r['exact']:>12.3f}{r['chapter']:>10.3f}"
            f"{r['format']:>9.2f}{r['total']:>8.3f}  {r['parsed']}"
        )


if __name__ == "__main__":
    main()
