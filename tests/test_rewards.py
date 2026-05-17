"""Unit tests for the GRPO reward functions. Pure CPU; no model required."""

from clinical_grpo.rewards.composite import (
    chapter_partial_reward,
    exact_match_reward,
    format_validity_reward,
    parse_completion,
)


def _completion(codes: list[str]) -> str:
    inner = ", ".join(f'"{c}"' for c in codes)
    return f'{{"codes": [{inner}]}}'


def test_parse_completion_valid():
    assert parse_completion(_completion(["E11.9", "I10"])) == ["E11.9", "I10"]


def test_parse_completion_strips_invalid_codes():
    # "250.00" is ICD-9 -> dropped; "I10" kept.
    assert parse_completion(_completion(["250.00", "I10"])) == ["I10"]


def test_parse_completion_malformed_json_returns_none():
    assert parse_completion("just some prose") is None
    assert parse_completion('{"codes": "not a list"}') is None
    assert parse_completion("") is None


def test_exact_match_perfect_recall():
    gold = [["E11.9", "I10"]]
    pred = [_completion(["E11.9", "I10"])]
    assert exact_match_reward(pred, gold) == [1.0]


def test_exact_match_partial_recall():
    gold = [["E11.9", "I10"]]
    pred = [_completion(["E11.9"])]
    assert exact_match_reward(pred, gold) == [0.5]


def test_exact_match_no_credit_for_chapter_only_match():
    gold = [["E11.9"]]                # diabetes
    pred = [_completion(["E78.5"])]   # hyperlipidemia, same chapter
    assert exact_match_reward(pred, gold) == [0.0]


def test_chapter_partial_awards_same_chapter():
    gold = [["E11.9"]]
    pred = [_completion(["E78.5"])]
    [r] = chapter_partial_reward(pred, gold)
    assert r == 0.3   # chapter_weight=0.3, no hallucinations


def test_chapter_partial_penalizes_hallucinations():
    gold = [["E11.9"]]
    pred = [_completion(["E11.9", "Z99.0"])]  # one exact, one out-of-chapter
    [r] = chapter_partial_reward(pred, gold)
    # chapter component = 0 (no chapter-only matches), hallucination penalty
    # = 0.1 * (1/2) for the Z code that is neither gold nor gold-chapter.
    assert r == -0.05


def test_format_validity_rewards_parse_failure_negative():
    [r] = format_validity_reward(["not json"])
    assert r == -0.5


def test_format_validity_rewards_parse_success_zero():
    [r] = format_validity_reward([_completion(["I10"])])
    assert r == 0.0


def test_duplicate_predicted_codes_are_deduped():
    gold = [["I10"]]
    pred = [_completion(["I10", "I10"])]
    assert exact_match_reward(pred, gold) == [1.0]
