"""End-to-end integration tests for the ClinicalGRPO pipeline.

All tests run on CPU with no GPU or model weights required.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Full reward pipeline integration
# ---------------------------------------------------------------------------


def test_full_reward_pipeline(tmp_path):
    """Parquet → dataset → rewards: all three functions return valid float lists."""
    from clinical_grpo.data.dataset import load_grpo_dataset
    from clinical_grpo.rewards.composite import (
        chapter_partial_reward,
        exact_match_reward,
        format_validity_reward,
    )

    rows = {
        "discharge_summary": [
            "Type 2 diabetes with hypertension.",
            "Heart failure, systolic.",
            "Sepsis due to UTI.",
            "CKD stage 3.",
        ],
        "icd10_codes": [["E11.9", "I10"], ["I50.2"], ["A41.9", "N39.0"], ["N18.3"]],
    }
    pd.DataFrame(rows).to_parquet(tmp_path / "data.parquet")
    ds = load_grpo_dataset(tmp_path / "data.parquet")

    # Simulate perfect completions
    completions = [json.dumps({"codes": gold}) for gold in ds["gold_codes"]]
    gold = ds["gold_codes"]

    exact = exact_match_reward(completions, gold)
    chapter = chapter_partial_reward(completions, gold)
    fmt = format_validity_reward(completions)

    assert len(exact) == len(completions)
    assert len(chapter) == len(completions)
    assert len(fmt) == len(completions)
    assert all(isinstance(r, float) for r in exact)
    assert all(r == 1.0 for r in exact)  # perfect completions → recall = 1.0
    assert all(r == 0.0 for r in fmt)  # valid JSON → no format penalty


# ---------------------------------------------------------------------------
# 2. Dataset column contract
# ---------------------------------------------------------------------------


def test_dataset_prompt_is_chat_format(tmp_path):
    """Dataset rows have prompt (list of role/content dicts) and gold_codes (list of str)."""
    from clinical_grpo.data.dataset import load_grpo_dataset

    pd.DataFrame(
        {
            "discharge_summary": ["A note."],
            "icd10_codes": [["I10"]],
        }
    ).to_parquet(tmp_path / "d.parquet")
    ds = load_grpo_dataset(tmp_path / "d.parquet")

    assert "prompt" in ds.column_names
    assert "gold_codes" in ds.column_names
    prompt = ds[0]["prompt"]
    assert isinstance(prompt, list)
    assert all("role" in m and "content" in m for m in prompt)
    roles = [m["role"] for m in prompt]
    assert "system" in roles and "user" in roles


# ---------------------------------------------------------------------------
# 3. Reward functions accept mismatched predictions gracefully
# ---------------------------------------------------------------------------


def test_reward_handles_malformed_and_missing():
    """Malformed JSON gets format penalty; partial predictions get partial recall."""
    from clinical_grpo.rewards.composite import (
        exact_match_reward,
        format_validity_reward,
    )

    completions = ["not json at all", '{"codes": ["E11.9"]}']
    gold = [["E11.9", "I10"], ["E11.9"]]

    exact = exact_match_reward(completions, gold)
    fmt = format_validity_reward(completions)

    assert exact[0] == 0.0  # malformed → no credit
    assert exact[1] == 1.0  # perfect match
    assert fmt[0] == -0.5  # malformed → penalty
    assert fmt[1] == 0.0  # valid JSON → no penalty


# ---------------------------------------------------------------------------
# 4. CLI entry points parse without error
# ---------------------------------------------------------------------------

_CWD = "/home/user/ClinicalGRPO"


def test_train_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--help"],
        capture_output=True,
        text=True,
        cwd=_CWD,
    )
    assert result.returncode == 0
    assert "config" in result.stdout.lower()


def test_train_sft_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/train_sft.py", "--help"],
        capture_output=True,
        text=True,
        cwd=_CWD,
    )
    assert result.returncode == 0


def test_eval_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/eval.py", "--help"],
        capture_output=True,
        text=True,
        cwd=_CWD,
    )
    assert result.returncode == 0


def test_preprocess_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "clinical_grpo.data.preprocess", "--help"],
        capture_output=True,
        text=True,
        cwd=_CWD,
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# 5. ICD-10 + reward round-trip
# ---------------------------------------------------------------------------


def test_icd10_reward_roundtrip():
    """Codes normalized by icd10.normalize() are accepted by reward functions."""
    from clinical_grpo.rewards.composite import exact_match_reward
    from clinical_grpo.utils.icd10 import normalize

    raw = ["e119", "i10", "n179"]  # lowercase, no dots
    normalized = [normalize(c) for c in raw]
    completion = json.dumps({"codes": normalized})
    gold = [normalized]
    assert exact_match_reward([completion], gold) == [1.0]
