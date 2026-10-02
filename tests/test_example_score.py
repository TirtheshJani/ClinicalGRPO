"""Tests for examples/score_completion.py (CPU-only reward walkthrough)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_score_examples_returns_one_row_per_completion():
    from examples.score_completion import GOLD, SAMPLES, score_examples

    rows = score_examples(GOLD, SAMPLES)
    assert [r["name"] for r in rows] == [name for name, _ in SAMPLES]
    by_name = {r["name"]: r for r in rows}
    assert by_name["perfect"]["exact"] == 1.0
    assert by_name["perfect"]["format"] == 0.0
    assert by_name["malformed"]["format"] == -0.5
    assert by_name["malformed"]["exact"] == 0.0
    assert by_name["perfect"]["total"] > by_name["hallucinating"]["total"]


def test_score_completion_cli_runs():
    r = subprocess.run(
        [sys.executable, "examples/score_completion.py"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr
    assert "exact_match" in r.stdout
