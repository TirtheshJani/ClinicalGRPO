"""Tests for clinical_grpo.eval.run_eval (Unsloth mocked).

Strategy for safe module isolation
------------------------------------
``run_eval`` does ``from unsloth import FastLanguageModel`` *inside* the body
of ``generate_predictions``, so the top-level import of ``run_eval`` itself
works fine even when ``unsloth`` is not installed.  The module is imported
once at collection time (module-level import below).  Individual tests then
patch the ``unsloth`` name *inside* ``run_eval``'s namespace, or patch
``generate_predictions`` itself, so ``sys.modules`` is never manipulated in a
way that could evict already-loaded C-extension modules (numpy, pandas, …).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Import once at module scope so numpy/pandas C-extensions are already loaded
# before any test runs.  All tests re-use this module object.
import clinical_grpo.eval.run_eval as run_eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tok(decode_value: str = '{"codes": ["E11.9"]}') -> MagicMock:
    """Build a fully-wired mock tokenizer."""
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "prompt"

    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 5]

    mock_inputs_on_device = MagicMock()
    mock_inputs_on_device.__getitem__ = lambda self, key: mock_input_ids

    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: mock_input_ids
    mock_inputs.to.return_value = mock_inputs_on_device

    mock_tok.return_value = mock_inputs
    mock_tok.decode.return_value = decode_value
    return mock_tok


def _make_mock_model() -> MagicMock:
    """Build a fully-wired mock model."""
    mock_model = MagicMock()
    mock_out = MagicMock()
    mock_out.__getitem__ = lambda self, idx: MagicMock()
    mock_model.generate.return_value = mock_out
    mock_model.device = "cpu"
    return mock_model


def _make_unsloth_patch(mock_model, mock_tok) -> MagicMock:
    """Return a mock FastLanguageModel class whose from_pretrained returns (model, tok)."""
    mock_fl = MagicMock()
    mock_fl.from_pretrained.return_value = (mock_model, mock_tok)
    return mock_fl


def _make_dataset(n: int = 3) -> list[dict]:
    """Minimal list of dicts simulating a loaded HF dataset."""
    return [
        {"prompt": [{"role": "user", "content": "note"}], "gold_codes": ["E11.9"]}
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests for generate_predictions
# ---------------------------------------------------------------------------

def test_generate_predictions_returns_one_per_row():
    """generate_predictions returns one string per dataset row."""
    mock_tok = _make_mock_tok()
    mock_model = _make_mock_model()
    mock_fl = _make_unsloth_patch(mock_model, mock_tok)

    with patch.dict("clinical_grpo.eval.run_eval.__dict__", {}):
        pass  # just verifying the module attribute approach below

    # Patch the FastLanguageModel name as it is used *inside* generate_predictions
    fake_unsloth = MagicMock()
    fake_unsloth.FastLanguageModel = mock_fl
    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        ds = _make_dataset(3)
        preds = run_eval.generate_predictions(Path("/fake/adapter"), ds)

    assert len(preds) == 3
    assert all(isinstance(p, str) for p in preds)


def test_generate_predictions_decodes_generation():
    """generate_predictions returns what tokenizer.decode returns."""
    mock_tok = _make_mock_tok(decode_value='{"codes": ["Z38.00"]}')
    mock_model = _make_mock_model()
    mock_fl = _make_unsloth_patch(mock_model, mock_tok)

    fake_unsloth = MagicMock()
    fake_unsloth.FastLanguageModel = mock_fl
    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        ds = _make_dataset(2)
        preds = run_eval.generate_predictions(Path("/fake/adapter"), ds)

    assert all(p == '{"codes": ["Z38.00"]}' for p in preds)


# ---------------------------------------------------------------------------
# Test for main()
# ---------------------------------------------------------------------------

def test_main_writes_report_json(tmp_path):
    """main() writes a JSON report with expected top-level keys."""
    # load_grpo_dataset expects columns 'discharge_summary' and 'icd10_codes'.
    # main() builds path as: stem.parent / f"{stem.name}_{split}.parquet"
    # With --data-stem tmp_path/data and default --split test it reads
    # tmp_path/data_test.parquet.
    parquet = tmp_path / "data_test.parquet"
    pd.DataFrame(
        {
            "discharge_summary": ["Patient with diabetes."],
            "icd10_codes": [["E11.9"]],
        }
    ).to_parquet(parquet)

    out = tmp_path / "report.json"

    with patch(
        "sys.argv",
        [
            "eval",
            "--adapter", str(tmp_path / "adapter"),
            "--data-stem", str(tmp_path / "data"),
            "--out", str(out),
        ],
    ), patch.object(
        run_eval,
        "generate_predictions",
        return_value=['{"codes": ["E11.9"]}'],
    ):
        run_eval.main()

    assert out.exists(), "main() did not write output file"
    report = json.loads(out.read_text())
    assert "micro_code_f1" in report, f"missing micro_code_f1, got keys: {list(report)}"
    assert "chapter_f1" in report, f"missing chapter_f1, got keys: {list(report)}"
    assert "micro_code_f1_bootstrap" in report, f"missing bootstrap, got keys: {list(report)}"
    assert report["n"] == 1
