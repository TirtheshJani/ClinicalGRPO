"""Tests for clinical_grpo.data.dataset.load_grpo_dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from clinical_grpo.data.dataset import load_grpo_dataset


def _write_parquet(tmp_path, rows: dict) -> str:
    df = pd.DataFrame(rows)
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture()
def basic_parquet(tmp_path):
    rows = {
        "discharge_summary": ["Patient admitted with diabetes.", "Heart failure case."],
        "icd10_codes": [["E11.9"], ["I50.9"]],
    }
    return _write_parquet(tmp_path, rows)


def test_returns_prompt_and_gold_codes_columns(basic_parquet):
    dataset = load_grpo_dataset(basic_parquet)
    assert "prompt" in dataset.column_names
    assert "gold_codes" in dataset.column_names


def test_length_equals_number_of_input_rows(basic_parquet):
    dataset = load_grpo_dataset(basic_parquet)
    assert len(dataset) == 2


def test_prompt_is_list_of_dicts(basic_parquet):
    dataset = load_grpo_dataset(basic_parquet)
    for prompt in dataset["prompt"]:
        assert isinstance(prompt, list)
        assert len(prompt) > 0


def test_each_prompt_dict_has_role_and_content(basic_parquet):
    dataset = load_grpo_dataset(basic_parquet)
    for prompt in dataset["prompt"]:
        for msg in prompt:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg


def test_gold_codes_are_lists_of_strings(basic_parquet):
    dataset = load_grpo_dataset(basic_parquet)
    for codes in dataset["gold_codes"]:
        assert isinstance(codes, list)
        for code in codes:
            assert isinstance(code, str)


def test_rows_with_empty_codes_are_filtered(tmp_path):
    rows = {
        "discharge_summary": [
            "Patient admitted with diabetes.",
            "Heart failure case.",
            "No codes assigned.",
        ],
        "icd10_codes": [["E11.9"], ["I50.9"], []],
    }
    path = _write_parquet(tmp_path, rows)
    dataset = load_grpo_dataset(path)
    assert len(dataset) == 2
