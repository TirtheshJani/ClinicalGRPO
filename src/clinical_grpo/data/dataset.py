"""Build a HuggingFace Dataset of {prompt, gold_codes} rows from a parquet file.

`prompt` is the chat-format message list ready for TRL `GRPOTrainer` (which
applies the tokenizer's chat template internally). `gold_codes` is passed
through to reward functions as a dataset column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset

from clinical_grpo.prompts.templates import build_chat


def load_grpo_dataset(parquet_path: Path | str) -> Dataset:
    df = pd.read_parquet(parquet_path)
    if "discharge_summary" not in df.columns or "icd10_codes" not in df.columns:
        raise ValueError(
            f"{parquet_path} missing required columns 'discharge_summary' / 'icd10_codes'"
        )
    df = df[df["icd10_codes"].map(len) > 0].reset_index(drop=True)
    return Dataset.from_dict(
        {
            "prompt": [build_chat(s) for s in df["discharge_summary"]],
            "gold_codes": [list(cs) for cs in df["icd10_codes"]],
        }
    )
