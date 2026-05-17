"""Run batch ICD-10 inference on a parquet file.

Usage: python scripts/batch_infer.py --adapter outputs/grpo/adapter --input data/processed/mimic_test.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import tyro


def run_batch_infer(predictor, input_path: Path, out_path: Path) -> None:
    """Read *input_path* parquet, run predict_batch, write results to *out_path*."""
    df = pd.read_parquet(input_path)
    df["predicted_codes"] = predictor.predict_batch(df["discharge_summary"].tolist())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")


def main(
    adapter: Path,
    input: Path = Path("data/processed/mimic_test.parquet"),
    out: Path = Path("outputs/predictions.parquet"),
    load_in_4bit: bool = True,
) -> None:
    """Batch ICD-10 inference: reads a parquet, writes predicted_codes column."""
    from clinical_grpo.inference import ICD10Predictor  # deferred: GPU/Unsloth optional

    run_batch_infer(ICD10Predictor(str(adapter), load_in_4bit=load_in_4bit), input, out)


if __name__ == "__main__":
    tyro.cli(main)
