"""Discharge-summary preprocessing: cleanup, truncation, patient-disjoint splits.

Splits are patient-disjoint (not encounter-disjoint) — multiple admissions for
the same patient must not straddle train/val/test or evaluation leaks.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from clinical_grpo.data import mimic_demo, synthea
from clinical_grpo.utils.icd10 import dedupe_codes, is_valid

MAX_CHARS = 8000  # ~2048 tokens at typical English ratio; final truncation done by tokenizer


def clean_summary(text: str) -> str:
    """Strip MIMIC's PHI redaction tokens and collapse whitespace."""
    text = text.replace("[**", "").replace("**]", "")
    return " ".join(text.split())[:MAX_CHARS]


def build_table(records) -> pd.DataFrame:
    rows = []
    for r in records:
        codes = [c for c in dedupe_codes(r.icd10_codes) if is_valid(c)]
        if not codes:
            continue
        d = asdict(r)
        d["discharge_summary"] = clean_summary(d["discharge_summary"])
        d["icd10_codes"] = codes
        rows.append(d)
    return pd.DataFrame(rows)


def patient_split(
    df: pd.DataFrame,
    patient_col: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Patient-disjoint train/val/test split."""
    patients = sorted(df[patient_col].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    splits = {
        "train": set(patients[:n_train]),
        "val": set(patients[n_train : n_train + n_val]),
        "test": set(patients[n_train + n_val :]),
    }
    return {k: df[df[patient_col].isin(v)].reset_index(drop=True) for k, v in splits.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["mimic_demo", "synthea"], required=True)
    p.add_argument("--root", type=Path, default=None, help="Source data root (default: data/raw/<source>)")
    p.add_argument("--out", type=Path, required=True, help="Output parquet path (train split)")
    args = p.parse_args()

    root = args.root or Path("data/raw") / args.source
    if args.source == "mimic_demo":
        records = mimic_demo.load_demo(root)
        patient_col = "subject_id"
    else:
        records = synthea.load_bundles(root)
        patient_col = "patient_id"

    df = build_table(records)
    splits = patient_split(df, patient_col=patient_col)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    for name, part in splits.items():
        out_path = Path(f"{stem}_{name}.parquet") if name != "train" else args.out
        part.to_parquet(out_path, index=False)
        print(f"wrote {out_path}: {len(part)} rows")


if __name__ == "__main__":
    main()
