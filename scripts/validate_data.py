"""Validate a processed parquet before training.

Usage: python scripts/validate_data.py data/processed/mimic.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from clinical_grpo.utils.icd10 import is_valid


@dataclass
class ValidationReport:
    path: str
    n_rows: int
    n_empty_summaries: int
    n_empty_code_lists: int
    n_invalid_codes: int
    summary_len_min: float
    summary_len_max: float
    summary_len_mean: float
    summary_len_p50: float
    summary_len_p95: float
    codes_per_row_mean: float
    codes_per_row_max: int


def validate_parquet(path: Path | str) -> ValidationReport:
    path = Path(path)
    df = pd.read_parquet(path)
    lens = df["discharge_summary"].str.len()
    codes_per_row = df["icd10_codes"].map(len)
    n_invalid = int(
        df["icd10_codes"].map(lambda cs: sum(1 for c in cs if not is_valid(c))).sum()
    )
    return ValidationReport(
        path=str(path),
        n_rows=len(df),
        n_empty_summaries=int((lens == 0).sum()),
        n_empty_code_lists=int(codes_per_row.eq(0).sum()),
        n_invalid_codes=n_invalid,
        summary_len_min=float(lens.min()),
        summary_len_max=float(lens.max()),
        summary_len_mean=float(lens.mean()),
        summary_len_p50=float(lens.quantile(0.5)),
        summary_len_p95=float(lens.quantile(0.95)),
        codes_per_row_mean=float(codes_per_row.mean()),
        codes_per_row_max=int(codes_per_row.max()),
    )


def _print_report(r: ValidationReport) -> None:
    def tag(n):
        return "ok" if n == 0 else "WARN"

    print(f"[{tag(r.n_empty_summaries)}] empty summaries: {r.n_empty_summaries}/{r.n_rows}")
    print(f"[{tag(r.n_empty_code_lists)}] empty code lists: {r.n_empty_code_lists}/{r.n_rows}")
    print(f"[{tag(r.n_invalid_codes)}] invalid ICD-10 codes: {r.n_invalid_codes}")
    print(
        f"[info] summary chars: min={r.summary_len_min:.0f} p50={r.summary_len_p50:.0f} "
        f"p95={r.summary_len_p95:.0f} max={r.summary_len_max:.0f}"
    )
    print(f"[info] codes/row: mean={r.codes_per_row_mean:.1f} max={r.codes_per_row_max}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet", type=Path)
    args = p.parse_args()
    r = validate_parquet(args.parquet)
    _print_report(r)
    if r.n_empty_summaries or r.n_empty_code_lists or r.n_invalid_codes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
