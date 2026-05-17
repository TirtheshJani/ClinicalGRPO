"""Data setup helper: downloads GEM crosswalk and checks for MIMIC/Synthea data.

Usage: python scripts/setup_data.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GEM_PATH = Path("data/raw/gem/2018_I9gem.txt")
MIMIC_RAW = Path("data/raw/mimic-iii-demo")
SYNTHEA_RAW = Path("data/raw/synthea/fhir")
MIMIC_PROCESSED = Path("data/processed/mimic.parquet")


def _check_gem() -> None:
    if GEM_PATH.exists():
        print(f"[ok] GEM crosswalk: {GEM_PATH} ({GEM_PATH.stat().st_size:,} bytes)")
        return
    print("[downloading] ICD-9→ICD-10 GEM crosswalk ...")
    subprocess.run([sys.executable, "scripts/download_gem.py"], check=True)


def _check_mimic() -> None:
    if MIMIC_RAW.exists():
        csvs = list(MIMIC_RAW.glob("*.csv"))
        print(f"[ok] MIMIC-III demo: {MIMIC_RAW} ({len(csvs)} CSV files)")
    else:
        print(
            f"[missing] MIMIC-III demo not found at {MIMIC_RAW}\n"
            "  Download from https://physionet.org/content/mimiciii-demo/1.4/\n"
            f"  and extract to {MIMIC_RAW}/"
        )


def _check_synthea() -> None:
    if SYNTHEA_RAW.exists():
        bundles = list(SYNTHEA_RAW.glob("*.json"))
        print(f"[ok] Synthea FHIR: {SYNTHEA_RAW} ({len(bundles)} bundles)")
    else:
        print(
            f"[optional] Synthea FHIR not found at {SYNTHEA_RAW}\n"
            "  Generate with: https://github.com/synthetichealth/synthea\n"
            "  Run with --exporter.fhir.export=true and copy output/fhir/ "
            f"to {SYNTHEA_RAW}/"
        )


def _check_processed() -> None:
    if MIMIC_PROCESSED.exists():
        import pandas as pd
        df = pd.read_parquet(MIMIC_PROCESSED)
        print(f"[ok] Processed MIMIC train: {len(df)} rows")
    else:
        print(
            f"[missing] Processed data not found at {MIMIC_PROCESSED}\n"
            "  Run: python -m clinical_grpo.data.preprocess "
            "--source mimic_demo --out data/processed/mimic.parquet"
        )


def main() -> None:
    print("=== ClinicalGRPO data setup ===\n")
    _check_gem()
    _check_mimic()
    _check_synthea()
    _check_processed()
    print("\nDone.")


if __name__ == "__main__":
    main()
