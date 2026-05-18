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
    print("[downloading] ICD-9 -> ICD-10 GEM crosswalk ...")
    subprocess.run([sys.executable, "scripts/download_gem.py"], check=True)


def _check_mimic() -> None:
    if not MIMIC_RAW.exists():
        print(
            f"[missing] MIMIC-III demo not found at {MIMIC_RAW}\n"
            "  Open-access structured tables (no credentialing):\n"
            "    curl -fsSL -o data/raw/mimic-demo.zip "
            "https://physionet.org/content/mimiciii-demo/get-zip/1.4/\n"
            "  Clinical notes (NOTEEVENTS.csv with content) require\n"
            "  PhysioNet credentialing: https://physionet.org/settings/credentialing/"
        )
        return

    csvs = list(MIMIC_RAW.glob("*.csv"))
    note_path = MIMIC_RAW / "NOTEEVENTS.csv"
    notes_stub = note_path.exists() and note_path.stat().st_size < 1024

    if notes_stub:
        print(
            f"[warn] MIMIC-III demo: {MIMIC_RAW} ({len(csvs)} CSV files), "
            "but NOTEEVENTS.csv is a header-only stub.\n"
            "  The open-access demo intentionally omits clinical notes for de-id reasons.\n"
            "  Training will produce empty discharge summaries. To get real notes:\n"
            "    1. Register + credential at https://physionet.org/settings/credentialing/\n"
            "    2. Complete CITI 'Data or Specimens Only Research' training (~2h, free)\n"
            "    3. Sign the MIMIC-III data use agreement\n"
            "    4. Replace NOTEEVENTS.csv with the credentialed version"
        )
    else:
        print(f"[ok] MIMIC-III demo: {MIMIC_RAW} ({len(csvs)} CSV files, notes present)")


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
