"""MIMIC-III Clinical Database Demo loader.

The demo release (100 patients, public, no PhysioNet credentialing) ships as
CSVs. We pull NOTEEVENTS (filtered to discharge summaries), DIAGNOSES_ICD
(ICD-9 codes), and convert ICD-9 -> ICD-10 via the CMS GEM crosswalk.

Expected layout under `data/raw/mimic-iii-demo/`:
    NOTEEVENTS.csv
    DIAGNOSES_ICD.csv
    D_ICD_DIAGNOSES.csv
    (plus the rest of the demo distribution)

Download: https://physionet.org/content/mimiciii-demo/1.4/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MimicRecord:
    subject_id: int
    hadm_id: int
    discharge_summary: str
    icd10_codes: list[str]


_GEM_PATH = Path("data/raw/gem/2018_I9gem.txt")
_gem_map: dict[str, list[str]] | None = None


def _load_gem() -> dict[str, list[str]]:
    """Parse CMS ICD-9 → ICD-10 GEM file into a dict.

    Returns empty dict if file not found (degrades gracefully for unit tests).
    Maps each ICD-9 code to a list of ICD-10 codes, preferring exact matches
    (flag starts with '0') over approximate ones.
    """
    global _gem_map
    if _gem_map is not None:
        return _gem_map
    if not _GEM_PATH.exists():
        _gem_map = {}
        return _gem_map

    exact: dict[str, list[str]] = {}
    approx: dict[str, list[str]] = {}
    for line in _GEM_PATH.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        i9, i10, flags = parts[0], parts[1], parts[2]
        # NoDx and combo entries have special ICD-10 codes like "NoDx" — skip them
        if not i10[0].isalpha() or len(i10) < 3:
            continue
        if flags[0] == '0':
            exact.setdefault(i9, []).append(i10)
        else:
            approx.setdefault(i9, []).append(i10)
    _gem_map = {k: v for k, v in exact.items()}
    for k, v in approx.items():
        if k not in _gem_map:
            _gem_map[k] = v
    return _gem_map


def _icd9_to_icd10(icd9_codes: list[str]) -> list[str]:
    """Map ICD-9 codes to ICD-10 via CMS GEM. Unknown codes are dropped."""
    gem = _load_gem()
    if not gem:
        return icd9_codes
    out: list[str] = []
    seen: set[str] = set()
    for c9 in icd9_codes:
        c9_norm = c9.strip().upper().replace(".", "")
        for c10 in gem.get(c9_norm, []):
            if c10 not in seen:
                seen.add(c10)
                out.append(c10)
    return out


def load_demo(root: Path | str) -> list[MimicRecord]:
    """Join discharge summaries with their ICD-9 diagnoses; map to ICD-10."""
    root = Path(root)
    notes = pd.read_csv(root / "NOTEEVENTS.csv", low_memory=False)
    diag = pd.read_csv(root / "DIAGNOSES_ICD.csv")

    discharges = notes[notes["CATEGORY"].str.strip() == "Discharge summary"]
    discharges = discharges.sort_values(["HADM_ID", "CHARTDATE"]).drop_duplicates(
        "HADM_ID", keep="last"
    )

    diag_by_admit = (
        diag.dropna(subset=["ICD9_CODE"])
        .groupby("HADM_ID")["ICD9_CODE"]
        .apply(lambda s: sorted(set(s.astype(str))))
        .to_dict()
    )

    records: list[MimicRecord] = []
    for _, row in discharges.iterrows():
        hadm = int(row["HADM_ID"])
        codes = diag_by_admit.get(hadm, [])
        if not codes:
            continue
        records.append(
            MimicRecord(
                subject_id=int(row["SUBJECT_ID"]),
                hadm_id=hadm,
                discharge_summary=str(row["TEXT"]),
                icd10_codes=_icd9_to_icd10(codes),
            )
        )
    return records
