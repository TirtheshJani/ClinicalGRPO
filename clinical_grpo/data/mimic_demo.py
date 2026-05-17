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


def load_demo(root: Path | str) -> list[MimicRecord]:
    """Join discharge summaries with their ICD-9 diagnoses; map to ICD-10.

    TODO: implement ICD-9 -> ICD-10 mapping via the CMS GEM file (ship under
    `data/raw/gem/2018_I9gem.txt`). For now this returns the ICD-9 codes as-is
    so the rest of the pipeline can be exercised end-to-end.
    """
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


def _icd9_to_icd10(icd9_codes: list[str]) -> list[str]:
    """TODO: real GEM mapping. Returns the inputs unchanged for now."""
    return list(icd9_codes)
