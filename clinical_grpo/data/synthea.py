"""Synthea-generated synthetic discharge summary loader.

Synthea outputs FHIR bundles per patient (`output/fhir/<name>_<id>.json`) with
Encounter, Condition, and DocumentReference resources. We extract discharge
summaries and ICD-10 codes (Synthea Conditions already carry ICD-10-CM codings
via the SNOMED -> ICD-10 mapping in its output config).

Generator: https://github.com/synthetichealth/synthea

TODO: implement FHIR bundle parsing once we have a sample dump in
`data/raw/synthea/`. The signature below is the contract `preprocess.py` calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyntheaRecord:
    patient_id: str
    discharge_summary: str
    icd10_codes: list[str]


def load_bundles(root: Path | str) -> list[SyntheaRecord]:
    """Walk Synthea FHIR output and yield discharge encounters with ICD-10 codes."""
    # TODO: parse FHIR Bundles in <root>/fhir/*.json
    raise NotImplementedError("Synthea loader not yet implemented.")
