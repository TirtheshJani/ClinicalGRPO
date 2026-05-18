"""Synthea-generated synthetic discharge summary loader.

Synthea outputs FHIR bundles per patient (`output/fhir/<name>_<id>.json`) with
Encounter, Condition, and DocumentReference resources. We extract discharge
summaries and ICD-10 codes (Synthea Conditions already carry ICD-10-CM codings
via the SNOMED -> ICD-10 mapping in its output config).

Generator: https://github.com/synthetichealth/synthea
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


_ICD10_CM_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"


@dataclass
class SyntheaRecord:
    patient_id: str
    discharge_summary: str
    icd10_codes: list[str]


def _extract_patient_id(bundle: dict) -> str:
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            return resource.get("id", "")
    return ""


def _extract_icd10_codes(bundle: dict) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        for coding in resource.get("code", {}).get("coding", []):
            if coding.get("system") == _ICD10_CM_SYSTEM:
                code = coding.get("code", "").strip()
                if code and code not in seen:
                    seen.add(code)
                    codes.append(code)
    return codes


def _extract_discharge_summary(bundle: dict) -> str:
    """Concatenate Condition display texts as a proxy discharge narrative."""
    displays: list[str] = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        for coding in resource.get("code", {}).get("coding", []):
            display = coding.get("display", "").strip()
            if display:
                displays.append(display)
    return "; ".join(displays)


def load_bundles(root: Path | str) -> list[SyntheaRecord]:
    """Walk Synthea FHIR output and yield discharge encounters with ICD-10 codes."""
    root = Path(root)
    fhir_dir = root / "fhir"

    if not fhir_dir.exists():
        raise FileNotFoundError(
            f"Synthea FHIR directory not found: {fhir_dir}\n"
            "Generate data with: https://github.com/synthetichealth/synthea\n"
            "  java -jar synthea-with-dependencies.jar -p 100 --exporter.fhir.export true"
        )

    json_files = list(fhir_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No FHIR bundle JSON files found in {fhir_dir}\n"
            "Generate data with: https://github.com/synthetichealth/synthea\n"
            "  java -jar synthea-with-dependencies.jar -p 100 --exporter.fhir.export true"
        )

    records: list[SyntheaRecord] = []
    for path in json_files:
        bundle = json.loads(path.read_text())
        if bundle.get("resourceType") != "Bundle":
            continue
        patient_id = _extract_patient_id(bundle)
        icd10_codes = _extract_icd10_codes(bundle)
        if not icd10_codes:
            continue
        discharge_summary = _extract_discharge_summary(bundle)
        records.append(
            SyntheaRecord(
                patient_id=patient_id,
                discharge_summary=discharge_summary,
                icd10_codes=icd10_codes,
            )
        )
    return records
