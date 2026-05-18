"""Tests for clinical_grpo.data.synthea.load_bundles."""

from __future__ import annotations

import json

import pytest

from clinical_grpo.data.synthea import load_bundles


def _write_bundle(directory, filename: str, bundle: dict) -> None:
    fhir_dir = directory / "fhir"
    fhir_dir.mkdir(exist_ok=True)
    (fhir_dir / filename).write_text(json.dumps(bundle))


def _diabetes_bundle() -> dict:
    return {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pt-001"}},
            {
                "resource": {
                    "resourceType": "Condition",
                    "subject": {"reference": "Patient/pt-001"},
                    "code": {
                        "coding": [
                            {
                                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                                "code": "E11.9",
                                "display": "Type 2 diabetes mellitus",
                            }
                        ]
                    },
                }
            },
        ],
    }


def test_load_bundles_returns_one_record(tmp_path):
    _write_bundle(tmp_path, "patient1.json", _diabetes_bundle())
    records = load_bundles(tmp_path)
    assert len(records) == 1


def test_record_has_correct_patient_id(tmp_path):
    _write_bundle(tmp_path, "patient1.json", _diabetes_bundle())
    records = load_bundles(tmp_path)
    assert records[0].patient_id == "pt-001"


def test_record_has_correct_icd10_codes(tmp_path):
    _write_bundle(tmp_path, "patient1.json", _diabetes_bundle())
    records = load_bundles(tmp_path)
    assert records[0].icd10_codes == ["E11.9"]


def test_discharge_summary_contains_display_text(tmp_path):
    _write_bundle(tmp_path, "patient1.json", _diabetes_bundle())
    records = load_bundles(tmp_path)
    assert "Type 2 diabetes mellitus" in records[0].discharge_summary


def test_raises_file_not_found_when_fhir_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundles(tmp_path)


def test_bundles_with_no_icd10_codes_are_skipped(tmp_path):
    bundle_no_codes = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pt-002"}},
        ],
    }
    _write_bundle(tmp_path, "patient_no_codes.json", bundle_no_codes)
    records = load_bundles(tmp_path)
    assert len(records) == 0
