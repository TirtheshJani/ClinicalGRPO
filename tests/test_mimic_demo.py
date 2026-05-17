"""Tests for clinical_grpo.data.mimic_demo._load_gem and _icd9_to_icd10."""

from __future__ import annotations

import clinical_grpo.data.mimic_demo as mod


def test_load_gem_returns_empty_dict_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_GEM_PATH", tmp_path / "nonexistent.txt")
    monkeypatch.setattr(mod, "_gem_map", None)
    result = mod._load_gem()
    assert result == {}


def test_icd9_to_icd10_returns_input_unchanged_when_gem_empty(monkeypatch):
    monkeypatch.setattr(mod, "_gem_map", {})
    codes = ["41401", "4019"]
    result = mod._icd9_to_icd10(codes)
    assert result == codes


def test_gem_parsing(tmp_path, monkeypatch):
    gem_content = "41401 I2510 10888\n4019 I10 10888\n00321 A1811 00000\n"
    gem_file = tmp_path / "gem.txt"
    gem_file.write_text(gem_content)
    monkeypatch.setattr(mod, "_GEM_PATH", gem_file)
    monkeypatch.setattr(mod, "_gem_map", None)
    result = mod._load_gem()
    assert "41401" in result
    assert "4019" in result


def test_gem_parsed_values_are_icd10_codes(tmp_path, monkeypatch):
    gem_content = "41401 I2510 10888\n4019 I10 10888\n"
    gem_file = tmp_path / "gem.txt"
    gem_file.write_text(gem_content)
    monkeypatch.setattr(mod, "_GEM_PATH", gem_file)
    monkeypatch.setattr(mod, "_gem_map", None)
    result = mod._load_gem()
    assert "I2510" in result["41401"]
    assert "I10" in result["4019"]


def test_icd9_to_icd10_with_patched_gem(tmp_path, monkeypatch):
    gem_content = "41401 I2510 10888\n4019 I10 10888\n"
    gem_file = tmp_path / "gem.txt"
    gem_file.write_text(gem_content)
    monkeypatch.setattr(mod, "_GEM_PATH", gem_file)
    monkeypatch.setattr(mod, "_gem_map", None)
    result = mod._icd9_to_icd10(["41401", "4019"])
    assert "I2510" in result
    assert "I10" in result


def test_icd9_to_icd10_drops_unknown_codes(tmp_path, monkeypatch):
    gem_content = "41401 I2510 10888\n"
    gem_file = tmp_path / "gem.txt"
    gem_file.write_text(gem_content)
    monkeypatch.setattr(mod, "_GEM_PATH", gem_file)
    monkeypatch.setattr(mod, "_gem_map", None)
    result = mod._icd9_to_icd10(["41401", "99999"])
    assert result == ["I2510"]
