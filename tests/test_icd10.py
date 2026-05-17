"""Unit tests for ICD-10 normalization, validation, and chapter lookup."""

from clinical_grpo.utils.icd10 import chapter_of, dedupe_codes, is_valid, normalize


def test_normalize_inserts_dot():
    assert normalize("e119") == "E11.9"
    assert normalize("E11.9") == "E11.9"
    assert normalize("  i10 ") == "I10"


def test_normalize_uppercases():
    assert normalize("n17.9") == "N17.9"


def test_is_valid_accepts_well_formed():
    assert is_valid("E11.9")
    assert is_valid("I10")
    assert is_valid("S72.001A")


def test_is_valid_rejects_icd9():
    # ICD-9 codes are numeric-only; must not be accepted.
    assert not is_valid("250.00")
    assert not is_valid("4019")


def test_is_valid_rejects_garbage():
    assert not is_valid("")
    assert not is_valid("XYZ")
    assert not is_valid("E11.999999")


def test_chapter_of_known_codes():
    assert chapter_of("E11.9") == "IV"      # Endocrine
    assert chapter_of("I10") == "IX"        # Circulatory
    assert chapter_of("N17.9") == "XIV"     # Genitourinary
    assert chapter_of("A41.9") == "I"       # Infectious


def test_chapter_of_invalid_returns_none():
    assert chapter_of("250.00") is None
    assert chapter_of("nonsense") is None


def test_dedupe_preserves_order_and_normalizes():
    assert dedupe_codes(["e119", "I10", "E11.9", "i10"]) == ["E11.9", "I10"]
