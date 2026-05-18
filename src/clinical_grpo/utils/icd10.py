"""ICD-10-CM normalization, validation, and chapter lookup.

Single source of truth for what counts as a valid ICD-10 code and which codes
share a chapter. Both reward functions depend on this module — keep it pure
(no I/O, no model imports) so reward unit tests stay fast.
"""

from __future__ import annotations

import re

# ICD-10-CM chapter boundaries (inclusive). Each entry: (start, end, chapter_roman).
# Source: WHO ICD-10-CM 2024.
_CHAPTER_RANGES: list[tuple[str, str, str]] = [
    ("A00", "B99", "I"),       # Infectious and parasitic diseases
    ("C00", "D49", "II"),      # Neoplasms
    ("D50", "D89", "III"),     # Blood and immune disorders
    ("E00", "E89", "IV"),      # Endocrine, nutritional, metabolic
    ("F01", "F99", "V"),       # Mental, behavioral, neurodevelopmental
    ("G00", "G99", "VI"),      # Nervous system
    ("H00", "H59", "VII"),     # Eye and adnexa
    ("H60", "H95", "VIII"),    # Ear and mastoid
    ("I00", "I99", "IX"),      # Circulatory
    ("J00", "J99", "X"),       # Respiratory
    ("K00", "K95", "XI"),      # Digestive
    ("L00", "L99", "XII"),     # Skin and subcutaneous
    ("M00", "M99", "XIII"),    # Musculoskeletal
    ("N00", "N99", "XIV"),     # Genitourinary
    ("O00", "O9A", "XV"),      # Pregnancy, childbirth, puerperium
    ("P00", "P96", "XVI"),     # Perinatal conditions
    ("Q00", "Q99", "XVII"),    # Congenital malformations
    ("R00", "R99", "XVIII"),   # Symptoms, signs, abnormal findings
    ("S00", "T88", "XIX"),     # Injury, poisoning, external causes
    ("V00", "Y99", "XX"),      # External causes of morbidity
    ("Z00", "Z99", "XXI"),     # Factors influencing health status
    ("U00", "U85", "XXII"),    # Codes for special purposes
]

# Matches a 3-7 character ICD-10-CM code: letter + 2 digits, optional dot + up to 4 alphanumerics.
_ICD10_RE = re.compile(r"^[A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")


def normalize(code: str) -> str:
    """Normalize an ICD-10 code string for comparison.

    - Strip whitespace, uppercase.
    - Insert a dot after the third character if missing and length > 3.
    - Does not validate; pair with `is_valid` if you need a real code.
    """
    s = code.strip().upper().replace(" ", "")
    if "." not in s and len(s) > 3:
        s = s[:3] + "." + s[3:]
    return s


def is_valid(code: str) -> bool:
    """True iff `code` looks like a well-formed ICD-10-CM code.

    Rejects ICD-9 (numeric-only) codes intentionally — earning chapter credit
    for unrecognizable strings would be a free reward-hacking exploit.
    """
    return bool(_ICD10_RE.match(normalize(code)))


def _prefix(code: str) -> str:
    """First three characters (category) used for chapter lookup."""
    return normalize(code)[:3]


def chapter_of(code: str) -> str | None:
    """Return the Roman-numeral chapter for an ICD-10 code, or None if unknown."""
    if not is_valid(code):
        return None
    p = _prefix(code)
    for start, end, ch in _CHAPTER_RANGES:
        if start <= p <= end:
            return ch
    return None


def dedupe_codes(codes: list[str]) -> list[str]:
    """Normalize and deduplicate a list of codes, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        n = normalize(c)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out
