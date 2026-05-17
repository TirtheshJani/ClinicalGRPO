import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.validate_data import ValidationReport, validate_parquet  # noqa: F401


def _parquet(tmp_path, summaries, codes):
    p = tmp_path / "d.parquet"
    pd.DataFrame({"discharge_summary": summaries, "icd10_codes": codes}).to_parquet(p)
    return p


def test_clean_data_no_issues(tmp_path):
    p = _parquet(tmp_path, ["Diabetes with HTN.", "Heart failure."], [["E11.9", "I10"], ["I50.9"]])
    r = validate_parquet(p)
    assert r.n_rows == 2
    assert r.n_empty_summaries == 0
    assert r.n_invalid_codes == 0
    assert r.n_empty_code_lists == 0


def test_detects_empty_summary(tmp_path):
    p = _parquet(tmp_path, ["", "Real note."], [["I10"], ["E11.9"]])
    r = validate_parquet(p)
    assert r.n_empty_summaries == 1


def test_detects_invalid_icd9_code(tmp_path):
    p = _parquet(tmp_path, ["Note one.", "Note two."], [["41401", "I10"], ["E11.9"]])
    r = validate_parquet(p)
    assert r.n_invalid_codes == 1   # "41401" is ICD-9, rejected by is_valid


def test_detects_empty_code_list(tmp_path):
    p = _parquet(tmp_path, ["Note.", "Note."], [[], ["E11.9"]])
    r = validate_parquet(p)
    assert r.n_empty_code_lists == 1


def test_summary_length_stats_are_ordered(tmp_path):
    p = _parquet(tmp_path, ["Short.", "A " * 200], [["I10"], ["E11.9"]])
    r = validate_parquet(p)
    assert r.summary_len_min < r.summary_len_max
    assert r.summary_len_min <= r.summary_len_mean <= r.summary_len_max
