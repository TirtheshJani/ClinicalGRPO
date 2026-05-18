import pandas as pd

from clinical_grpo.data.preprocess import MAX_CHARS, clean_summary, patient_split


def test_clean_summary_strips_phi_open_token():
    assert "[**" not in clean_summary("Hello [** Name **] world")


def test_clean_summary_strips_phi_close_token():
    assert "**]" not in clean_summary("Hello [** Name **] world")


def test_clean_summary_collapses_whitespace():
    result = clean_summary("word1   word2\tword3\nword4")
    assert "  " not in result
    assert result == "word1 word2 word3 word4"


def test_clean_summary_truncates_to_max_chars():
    long_text = "x" * (MAX_CHARS + 500)
    result = clean_summary(long_text)
    assert len(result) <= MAX_CHARS


def test_clean_summary_preserves_content_under_limit():
    text = "Patient admitted for chest pain and hypertension."
    assert clean_summary(text) == text


def test_clean_summary_strips_phi_and_collapses():
    result = clean_summary("[** Dr Smith **]  admitted  patient")
    assert result == "Dr Smith admitted patient"


def _make_df(n_patients: int, rows_per_patient: int = 2) -> pd.DataFrame:
    records = []
    for pid in range(n_patients):
        for _ in range(rows_per_patient):
            records.append({"patient_id": f"P{pid:03d}", "text": "note"})
    return pd.DataFrame(records)


def test_patient_split_returns_three_keys():
    df = _make_df(10)
    splits = patient_split(df, "patient_id")
    assert set(splits.keys()) == {"train", "val", "test"}


def test_patient_split_train_is_largest():
    df = _make_df(10)
    splits = patient_split(df, "patient_id")
    assert len(splits["train"]) >= len(splits["val"])
    assert len(splits["train"]) >= len(splits["test"])


def test_patient_split_all_rows_accounted_for():
    df = _make_df(10)
    splits = patient_split(df, "patient_id")
    total = sum(len(v) for v in splits.values())
    assert total == len(df)


def test_patient_split_train_disjoint_from_val():
    df = _make_df(10)
    splits = patient_split(df, "patient_id")
    train_patients = set(splits["train"]["patient_id"])
    val_patients = set(splits["val"]["patient_id"])
    assert train_patients.isdisjoint(val_patients)


def test_patient_split_train_disjoint_from_test():
    df = _make_df(10)
    splits = patient_split(df, "patient_id")
    train_patients = set(splits["train"]["patient_id"])
    test_patients = set(splits["test"]["patient_id"])
    assert train_patients.isdisjoint(test_patients)


def test_patient_split_val_disjoint_from_test():
    df = _make_df(10)
    splits = patient_split(df, "patient_id")
    val_patients = set(splits["val"]["patient_id"])
    test_patients = set(splits["test"]["patient_id"])
    assert val_patients.isdisjoint(test_patients)


def test_patient_split_default_ratios_approx():
    df = _make_df(10)
    splits = patient_split(df, "patient_id")
    n_patients = 10
    n_train_patients = len(splits["train"]["patient_id"].unique())
    assert n_train_patients == int(n_patients * 0.8)


def test_patient_split_deterministic_with_seed():
    df = _make_df(8)
    s1 = patient_split(df, "patient_id", seed=99)
    s2 = patient_split(df, "patient_id", seed=99)
    assert list(s1["train"]["patient_id"]) == list(s2["train"]["patient_id"])


def test_patient_split_different_seeds_differ():
    df = _make_df(10)
    s1 = patient_split(df, "patient_id", seed=1)
    s2 = patient_split(df, "patient_id", seed=2)
    assert list(s1["train"]["patient_id"]) != list(s2["train"]["patient_id"])
