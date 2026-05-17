import pytest
from clinical_grpo.eval.bootstrap import bootstrap_f1_ci, BootstrapResult  # noqa: F401


def _perfect(n):
    codes = [["E11.9", "I10"]] * n
    return codes, codes


def _half_recall(n):
    # Half of samples are perfect hits, half are complete misses.
    # Micro F1 ≈ 0.667; bootstrap resamples have non-zero variance.
    half = n // 2
    pred = [["E11.9", "I10"]] * half + [[]] * half
    gold = [["E11.9", "I10"]] * n
    return pred, gold


def test_result_has_fields():
    preds, golds = _perfect(10)
    r = bootstrap_f1_ci(preds, golds, n_resamples=50, seed=0)
    assert hasattr(r, "mean_f1")
    assert hasattr(r, "ci_low")
    assert hasattr(r, "ci_high")
    assert hasattr(r, "n_resamples")


def test_perfect_predictions_mean_is_one():
    preds, golds = _perfect(20)
    r = bootstrap_f1_ci(preds, golds, n_resamples=200, seed=42)
    assert r.mean_f1 == pytest.approx(1.0, abs=1e-9)
    assert r.ci_low > 0.95
    assert r.ci_high == pytest.approx(1.0, abs=1e-9)


def test_half_recall_ci_brackets_true_f1():
    # true F1 ≈ 0.667 (precision=1.0, recall=0.5)
    preds, golds = _half_recall(50)
    r = bootstrap_f1_ci(preds, golds, n_resamples=500, seed=42)
    assert r.ci_low < 0.667 < r.ci_high


def test_ci_ordering():
    preds = [[f"E{i:02d}.9"] for i in range(30)]
    golds = [[f"E{i:02d}.9", "I10"] for i in range(30)]
    r = bootstrap_f1_ci(preds, golds, n_resamples=200, seed=1)
    assert r.ci_low <= r.mean_f1 <= r.ci_high


def test_n_resamples_stored():
    preds, golds = _perfect(5)
    r = bootstrap_f1_ci(preds, golds, n_resamples=77, seed=0)
    assert r.n_resamples == 77
