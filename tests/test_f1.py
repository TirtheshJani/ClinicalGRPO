from clinical_grpo.eval.f1 import F1Result, chapter_f1, micro_code_f1


def test_perfect_prediction_f1_is_one():
    result = micro_code_f1([["E11.9", "I10"]], [["E11.9", "I10"]])
    assert result.f1 == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_perfect_prediction_counts():
    result = micro_code_f1([["E11.9", "I10"]], [["E11.9", "I10"]])
    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0


def test_empty_prediction_recall_zero():
    result = micro_code_f1([[]], [["E11.9", "I10"]])
    assert result.recall == 0.0


def test_empty_prediction_precision_zero():
    result = micro_code_f1([[]], [["E11.9", "I10"]])
    assert result.precision == 0.0


def test_empty_prediction_f1_zero():
    result = micro_code_f1([[]], [["E11.9", "I10"]])
    assert result.f1 == 0.0


def test_partial_match_single_sample():
    result = micro_code_f1([["E11.9"]], [["E11.9", "I10"]])
    assert result.tp == 1
    assert result.fn == 1
    assert result.fp == 0


def test_micro_averaging_across_samples():
    preds = [["E11.9"], ["I10", "N17.9"]]
    golds = [["E11.9", "I10"], ["I10"]]
    result = micro_code_f1(preds, golds)
    assert result.tp == 2
    assert result.fp == 1
    assert result.fn == 1


def test_micro_averaging_f1_value():
    preds = [["E11.9"], ["I10", "N17.9"]]
    golds = [["E11.9", "I10"], ["I10"]]
    result = micro_code_f1(preds, golds)
    prec = 2 / 3
    rec = 2 / 3
    expected_f1 = 2 * prec * rec / (prec + rec)
    assert abs(result.f1 - expected_f1) < 1e-9


def test_chapter_f1_gives_chapter_credit():
    # E11.9 (chapter IV) and E78.5 (chapter IV) share chapter
    result = chapter_f1([["E78.5"]], [["E11.9"]])
    assert result.f1 == 1.0


def test_chapter_f1_no_credit_different_chapters():
    # E11.9 (chapter IV) vs I10 (chapter IX)
    result = chapter_f1([["I10"]], [["E11.9"]])
    assert result.f1 == 0.0


def test_chapter_f1_perfect_same_codes():
    result = chapter_f1([["E11.9", "I10"]], [["E11.9", "I10"]])
    assert result.f1 == 1.0


def test_f1result_is_dataclass():
    r = micro_code_f1([["I10"]], [["I10"]])
    assert hasattr(r, "precision")
    assert hasattr(r, "recall")
    assert hasattr(r, "f1")
    assert hasattr(r, "tp")
    assert hasattr(r, "fp")
    assert hasattr(r, "fn")


def test_empty_both_sides():
    result = micro_code_f1([[]], [[]])
    assert result.f1 == 0.0
    assert result.tp == 0
