import json
import sys
from unittest.mock import MagicMock, patch

from clinical_grpo.inference import ICD10Predictor


def _predictor(tmp_path, completions):
    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "prompt"
    mock_tok.return_value = {"input_ids": MagicMock(shape=[1, 10])}
    mock_tok.decode.side_effect = list(completions)
    mock_model.generate.return_value = MagicMock()
    mock_model.device = "cpu"
    fake_fl = MagicMock()
    fake_fl.from_pretrained.return_value = (mock_model, mock_tok)
    fake_unsloth = MagicMock()
    fake_unsloth.FastLanguageModel = fake_fl
    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        p = ICD10Predictor(str(tmp_path / "adapter"), load_in_4bit=False)
    p._model = mock_model
    p._tokenizer = mock_tok
    return p


def test_predict_returns_list_of_strings(tmp_path):
    p = _predictor(tmp_path, [json.dumps({"codes": ["E11.9", "I10"]})])
    result = p.predict("Patient with diabetes.")
    assert isinstance(result, list)
    assert all(isinstance(c, str) for c in result)


def test_predict_parses_codes(tmp_path):
    p = _predictor(tmp_path, [json.dumps({"codes": ["E11.9", "I10"]})])
    assert p.predict("note") == ["E11.9", "I10"]


def test_predict_returns_empty_on_malformed_json(tmp_path):
    p = _predictor(tmp_path, ["not json"])
    assert p.predict("note") == []


def test_predict_batch_returns_one_list_per_summary(tmp_path):
    completions = [json.dumps({"codes": ["E11.9"]}), json.dumps({"codes": ["I10"]})]
    p = _predictor(tmp_path, completions)
    results = p.predict_batch(["note one", "note two"])
    assert len(results) == 2
    assert results[0] == ["E11.9"]
    assert results[1] == ["I10"]


def test_adapter_path_stored(tmp_path):
    p = _predictor(tmp_path, [])
    assert p.adapter_path == str(tmp_path / "adapter")
