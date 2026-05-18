"""Tests for app/app.py:predict() with Unsloth mocked out."""
import json
import sys
from unittest.mock import MagicMock, patch


def _make_app_predict(completion: str):
    """Return the predict() function with Unsloth and spaces mocked."""
    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "prompt text"
    # tokenizer(text, return_tensors="pt").to(device) must be chainable
    # and inputs["input_ids"].shape[1] must equal an integer
    input_ids_mock = MagicMock()
    input_ids_mock.shape = [1, 10]  # list so shape[1] == 10 works
    inputs_mock = MagicMock()
    inputs_mock.__getitem__ = lambda self, key: input_ids_mock if key == "input_ids" else MagicMock()
    inputs_mock.to.return_value = inputs_mock  # .to(device) returns itself
    mock_tok.return_value = inputs_mock
    mock_tok.decode.return_value = completion
    mock_model.generate.return_value = [MagicMock()]  # out[0][...] must be indexable
    mock_model.device = "cpu"

    mock_fl = MagicMock()
    mock_fl.from_pretrained.return_value = (mock_model, mock_tok)

    fake_unsloth = MagicMock()
    fake_unsloth.FastLanguageModel = mock_fl

    fake_spaces = MagicMock()
    fake_spaces.GPU = lambda fn: fn  # strip decorator transparently

    # Remove cached modules so a fresh import picks up the mocks
    for mod in list(sys.modules.keys()):
        if mod in ("app.app", "app"):
            del sys.modules[mod]

    fake_gradio = MagicMock()

    with patch.dict(sys.modules, {"unsloth": fake_unsloth, "spaces": fake_spaces, "gradio": fake_gradio}):
        sys.path.insert(0, "/home/user/ClinicalGRPO")
        import app.app as app_module
        # Bypass _load() so Unsloth is never actually called
        app_module._pipeline = (mock_model, mock_tok)
        return app_module.predict


def test_predict_returns_json_string():
    """predict() returns a JSON-parseable string."""
    predict = _make_app_predict('{"codes": ["E11.9"]}')
    result = predict("Patient with diabetes.")
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


def test_predict_includes_raw_and_parsed_codes():
    """predict() result has 'raw' and 'parsed_codes' keys with correct values."""
    predict = _make_app_predict('{"codes": ["E11.9", "I10"]}')
    result = json.loads(predict("note"))
    assert "raw" in result
    assert "parsed_codes" in result
    assert result["parsed_codes"] == ["E11.9", "I10"]


def test_predict_malformed_completion_returns_empty_codes():
    """When the model outputs non-JSON, parsed_codes is an empty list."""
    predict = _make_app_predict("I cannot determine the codes.")
    result = json.loads(predict("note"))
    assert result["parsed_codes"] == []
