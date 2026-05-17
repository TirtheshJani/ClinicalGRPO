"""Unit tests for clinical_grpo.eval.groq_judge. No real API calls are made."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from clinical_grpo.eval.groq_judge import _cache_key, GroqJudge


# ---------------------------------------------------------------------------
# _cache_key helpers
# ---------------------------------------------------------------------------


def test_cache_key_returns_hex_string() -> None:
    key = _cache_key(["E11.9"], ["I10"])
    assert isinstance(key, str)
    # sha256 hex digest is always 64 characters
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_cache_key_same_inputs_same_key() -> None:
    k1 = _cache_key(["E11.9", "I10"], ["N17.9"])
    k2 = _cache_key(["E11.9", "I10"], ["N17.9"])
    assert k1 == k2


def test_cache_key_order_independent() -> None:
    # Gold and pred lists are sorted internally, so order must not matter.
    k1 = _cache_key(["E11.9", "I10"], ["N17.9", "Z99.0"])
    k2 = _cache_key(["I10", "E11.9"], ["Z99.0", "N17.9"])
    assert k1 == k2


def test_cache_key_different_inputs_different_key() -> None:
    k1 = _cache_key(["E11.9"], ["I10"])
    k2 = _cache_key(["E11.9"], ["N17.9"])
    assert k1 != k2


# ---------------------------------------------------------------------------
# GroqJudge.score — cache behaviour
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"matched": 1, "near_matches": 0, "spurious": 0, "missed": 0}'
    )
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_groq_cache_hit_skips_api(tmp_path: Path) -> None:
    mock_client = _make_mock_client()

    with patch("groq.Groq", return_value=mock_client):
        judge = GroqJudge(api_key="fake", cache_dir=tmp_path)
        v1 = judge.score(["E11.9"], ["E11.9"])
        v2 = judge.score(["E11.9"], ["E11.9"])  # must hit disk cache

    assert mock_client.chat.completions.create.call_count == 1
    assert v1.matched == 1
    assert v2.matched == 1


def test_groq_score_returns_judge_verdict_fields(tmp_path: Path) -> None:
    mock_client = _make_mock_client()

    with patch("groq.Groq", return_value=mock_client):
        judge = GroqJudge(api_key="fake", cache_dir=tmp_path)
        verdict = judge.score(["E11.9"], ["E11.9"])

    assert verdict.matched == 1
    assert verdict.near_matches == 0
    assert verdict.spurious == 0
    assert verdict.missed == 0


def test_groq_different_inputs_call_api_twice(tmp_path: Path) -> None:
    mock_client = _make_mock_client()

    with patch("groq.Groq", return_value=mock_client):
        judge = GroqJudge(api_key="fake", cache_dir=tmp_path)
        judge.score(["E11.9"], ["E11.9"])
        judge.score(["I10"], ["I10"])  # different pair — must call API again

    assert mock_client.chat.completions.create.call_count == 2
