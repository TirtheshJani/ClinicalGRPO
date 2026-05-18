# tests/test_think_strip.py
from clinical_grpo.rewards.composite import parse_completion


def test_no_think_block_unchanged():
    """Baseline: still works without any <think> block."""
    assert parse_completion('{"codes": ["E11.9"]}') == ["E11.9"]


def test_think_block_before_json():
    """<think> before JSON: codes extracted correctly."""
    text = "<think>Some reasoning here.</think>\n{\"codes\": [\"E11.9\", \"I10\"]}"
    assert parse_completion(text) == ["E11.9", "I10"]


def test_think_block_contains_inner_json():
    """Inner JSON inside <think> must NOT be used; only the outer JSON counts."""
    text = '<think>{"wrong": "json"}</think>{"codes": ["Z38.00"]}'
    assert parse_completion(text) == ["Z38.00"]


def test_think_block_only_no_outer_json():
    """<think> block with no JSON outside returns None."""
    text = "<think>I think the code is E11.9.</think>"
    assert parse_completion(text) is None


def test_multiline_think_block():
    """Multi-line thinking block followed by JSON."""
    text = (
        "<think>\n"
        "Line one of reasoning.\n"
        "Line two.\n"
        "</think>\n"
        '{"codes": ["N17.9"]}'
    )
    assert parse_completion(text) == ["N17.9"]
