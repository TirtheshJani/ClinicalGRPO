"""Prompt templates and output contract.

The output schema declared here is what `clinical_grpo.rewards.composite` parses.
If you change the contract, update the parser in lockstep or rewards will silently
return penalty scores for every generation.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert clinical coder. Read the discharge summary provided by the user "
    "and identify every relevant ICD-10-CM diagnosis code documented in the encounter. "
    "Respond with a single JSON object and no other text."
)

# The contract the reward parser depends on.
OUTPUT_SCHEMA_DESCRIPTION = (
    'Return a single JSON object of the form {"codes": ["E11.9", "I10", ...]} '
    "containing valid ICD-10-CM codes only. Do not include explanations, prose, "
    "markdown fences, or any text outside the JSON object."
)


def user_message(discharge_summary: str) -> str:
    """Render the user turn for a single discharge summary."""
    return (
        "Discharge summary:\n"
        "---\n"
        f"{discharge_summary.strip()}\n"
        "---\n\n"
        f"{OUTPUT_SCHEMA_DESCRIPTION}"
    )


def build_chat(discharge_summary: str) -> list[dict[str, str]]:
    """Build the chat-format messages list passed to the tokenizer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(discharge_summary)},
    ]
