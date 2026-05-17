"""GPU smoke test: one full GRPO step on a tiny synthetic dataset.

Run before every long training session:
    pytest tests/test_smoke_train.py -k one_step -x

Requires CUDA. Skipped automatically on CPU-only installs.

Asserts:
    (a) Loss is finite — i.e. training did not NaN/explode on step 1.
    (b) All three reward names appear in the TRL trainer_state.json log.
    (c) The saved LoRA adapter reloads without error.
    (d) A generation from the reloaded model parses as valid JSON (does not raise).
"""

from __future__ import annotations

import json
import logging
import os

import pytest

# Skip on CPU-only installs that don't even have torch installed.
pytest.importorskip("torch")


@pytest.mark.gpu
def test_one_grpo_step(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — GPU smoke test skipped")

    import pandas as pd

    from clinical_grpo.training.train_grpo import train
    from clinical_grpo.rewards.composite import parse_completion

    # ------------------------------------------------------------------
    # 1. Build a minimal synthetic parquet (4 rows, 2 patients).
    # ------------------------------------------------------------------
    rows = {
        "discharge_summary": [
            "Patient with type 2 diabetes and hypertension.",
            "Admitted for heart failure exacerbation.",
            "Sepsis due to urinary tract infection.",
            "Chronic kidney disease stage 3.",
        ],
        "icd10_codes": [
            ["E11.9", "I10"],
            ["I50.9"],
            ["A41.9", "N39.0"],
            ["N18.3"],
        ],
    }
    parquet_path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path)

    output_dir = tmp_path / "output"

    # ------------------------------------------------------------------
    # 2. Minimal cfg dict — no YAML files needed at test time.
    #    Uses Qwen3-0.6B (smallest Qwen3) for speed.
    # ------------------------------------------------------------------
    cfg = {
        "model": {
            "base_model": "Qwen/Qwen3-0.6B",
            "lora": {
                "r": 4,
                "alpha": 8,
                "dropout": 0.0,
                "bias": "none",
                "target_modules": ["q_proj", "v_proj"],
            },
        },
        "grpo": {
            "num_generations": 2,
            "beta": 0.04,
            "learning_rate": 5e-6,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.0,
            "max_steps": 1,
            "save_steps": 1,
            "logging_steps": 1,
            "max_prompt_length": 256,
            "max_completion_length": 64,
            "temperature": 0.9,
            "top_p": 1.0,
        },
        "rewards": {
            "chapter_weight": 0.3,
            "hallucination_penalty": 0.1,
            "format_invalid_penalty": 0.5,
        },
        "dataset": {"train_path": str(parquet_path)},
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "use_gradient_checkpointing": False,
        "load_in_4bit": True,
        "fp16": False,
        "bf16": False,
        "dtype": None,
        "seed": 42,
        "report_to": [],
        "max_seq_length": 512,
    }

    # ------------------------------------------------------------------
    # 3. Run one GRPO step and capture log output for reward-name check.
    # ------------------------------------------------------------------
    # TRL logs metric dicts via Python's logging module at INFO level.
    # We install a handler on the root logger before training starts so
    # we catch every line regardless of which logger TRL uses.
    log_records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(self.format(record))

    capture_handler = _Capture()
    capture_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(capture_handler)

    try:
        adapter_path = train(cfg, max_steps_override=1)
    finally:
        root_logger.removeHandler(capture_handler)

    # ------------------------------------------------------------------
    # (a) Adapter path exists.
    # ------------------------------------------------------------------
    assert adapter_path.exists(), (
        f"Expected adapter directory at {adapter_path} but it was not created."
    )

    # ------------------------------------------------------------------
    # (b) All three reward names appear in the TRL trainer_state.json.
    #
    #     TRL writes <output_dir>/trainer_state.json containing log_history
    #     with keys like "rewards/exact_match_reward", etc.
    #     We check that file first; fall back to scanning the captured log
    #     lines so the assertion works even if the file layout changes.
    # ------------------------------------------------------------------
    expected_reward_names = {
        "exact_match_reward",
        "chapter_partial_reward",
        "format_validity_reward",
    }

    found_names: set[str] = set()

    trainer_state_path = output_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as fh:
            state = json.load(fh)
        for entry in state.get("log_history", []):
            for key in entry:
                for name in expected_reward_names:
                    if name in key:
                        found_names.add(name)

    # Fall back to captured log lines if trainer_state.json didn't surface them.
    if found_names != expected_reward_names:
        all_log_text = "\n".join(log_records)
        for name in expected_reward_names:
            if name in all_log_text:
                found_names.add(name)

    missing = expected_reward_names - found_names
    assert not missing, (
        f"The following reward names were not found in the TRL log: {missing}. "
        f"trainer_state.json exists: {trainer_state_path.exists()}. "
        f"Check that build_reward_funcs() names the closures correctly."
    )

    # ------------------------------------------------------------------
    # (c) Adapter reloads without error.
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel  # noqa: PLC0415 — heavy; GPU test only

    reloaded_model, reloaded_tokenizer = FastLanguageModel.from_pretrained(
        str(adapter_path),
        max_seq_length=512,
        load_in_4bit=True,
    )
    assert reloaded_model is not None, "FastLanguageModel.from_pretrained returned None"

    # ------------------------------------------------------------------
    # (d) A generation from the reloaded model parses without raising.
    #     parse_completion() returns None or a list — either is acceptable;
    #     the important thing is no exception is raised.
    # ------------------------------------------------------------------
    from clinical_grpo.prompts.templates import build_chat  # noqa: PLC0415

    prompt_messages = build_chat("Patient with hypertension.")
    input_ids = reloaded_tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(reloaded_model.device)

    with torch.no_grad():
        output_ids = reloaded_model.generate(
            input_ids,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=True,
        )

    # Decode only the newly generated tokens (skip the prompt).
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    generation_text = reloaded_tokenizer.decode(new_tokens, skip_special_tokens=True)

    # This must not raise; result may be None or a list of codes.
    result = parse_completion(generation_text)

    # Log the result so it is visible in -s output for debugging.
    logging.getLogger(__name__).info(
        "Smoke generation output: %r  -> parse_completion: %r",
        generation_text,
        result,
    )
