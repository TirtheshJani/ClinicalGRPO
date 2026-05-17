"""GPU smoke test: one full SFT step on a tiny synthetic dataset.

Run before every long SFT training session:
    pytest tests/test_smoke_sft.py -k one_step -x

Requires CUDA. Skipped automatically on CPU-only installs.

Asserts:
    (a) adapter directory is created.
    (b) adapter_config.json exists inside the adapter directory (valid LoRA adapter).
    (c) trainer_state.json contains a log_history entry with a "loss" key.
    (d) The saved LoRA adapter reloads without error.
    (e) A generation from the reloaded model does not crash (SFT output format
        is less constrained than GRPO — we do not require valid JSON here).
"""

from __future__ import annotations

import json
import logging

import pytest

# Skip on CPU-only installs that don't even have torch installed.
pytest.importorskip("torch")


@pytest.mark.gpu
def test_one_sft_step(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — GPU smoke test skipped")

    import pandas as pd

    from clinical_grpo.training.train_sft import train_sft

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
    #    SFT trainer reads from cfg["sft"] instead of cfg["grpo"].
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
        "sft": {
            "max_steps": 1,
            "learning_rate": 2e-5,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.0,
            "save_steps": 1,
            "logging_steps": 1,
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
        "max_seq_length": 256,
    }

    # ------------------------------------------------------------------
    # 3. Run one SFT step.
    # ------------------------------------------------------------------
    adapter_path = train_sft(cfg, max_steps_override=1)

    # ------------------------------------------------------------------
    # (a) Adapter directory was created.
    # ------------------------------------------------------------------
    assert adapter_path.exists(), (
        f"Expected adapter directory at {adapter_path} but it was not created."
    )

    # ------------------------------------------------------------------
    # (b) adapter_config.json exists — confirms a valid LoRA adapter was saved.
    # ------------------------------------------------------------------
    adapter_config_path = adapter_path / "adapter_config.json"
    assert adapter_config_path.exists(), (
        f"Expected {adapter_config_path} but it was not found. "
        f"trainer.save_model() may have failed or saved to the wrong path."
    )

    # ------------------------------------------------------------------
    # (c) trainer_state.json contains a log_history entry with a "loss" key.
    #
    #     train_sft() uses the output_dir from cfg (not "grpo"/"sft" mangled),
    #     so trainer_state.json lives under that directory.
    # ------------------------------------------------------------------
    # Resolve the actual output directory: train_sft replaces "grpo" -> "sft"
    # in cfg["output_dir"]; since our output_dir has neither, it falls back to
    # "outputs/sft". Adapter is always saved at <output_dir>/adapter, so we
    # derive the trainer_state path from adapter_path's parent.
    trainer_state_path = adapter_path.parent / "trainer_state.json"
    assert trainer_state_path.exists(), (
        f"Expected trainer_state.json at {trainer_state_path}. "
        f"TRL SFTTrainer should write this automatically."
    )

    with open(trainer_state_path) as fh:
        state = json.load(fh)

    log_history = state.get("log_history", [])
    loss_found = any("loss" in entry for entry in log_history)
    assert loss_found, (
        f"No 'loss' key found in any log_history entry. "
        f"log_history contents: {log_history}"
    )

    # ------------------------------------------------------------------
    # (d) Adapter reloads without error.
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel  # noqa: PLC0415 — heavy; GPU test only

    reloaded_model, reloaded_tokenizer = FastLanguageModel.from_pretrained(
        str(adapter_path),
        max_seq_length=256,
        load_in_4bit=True,
    )
    assert reloaded_model is not None, "FastLanguageModel.from_pretrained returned None"

    # ------------------------------------------------------------------
    # (e) A generation from the reloaded model does not crash.
    #     SFT output format is less constrained than GRPO — we do not
    #     require valid JSON; just that generation completes without error.
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
            max_new_tokens=32,
            temperature=0.1,
            do_sample=True,
        )

    # Decode only the newly generated tokens (skip the prompt).
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    generation_text = reloaded_tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Log the result so it is visible in -s output for debugging.
    logging.getLogger(__name__).info(
        "SFT smoke generation output: %r",
        generation_text,
    )
