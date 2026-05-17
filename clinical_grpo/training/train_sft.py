"""SFT training entrypoint: Unsloth FastLanguageModel + TRL SFTTrainer.

Mirrors train_grpo.py in structure. No reward functions — supervised fine-tuning
on (prompt + gold completion) pairs where the target completion is the gold
JSON: {"codes": ["E11.9", "I10", ...]}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinical_grpo.data.dataset import load_grpo_dataset
from clinical_grpo.prompts.templates import build_chat


def _build_sft_dataset(parquet_path: str, tokenizer: Any) -> Any:
    from datasets import Dataset

    raw = load_grpo_dataset(parquet_path)

    def _to_text(example: dict) -> dict:
        messages = list(example["prompt"]) + [
            {
                "role": "assistant",
                "content": json.dumps({"codes": example["gold_codes"]}),
            }
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    return Dataset.from_dict(
        {"text": [_to_text(ex)["text"] for ex in raw]}
    )


def train_sft(cfg: dict[str, Any], max_steps_override: int | None = None) -> Path:
    """Run SFT training. Returns path to saved LoRA adapter."""
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model_cfg = cfg["model"]
    sft = cfg.get("sft", {})

    max_steps = max_steps_override if max_steps_override is not None else sft.get("max_steps", 500)

    raw_output_dir = cfg.get("output_dir", "outputs/grpo")
    if "grpo" in raw_output_dir:
        output_dir = raw_output_dir.replace("grpo", "sft")
    else:
        output_dir = "outputs/sft"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=cfg.get("max_seq_length", 2048),
        load_in_4bit=cfg.get("load_in_4bit", True),
        dtype=cfg.get("dtype"),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_cfg["lora"]["r"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"]["dropout"],
        bias=model_cfg["lora"]["bias"],
        target_modules=model_cfg["lora"]["target_modules"],
        use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", "unsloth"),
    )

    train_ds = _build_sft_dataset(cfg["dataset"]["train_path"], tokenizer)

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=sft.get("learning_rate", 2e-5),
        lr_scheduler_type=sft.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft.get("warmup_ratio", 0.03),
        save_steps=sft.get("save_steps", 100),
        logging_steps=sft.get("logging_steps", 5),
        max_seq_length=cfg.get("max_seq_length", 2048),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        report_to=cfg.get("report_to", []),
        seed=cfg.get("seed", 42),
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
    )
    trainer.train()

    adapter_path = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter_path))
    return adapter_path
