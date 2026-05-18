"""GRPO training entrypoint: Unsloth FastLanguageModel + TRL GRPOTrainer.

This module is the integration point — everything else (data, rewards, prompts,
configs) is wired together here. Keep it readable; complexity belongs in the
modules it imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from clinical_grpo.data.dataset import load_grpo_dataset
from clinical_grpo.rewards.composite import build_reward_funcs

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:  # pragma: no cover
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    train_cfg: Path | str,
    hw_cfg: Path | str | None = None,
    extra_cfgs: list[Path | str] | None = None,
) -> dict[str, Any]:
    with open(train_cfg) as f:
        cfg = yaml.safe_load(f)
    if hw_cfg:
        with open(hw_cfg) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    for extra in (extra_cfgs or []):
        with open(extra) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    model_cfg_path = Path(cfg.get("model_config", "configs/model.yaml"))
    with open(model_cfg_path) as f:
        cfg["model"] = yaml.safe_load(f)
    return cfg


def train(cfg: dict[str, Any], max_steps_override: int | None = None) -> Path:
    """Run GRPO training. Returns the path to the saved LoRA adapter."""
    # Heavy imports kept inside the function so unit tests can import this
    # module without paying the Unsloth/TRL startup cost.
    from unsloth import FastLanguageModel

    model_cfg = cfg["model"]
    grpo = cfg["grpo"]

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

    train_ds = load_grpo_dataset(cfg["dataset"]["train_path"])
    reward_funcs = build_reward_funcs(cfg["rewards"])

    grpo_cfg = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=grpo["num_generations"],
        beta=grpo["beta"],
        learning_rate=grpo["learning_rate"],
        lr_scheduler_type=grpo["lr_scheduler_type"],
        warmup_ratio=grpo["warmup_ratio"],
        max_steps=max_steps_override if max_steps_override is not None else grpo["max_steps"],
        save_steps=grpo["save_steps"],
        logging_steps=grpo["logging_steps"],
        max_prompt_length=grpo["max_prompt_length"],
        max_completion_length=grpo["max_completion_length"],
        temperature=grpo["temperature"],
        top_p=grpo["top_p"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        report_to=cfg.get("report_to", []),
        seed=cfg.get("seed", 42),
        resume_from_checkpoint=cfg.get("resume_from"),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_cfg,
        train_dataset=train_ds,
    )
    trainer.train()

    adapter_path = Path(cfg["output_dir"]) / "adapter"
    trainer.save_model(str(adapter_path))
    return adapter_path
