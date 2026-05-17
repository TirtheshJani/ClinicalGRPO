# tests/test_checkpoint_resume.py
from unittest.mock import patch
from clinical_grpo.training.train_grpo import load_config


def _write_yamls(tmp_path):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        "base_model: Qwen/Qwen3-4B-Instruct\n"
        "lora:\n  r: 16\n  alpha: 32\n  dropout: 0.0\n"
        "  bias: none\n  target_modules: [q_proj]\n"
    )
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        f"model_config: {model_yaml}\n"
        "output_dir: outputs/grpo\n"
        "grpo:\n  max_steps: 100\n  beta: 0.04\n"
        "sft:\n  max_steps: 50\n"
    )
    return train_yaml


def test_resume_from_absent_gives_none(tmp_path):
    cfg = load_config(_write_yamls(tmp_path))
    assert cfg.get("resume_from") is None


def test_resume_from_key_in_yaml(tmp_path):
    train_yaml = _write_yamls(tmp_path)
    ckpt = str(tmp_path / "checkpoint-100")
    train_yaml.write_text(train_yaml.read_text() + f"resume_from: {ckpt}\n")
    cfg = load_config(train_yaml)
    assert cfg["resume_from"] == ckpt


def test_grpo_config_receives_resume_from(tmp_path):
    """train() passes resume_from_checkpoint to GRPOConfig."""
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock
    import pandas as pd

    pd.DataFrame({
        "discharge_summary": ["Patient note."],
        "icd10_codes": [["I10"]],
    }).to_parquet(tmp_path / "train.parquet")

    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()

    cfg = {
        "model": {
            "base_model": "Qwen/Qwen3-0.6B",
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none",
                     "target_modules": ["q_proj"]},
        },
        "grpo": {
            "num_generations": 2, "beta": 0.04, "learning_rate": 5e-6,
            "lr_scheduler_type": "constant", "warmup_ratio": 0.0,
            "max_steps": 1, "save_steps": 1, "logging_steps": 1,
            "max_prompt_length": 64, "max_completion_length": 32,
            "temperature": 0.9, "top_p": 1.0,
        },
        "rewards": {"chapter_weight": 0.3, "hallucination_penalty": 0.1,
                    "format_invalid_penalty": 0.5},
        "dataset": {"train_path": str(tmp_path / "train.parquet")},
        "output_dir": str(tmp_path / "output"),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "use_gradient_checkpointing": False,
        "load_in_4bit": False,
        "fp16": False, "bf16": False, "dtype": None,
        "seed": 42, "report_to": [],
        "max_seq_length": 128,
        "resume_from": str(ckpt),
    }

    captured: dict = {}

    def fake_grpo_config(**kwargs):
        captured["resume_from_checkpoint"] = kwargs.get("resume_from_checkpoint")
        raise SystemExit(0)

    fake_unsloth = ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = MagicMock()
    fake_unsloth.FastLanguageModel.from_pretrained.return_value = (MagicMock(), MagicMock())
    fake_unsloth.FastLanguageModel.get_peft_model.return_value = MagicMock()

    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        with patch("clinical_grpo.training.train_grpo.GRPOConfig", side_effect=fake_grpo_config):
            try:
                from clinical_grpo.training import train_grpo
                train_grpo.train(cfg, max_steps_override=1)
            except SystemExit:
                pass

    assert captured.get("resume_from_checkpoint") == str(ckpt)


def test_sft_config_receives_resume_from(tmp_path):
    """train_sft() passes resume_from_checkpoint to SFTConfig."""
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock
    import pandas as pd

    pd.DataFrame({
        "discharge_summary": ["Patient note."],
        "icd10_codes": [["I10"]],
    }).to_parquet(tmp_path / "train.parquet")

    ckpt = tmp_path / "checkpoint-sft-1"
    ckpt.mkdir()

    cfg = {
        "model": {
            "base_model": "Qwen/Qwen3-0.6B",
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none",
                     "target_modules": ["q_proj"]},
        },
        "sft": {"max_steps": 1, "learning_rate": 2e-5,
                "lr_scheduler_type": "constant", "warmup_ratio": 0.0,
                "save_steps": 1, "logging_steps": 1},
        "dataset": {"train_path": str(tmp_path / "train.parquet")},
        "output_dir": str(tmp_path / "output"),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "use_gradient_checkpointing": False,
        "load_in_4bit": False,
        "fp16": False, "bf16": False, "dtype": None,
        "seed": 42, "report_to": [],
        "max_seq_length": 128,
        "resume_from": str(ckpt),
    }

    captured: dict = {}

    def fake_sft_config(**kwargs):
        captured["resume_from_checkpoint"] = kwargs.get("resume_from_checkpoint")
        raise SystemExit(0)

    fake_unsloth = ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt text"
    fake_unsloth.FastLanguageModel.from_pretrained.return_value = (MagicMock(), mock_tokenizer)
    fake_unsloth.FastLanguageModel.get_peft_model.return_value = MagicMock()

    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        with patch("clinical_grpo.training.train_sft.SFTConfig", side_effect=fake_sft_config):
            try:
                from clinical_grpo.training import train_sft as train_sft_mod
                train_sft_mod.train_sft(cfg, max_steps_override=1)
            except SystemExit:
                pass

    assert captured.get("resume_from_checkpoint") == str(ckpt)
