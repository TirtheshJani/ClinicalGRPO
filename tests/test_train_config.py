"""Unit tests for clinical_grpo.training.train_grpo.load_config."""

from pathlib import Path

import pytest

from clinical_grpo.training.train_grpo import load_config


def _write_model_yaml(tmp_path: Path) -> Path:
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        "base_model: Qwen/Qwen3-4B-Instruct\n"
        "lora:\n"
        "  r: 16\n"
    )
    return model_yaml


def _write_train_yaml(tmp_path: Path, model_yaml_path: Path) -> Path:
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        f"model_config: {model_yaml_path}\n"
        "grpo:\n"
        "  max_steps: 100\n"
        "  beta: 0.04\n"
        "output_dir: outputs/grpo\n"
    )
    return train_yaml


def test_load_config_reads_model_block(tmp_path: Path) -> None:
    model_yaml = _write_model_yaml(tmp_path)
    train_yaml = _write_train_yaml(tmp_path, model_yaml)

    cfg = load_config(train_yaml)

    assert cfg["model"]["base_model"] == "Qwen/Qwen3-4B-Instruct"


def test_load_config_no_hw(tmp_path: Path) -> None:
    model_yaml = _write_model_yaml(tmp_path)
    train_yaml = _write_train_yaml(tmp_path, model_yaml)

    cfg = load_config(train_yaml)

    assert cfg["grpo"]["max_steps"] == 100


def test_load_config_hw_overrides(tmp_path: Path) -> None:
    model_yaml = _write_model_yaml(tmp_path)
    train_yaml = _write_train_yaml(tmp_path, model_yaml)

    hw_yaml = tmp_path / "hw.yaml"
    hw_yaml.write_text(
        "per_device_train_batch_size: 4\n"
        "grpo:\n"
        "  beta: 0.01\n"
    )

    cfg = load_config(train_yaml, hw_cfg=hw_yaml)

    assert cfg["per_device_train_batch_size"] == 4
    # hw override changes beta but keeps other grpo keys
    assert cfg["grpo"]["beta"] == 0.01
    assert cfg["grpo"]["max_steps"] == 100


def test_load_config_deep_merge_doesnt_drop_keys(tmp_path: Path) -> None:
    model_yaml = _write_model_yaml(tmp_path)
    train_yaml = _write_train_yaml(tmp_path, model_yaml)

    # hw_cfg only sets one nested grpo key; other keys must survive
    hw_yaml = tmp_path / "hw_partial.yaml"
    hw_yaml.write_text(
        "grpo:\n"
        "  learning_rate: 0.00005\n"
    )

    cfg = load_config(train_yaml, hw_cfg=hw_yaml)

    assert cfg["grpo"]["max_steps"] == 100
    assert cfg["grpo"]["beta"] == pytest.approx(0.04)
    assert cfg["grpo"]["learning_rate"] == pytest.approx(5e-5)
