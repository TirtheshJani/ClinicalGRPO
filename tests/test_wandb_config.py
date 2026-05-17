from pathlib import Path
from clinical_grpo.training.train_grpo import load_config


def _base(tmp_path):
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
        "report_to: []\n"
        "grpo:\n  max_steps: 100\n  beta: 0.04\n"
    )
    return train_yaml


def test_no_extra_cfgs_report_to_is_empty(tmp_path):
    cfg = load_config(_base(tmp_path))
    assert cfg["report_to"] == []


def test_wandb_extra_cfg_enables_tracking(tmp_path):
    train_yaml = _base(tmp_path)
    wandb_yaml = tmp_path / "wandb.yaml"
    wandb_yaml.write_text(
        "report_to: [wandb]\n"
        "wandb:\n  project: clinical-grpo\n  run_name: test-run\n"
    )
    cfg = load_config(train_yaml, extra_cfgs=[wandb_yaml])
    assert "wandb" in cfg["report_to"]
    assert cfg["wandb"]["project"] == "clinical-grpo"
    assert cfg["wandb"]["run_name"] == "test-run"


def test_extra_cfgs_later_wins(tmp_path):
    train_yaml = _base(tmp_path)
    e1 = tmp_path / "e1.yaml"
    e1.write_text("custom: first\n")
    e2 = tmp_path / "e2.yaml"
    e2.write_text("custom: second\n")
    cfg = load_config(train_yaml, extra_cfgs=[e1, e2])
    assert cfg["custom"] == "second"


def test_extra_cfgs_none_is_default(tmp_path):
    # Calling without extra_cfgs arg must not raise
    cfg = load_config(_base(tmp_path))
    assert "grpo" in cfg


def test_wandb_yaml_is_valid(tmp_path):
    """The real configs/wandb.yaml parses cleanly."""
    import yaml
    wandb_path = Path("configs/wandb.yaml")
    assert wandb_path.exists(), "configs/wandb.yaml must be created"
    data = yaml.safe_load(wandb_path.read_text())
    assert "report_to" in data
    assert "wandb" in data
    assert "project" in data["wandb"]
