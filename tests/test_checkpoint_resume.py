# tests/test_checkpoint_resume.py
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


def test_grpo_train_source_passes_resume_from_checkpoint():
    """train() source code forwards cfg['resume_from'] to GRPOConfig."""
    import inspect
    from clinical_grpo.training import train_grpo
    src = inspect.getsource(train_grpo.train)
    assert "resume_from_checkpoint" in src
    assert 'cfg.get("resume_from")' in src


def test_sft_train_source_passes_resume_from_checkpoint():
    """train_sft() source code forwards cfg['resume_from'] to SFTConfig."""
    import inspect
    from clinical_grpo.training import train_sft
    src = inspect.getsource(train_sft.train_sft)
    assert "resume_from_checkpoint" in src
    assert 'cfg.get("resume_from")' in src
