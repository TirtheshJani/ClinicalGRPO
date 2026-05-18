import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_sweep_yaml_exists_and_has_params():
    import yaml
    p = Path("configs/sweep.yaml")
    assert p.exists()
    data = yaml.safe_load(p.read_text())
    assert "sweep" in data
    assert "params" in data["sweep"]


def test_generate_configs_cartesian_product():
    from scripts.sweep import generate_configs
    params = {"grpo.beta": [0.01, 0.04], "grpo.learning_rate": [1e-5, 5e-6]}
    configs = generate_configs(params)
    assert len(configs) == 4


def test_generate_configs_sets_values():
    from scripts.sweep import generate_configs
    configs = generate_configs({"grpo.beta": [0.01]})
    assert configs[0]["grpo.beta"] == 0.01


def test_apply_override_sets_nested_key():
    from scripts.sweep import apply_override
    cfg = {"grpo": {"beta": 0.04, "lr": 5e-6}}
    apply_override(cfg, "grpo.beta", 0.01)
    assert cfg["grpo"]["beta"] == 0.01
    assert cfg["grpo"]["lr"] == 5e-6


def test_apply_override_creates_missing_intermediate_keys():
    from scripts.sweep import apply_override
    cfg = {}
    apply_override(cfg, "grpo.beta", 0.99)
    assert cfg["grpo"]["beta"] == 0.99


def test_sweep_cli_help():
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/sweep.py", "--help"],
        capture_output=True, text=True, cwd="/home/user/ClinicalGRPO"
    )
    assert r.returncode == 0
