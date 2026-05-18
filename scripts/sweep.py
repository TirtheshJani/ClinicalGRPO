"""GRPO hyperparameter sweep — Cartesian product over configs/sweep.yaml params.

Usage: python scripts/sweep.py --sweep configs/sweep.yaml --config configs/train.yaml --hw configs/hardware/rtx4080.yaml
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any

import tyro
import yaml


def apply_override(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def generate_configs(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(params.keys())
    return [
        {k: v for k, v in zip(keys, combo)}
        for combo in itertools.product(*[params[k] for k in keys])
    ]


def main(
    sweep: Path = Path("configs/sweep.yaml"),
    config: Path = Path("configs/train.yaml"),
    hw: Path = Path("configs/hardware/rtx4080.yaml"),
) -> None:
    from clinical_grpo.training.train_grpo import load_config, train

    sweep_cfg = yaml.safe_load(sweep.read_text())["sweep"]
    overrides = generate_configs(sweep_cfg["params"])
    max_steps = sweep_cfg.get("max_steps", 50)
    out_csv = Path(sweep_cfg.get("output_csv", "outputs/sweep_results.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, override in enumerate(overrides):
        cfg = load_config(config, hw)
        for k, v in override.items():
            apply_override(cfg, k, v)
        cfg["output_dir"] = str(out_csv.parent / f"sweep_{i:03d}")
        print(f"\n[sweep {i + 1}/{len(overrides)}] {override}")
        try:
            train(cfg, max_steps_override=max_steps)
            results.append({"run": i, "status": "ok", **override})
        except Exception as e:
            results.append({"run": i, "status": f"error: {e}", **override})

    fieldnames = ["run", "status"] + list(overrides[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults: {out_csv}")


if __name__ == "__main__":
    tyro.cli(main)
