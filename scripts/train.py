"""Thin CLI wrapper around clinical_grpo.training.train_grpo.train."""

from __future__ import annotations

from pathlib import Path

import tyro

from clinical_grpo.training.train_grpo import load_config, train


def main(
    config: Path = Path("configs/train.yaml"),
    hw: Path = Path("configs/hardware/rtx4080.yaml"),
    max_steps: int | None = None,
) -> None:
    cfg = load_config(config, hw)
    adapter = train(cfg, max_steps_override=max_steps)
    print(f"adapter saved to: {adapter}")


if __name__ == "__main__":
    tyro.cli(main)
