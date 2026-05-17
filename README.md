# ClinicalGRPO

Post-training **Qwen3-4B-Instruct** with **GRPO** (Group Relative Policy Optimization,
the DeepSeek-R1-style algorithm) to predict **ICD-10 codes** from clinical discharge
summaries — using a rule-based, verifiable reward (Reinforcement Learning with
Verifiable Rewards). Built with the TRL + Unsloth recipe; designed to fit on a single
RTX 4080 (16 GB) at 4-bit + LoRA rank 16.

See [`CLAUDE.md`](./CLAUDE.md) for the architecture overview, common commands, and
hardware notes.

## Quickstart

```bash
pip install -e ".[dev]"

# preprocess MIMIC-III demo
python -m clinical_grpo.data.preprocess --source mimic_demo --out data/processed/mimic.parquet

# train
python scripts/train.py --config configs/train.yaml --hw configs/hardware/rtx4080.yaml

# evaluate
python scripts/eval.py --adapter outputs/<run>/adapter --split test
```

## Status

Initial scaffold. Modules are stubs with real signatures and TODOs; full
implementation lands incrementally.
