# ClinicalGRPO

Post-trains **Qwen3-4B-Instruct** with **GRPO** (Group Relative Policy Optimization,
the DeepSeek-R1-style algorithm) to predict **ICD-10-CM codes** from clinical discharge
summaries. Reward is rule-based and verifiable (RLVR) — no LLM judge required during
training. Built with TRL + Unsloth; fits on a single RTX 4080 (16 GB).

## Architecture

- **Three reward functions** logged separately by TRL:
  - `exact_match_reward` — recall over gold code set
  - `chapter_partial_reward` — partial credit for correct ICD-10 chapter + hallucination penalty
  - `format_validity_reward` — enforces `{"codes": [...]}` JSON output contract
- **LoRA rank 16** on 4-bit Qwen3-4B-Instruct, 2048-token sequences
- **Patient-disjoint splits** — 80/10/10 train/val/test, no patient leakage
- **SFT baseline** available alongside GRPO for controlled comparison

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Download the ICD-9->ICD-10 crosswalk
python scripts/download_gem.py

# 2. Preprocess MIMIC-III demo
python -m clinical_grpo.data.preprocess --source mimic_demo --out data/processed/mimic.parquet

# 3a. Train with GRPO
python scripts/train.py --config configs/train.yaml --hw configs/hardware/rtx4080.yaml

# 3b. Or train SFT baseline
python scripts/train_sft.py --config configs/train.yaml --hw configs/hardware/rtx4080.yaml

# 4. Evaluate
python scripts/eval.py --adapter outputs/grpo/adapter --split test
```

## Before a Long Training Run

```bash
pytest tests/test_smoke_train.py -k one_step -x  # requires CUDA, <5 min
```

Executes one full GRPO step on a tiny synthetic dataset and asserts: loss is finite,
all three reward names appear in the TRL log, the saved adapter reloads, and a
generation parses as valid JSON. If smoke fails, fix it before launching 24-hour runs.

## Status

| Module | Status |
|---|---|
| Reward functions (`rewards/`) | Done — 3 TRL-compatible callables, full test coverage |
| ICD-10 utils (`utils/icd10.py`) | Done — normalization, validation, chapter lookup |
| Data pipeline (`data/`) | Done — MIMIC-III demo + Synthea FHIR, ICD-9->ICD-10 GEM |
| Training — GRPO (`training/train_grpo.py`) | Done — Unsloth + TRL GRPOTrainer |
| Training — SFT (`training/train_sft.py`) | Done — Unsloth + TRL SFTTrainer (baseline) |
| Eval (`eval/`) | Done — micro F1, chapter F1, optional Groq LLM judge |
| Gradio app (`app/`) | Done — ZeroGPU HF Spaces ready |
| CI | Done — GitHub Actions, ruff, pytest |
| GPU smoke test | Done — `tests/test_smoke_train.py` |

## Data Sources

- **MIMIC-III demo**: 100 patients, public (no credentialing required). Download from
  [PhysioNet](https://physionet.org/content/mimiciii-demo/1.4/).
- **Synthea**: synthetic FHIR bundles. Generate with the
  [Synthea tool](https://github.com/synthetichealth/synthea).
- **CMS GEM** (ICD-9->ICD-10 crosswalk): `python scripts/download_gem.py` (automated).

## Hardware Notes

RTX 4080 (16 GB): 4-bit base + LoRA rank 16, 2048-token sequences,
`per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, fp16,
`use_gradient_checkpointing="unsloth"`. Full GRPO run: 24-36 h across 2-3 sessions.
Hardware-specific knobs live in `configs/hardware/rtx4080.yaml`; switching to an A100
requires changing one file.
