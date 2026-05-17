# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Post-train **Qwen3-4B-Instruct** with **GRPO** (TRL + Unsloth) to predict **ICD-10 codes** from clinical discharge summaries. The reward is rule-based and verifiable - per-code exact match plus partial credit for the correct ICD-10 chapter - making this a concrete RLVR (Reinforcement Learning with Verifiable Rewards) demonstration on healthcare data rather than the usual GSM8K math.

> Note: the architecture / paths / reward design described below reflect the initial scaffold and will change in Phase 0 reconciliation (move to `src/` layout, switch to single weighted-F1 scalar reward, add SFT baseline, add paper/CI/pre-commit/uv). The approved plan is at `/root/.claude/plans/take-qwen3-4b-instruct-apply-group-effervescent-quill.md`.

## Project Skills

Five skill bundles are version-controlled under `.claude/skills/`. Future Claude sessions on this repo should pick them up automatically and invoke the matching one before acting:

| Situation | Skill |
|---|---|
| Starting any feature, bugfix, or refactor under `src/clinical_grpo/{data,reward,eval,utils}` | `test-driven-development` (iron law: failing test first, watch it fail, minimal code to pass) |
| Have a multi-task spec to break into 2-5 minute steps | `writing-plans` |
| Have a written plan and need to execute task-by-task | `executing-plans` |
| 2+ independent failures, sweep cells, ablations, or paper sections to investigate or draft | `dispatching-parallel-agents` |
| About to write code (any task, any phase) | `karpathy-guidelines` (announce assumptions, simplicity first, surgical edits, explicit success criteria) |

TDD exception: the Unsloth + TRL training loop in `src/clinical_grpo/training/{sft,grpo}.py` is verified by the GPU smoke test (`tests/test_smoke_train.py @pytest.mark.gpu`), not by unit tests. This is documented in the `test-driven-development` SKILL.md as a permitted exception.

The skill bundles also reference sub-skills that are NOT installed in this repo (`subagent-driven-development`, `using-git-worktrees`, `finishing-a-development-branch`, `brainstorming`, `testing-anti-patterns`). If you hit a hard dependency on one of those, surface the gap to the user rather than fabricate a substitute.

## Read These First

Future Claude sessions should skim these before touching anything else — they encode the architectural decisions everything else depends on:

- `clinical_grpo/training/train_grpo.py` — Unsloth `FastLanguageModel` + TRL `GRPOTrainer` wiring. The integration point.
- `clinical_grpo/rewards/composite.py` — registers the three reward functions TRL consumes. Reward semantics drive model behavior.
- `clinical_grpo/prompts/templates.py` — system + user prompt and the **JSON output contract** (`{"codes": [...]}`) that the reward parser depends on. Change this and rewards break.
- `clinical_grpo/utils/icd10.py` — code normalization and chapter map. Single source of truth for what counts as a "valid" code and which codes share a chapter.
- `configs/train.yaml` + `configs/hardware/rtx4080.yaml` — every tunable knob.

## Architecture

Data flow:

```
raw CSV (MIMIC-III demo / Synthea FHIR)
  → clinical_grpo.data.preprocess     (de-id, truncate to 2048 tok, patient-disjoint split)
  → HF Dataset rows {prompt, gold_codes}
  → GRPOTrainer(reward_funcs=[exact, chapter, format])
  → LoRA adapter in outputs/<run>/adapter
  → eval (deterministic code-set F1 + optional Groq Llama-3.3-70B judge)
```

Two non-obvious decisions worth knowing:

1. **Three reward functions, not one.** TRL logs each `reward_funcs` entry as a separate scalar. Splitting into `exact_match`, `chapter_partial`, and `format_validity` makes the training curves diagnostic — you can see immediately whether the model is collapsing on formatting, chasing easy chapter credit, or actually learning codes. The composite scoring (weights, hallucination penalty) is in `rewards/composite.py`.

2. **Hardware config is split from model/train configs.** RTX 4080-specific knobs (`load_in_4bit`, `use_gradient_checkpointing="unsloth"`, `per_device_train_batch_size=2`, `fp16`) live in `configs/hardware/rtx4080.yaml`. Model architecture (`configs/model.yaml`) and GRPO hyperparameters (`configs/train.yaml`) are hardware-agnostic. Switching to an A100 should be one file, not three.

## Reward Function Contract

TRL `GRPOTrainer` expects `reward_funcs: list[Callable]` where each callable has the signature `fn(completions: list[str], **dataset_columns) -> list[float]`. Output contract the parser depends on:

```json
{"codes": ["E11.9", "I10", "N17.9"]}
```

The parser regex-extracts the first `{...}` block, `json.loads` it, validates against a jsonschema, then normalizes each code (uppercase, dot inserted at position 3 when length > 3).

Gotchas:
- Malformed JSON → `format_validity_reward` returns `-0.5` (actively discouraged), not `0.0`.
- Empty gold lists are filtered at the dataset stage — do not pass them into reward functions.
- Duplicate predicted codes are deduped before scoring.
- ICD-9-style codes (numeric only, no letter prefix) are **rejected** by `is_valid`; they do not earn chapter credit. This is intentional — chapter credit for unrecognizable strings would be free reward hacking.

## Common Commands

```bash
# install
pip install -e ".[dev]"

# data preprocessing
python -m clinical_grpo.data.preprocess --source mimic_demo --out data/processed/mimic.parquet
python -m clinical_grpo.data.preprocess --source synthea   --out data/processed/synthea.parquet

# training
python scripts/train.py --config configs/train.yaml --hw configs/hardware/rtx4080.yaml

# evaluation
python scripts/eval.py --adapter outputs/<run>/adapter --split test
python scripts/eval.py --adapter outputs/<run>/adapter --split test --judge groq

# tests
pytest tests/                                              # all CPU tests
pytest tests/test_rewards.py::test_chapter_partial -x      # single test
pytest tests/test_smoke_train.py -k one_step -x            # GPU smoke; requires CUDA

# demo (ZeroGPU Gradio Space)
cd app && python app.py
```

## Hardware Notes (RTX 4080, 16 GB)

Unsloth 4-bit base + LoRA rank 16, sequence length 2048, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, fp16, `use_gradient_checkpointing="unsloth"`. Full training run is 24–36 h split across 2–3 sessions.

**Always run the GPU smoke test (`pytest tests/test_smoke_train.py -k one_step -x`) before launching a long session.** It executes one full GRPO step on a tiny synthetic dataset in under 5 minutes and asserts (a) loss is finite, (b) all three reward names appear in the TRL log, (c) the saved adapter reloads, (d) a generation parses as valid JSON. If smoke fails, a 24-hour run will too — fix the smoke first.

## Branch Convention

Develop on `claude/qwen3-icd10-grpo-b95fR`. Do not push to `main` without explicit permission.
