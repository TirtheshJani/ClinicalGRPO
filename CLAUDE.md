# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Post-train **Qwen3-4B-Instruct** with **GRPO** (TRL + Unsloth) to predict **ICD-10-CM codes** from clinical discharge summaries. The reward is rule-based and verifiable (per-code exact-match recall, partial credit for the correct ICD-10 chapter, and a format penalty), making this a concrete RLVR (Reinforcement Learning with Verifiable Rewards) demonstration on healthcare data rather than the usual GSM8K math.

Current state: the `src/` layout, the SFT baseline, eval harness, inference wrapper, Gradio app and CI are in place. Still open from the original plan: replacing the three-term reward with a single weighted-F1 scalar (the current chapter term is unbounded, see "Reward Function Contract"), and running the actual experiments (blocked on PhysioNet credentialing for MIMIC notes; runbook in `docs/plans/2026-05-18-phase3-experiment-execution.md`). No training results exist yet; never report numbers that were not produced by a run.

## Project Skills

Five skill bundles are version-controlled under `.claude/skills/`. Pick the matching one before acting:

| Situation | Skill |
|---|---|
| Starting any feature, bugfix, or refactor under `src/clinical_grpo/{data,rewards,eval,utils,prompts}` | `test-driven-development` (failing test first, watch it fail, minimal code to pass) |
| Have a multi-task spec to break into 2-5 minute steps | `writing-plans` |
| Have a written plan and need to execute task-by-task | `executing-plans` |
| 2+ independent failures, sweep cells, ablations, or paper sections to investigate or draft | `dispatching-parallel-agents` |
| About to write code (any task, any phase) | `karpathy-guidelines` (announce assumptions, simplicity first, surgical edits, explicit success criteria) |

TDD exception: the Unsloth + TRL training loops in `src/clinical_grpo/training/{train_grpo,train_sft}.py` are verified by the GPU smoke tests (`tests/test_smoke_train.py`, `tests/test_smoke_sft.py`, marked `@pytest.mark.gpu`), not by unit tests.

The skill bundles reference sub-skills that are NOT installed in this repo (`subagent-driven-development`, `using-git-worktrees`, `finishing-a-development-branch`, `brainstorming`, `testing-anti-patterns`). If you hit a hard dependency on one of those, surface the gap to the user rather than fabricate a substitute.

## Read These First

- `src/clinical_grpo/training/train_grpo.py`: config loading (`load_config` deep-merges train + hardware YAML, then loads `model_config`) and Unsloth `FastLanguageModel` + TRL `GRPOTrainer` wiring. The integration point.
- `src/clinical_grpo/rewards/composite.py`: `parse_completion` and the three reward functions TRL consumes (`build_reward_funcs`). Reward semantics drive model behavior.
- `src/clinical_grpo/prompts/templates.py`: system + user prompt and the **JSON output contract** (`{"codes": [...]}`) that the parser depends on. Change this and rewards break.
- `src/clinical_grpo/utils/icd10.py`: code normalization, validation and chapter map. Single source of truth for what counts as a "valid" code and which codes share a chapter.
- `configs/train.yaml` + `configs/hardware/rtx4080.yaml`: every tunable knob, including reward weights under `rewards:`.

## Architecture

Data flow:

```
data/raw/mimic-iii-demo (NOTEEVENTS + DIAGNOSES_ICD, ICD-9 -> ICD-10 via CMS GEM)
data/raw/synthea/fhir   (synthetic FHIR bundles, native ICD-10)
  -> clinical_grpo.data.preprocess   (strip redaction tokens, truncate, patient-disjoint 80/10/10 split)
  -> data/processed/<name>.parquet, <name>_val.parquet, <name>_test.parquet
  -> clinical_grpo.data.dataset      (HF Dataset rows {prompt, gold_codes})
  -> GRPOTrainer(reward_funcs=[exact_match, chapter_partial, format_validity])  or SFTTrainer baseline
  -> LoRA adapter in outputs/<run>/adapter
  -> clinical_grpo.eval.run_eval     (micro code F1, chapter F1, bootstrap CI, optional Groq Llama-3.3-70B judge)
```

Two non-obvious decisions:

1. **Three reward functions, not one.** TRL logs each `reward_funcs` entry as a separate scalar and sums them (all weights 1.0) for the advantage. Splitting makes the training curves diagnostic: you can see whether the model is collapsing on formatting, chasing chapter credit, or learning codes.

2. **Hardware config is split from model/train configs.** GPU-specific knobs (`load_in_4bit`, `use_gradient_checkpointing="unsloth"`, batch size, fp16/bf16) live in `configs/hardware/{rtx4080,a100,t4}.yaml`. `configs/model.yaml` and `configs/train.yaml` are hardware-agnostic.

## Reward Function Contract

TRL `GRPOTrainer` calls each reward as `fn(completions, **dataset_columns) -> list[float]`; ours read the `gold_codes` column. Output contract:

```json
{"codes": ["E11.9", "I10", "N17.9"]}
```

`parse_completion` strips `<think>...</think>` blocks, regex-extracts the first `{...}` block, `json.loads` it, requires `codes` to be a list of strings, drops codes that fail `is_valid`, then normalizes (uppercase, dot inserted after the 3-character category) and dedupes. There is no jsonschema validation step.

Gotchas:
- Malformed JSON: `format_validity_reward` returns `-format_invalid_penalty` (default `-0.5`), not `0.0`.
- Empty gold lists are filtered in `preprocess.build_table`; do not pass them into reward functions.
- ICD-9-style codes (numeric only) are **rejected** by `is_valid` and earn no chapter credit. Intentional: chapter credit for unrecognizable strings would be free reward hacking.
- **Known issue:** `chapter_partial_reward` divides in-chapter misses by the number of gold codes, so it is unbounded above. Spamming codes from gold chapters can beat a perfect answer. Fix (cap, or the planned weighted-F1 reward) before any long run; changing it is a reward-semantics change, so confirm with the user.

## Common Commands

```bash
# install (CPU tests work without CUDA; unsloth/torch install but are not imported by unit tests)
pip install -e ".[dev]"          # add ".[app]" for the Gradio demo

# CI checks (.github/workflows/ci.yml)
ruff check .
python -m mypy src/clinical_grpo/utils/ src/clinical_grpo/rewards/ src/clinical_grpo/prompts/ \
  src/clinical_grpo/eval/f1.py src/clinical_grpo/data/dataset.py src/clinical_grpo/data/preprocess.py \
  --ignore-missing-imports
pytest tests/ -q --ignore=tests/test_smoke_train.py
pytest tests/test_rewards.py -x                            # single file

# reward walkthrough (CPU)
python examples/score_completion.py

# data
python scripts/download_gem.py
python scripts/setup_data.py                               # checks raw data, prints what is missing
python -m clinical_grpo.data.preprocess --source mimic_demo --out data/processed/mimic.parquet
python -m clinical_grpo.data.preprocess --source synthea   --out data/processed/synthea.parquet
python scripts/validate_data.py data/processed/mimic.parquet

# training (GPU)
pytest tests/test_smoke_train.py -k one_step -x            # always run first
python scripts/train_sft.py --config configs/train.yaml --hw configs/hardware/rtx4080.yaml
python scripts/train.py     --config configs/train.yaml --hw configs/hardware/rtx4080.yaml
python scripts/sweep.py --sweep configs/sweep.yaml --config configs/train.yaml --hw configs/hardware/rtx4080.yaml

# evaluation / inference (GPU)
python scripts/eval.py --adapter outputs/grpo/adapter --split test
python scripts/eval.py --adapter outputs/grpo/adapter --split test --judge groq   # needs GROQ_API_KEY
python scripts/batch_infer.py --adapter outputs/grpo/adapter --input data/processed/mimic_test.parquet
ADAPTER_REPO=<hf-user>/<repo> python app/app.py
```

## Conventions

- CI installs ruff unpinned, so new ruff releases can enable new default rules. If `ruff check .` fails on untouched code, that is the likely cause.
- Tests must not hardcode the checkout path; resolve the repo root from `Path(__file__)`.
- No em dashes in docs or commit messages.

## Hardware Notes (RTX 4080, 16 GB)

Unsloth 4-bit base + LoRA rank 16, sequence length 2048, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, fp16, `use_gradient_checkpointing="unsloth"`. The Phase 3 runbook estimates roughly 7-12 h for one GRPO run and about 2 h for the SFT baseline (estimates, not measurements).

**Always run the GPU smoke test (`pytest tests/test_smoke_train.py -k one_step -x`) before launching a long session.** It executes one GRPO step on a tiny synthetic dataset and asserts (a) loss is finite, (b) all three reward names appear in the TRL log, (c) the saved adapter reloads, (d) generation completes. If smoke fails, a long run will too.

## Branch Convention

Develop on a feature branch (for example `claude/<topic>`) and merge to `main` via pull request. Do not push to `main` without explicit permission.
