# Phase 3: Experiment Execution Run Plan

> **For the operator (TJ):** This is the runbook to take the merged codebase from "no data, no adapter" to "trained SFT baseline + trained GRPO model + eval reports + optional HF Hub push." Phases 1 and 2 are merged on `main`; the code is ready. This plan is the operational sequence, not a code task.

**Last audited:** 2026-05-18 by three Explore agents (data, training-config, eval/inference).

**Hardware target:** Windows 11 + RTX 4080 (16 GB).

---

## TL;DR — what state is the repo in?

| Area | Status | Notes |
|---|---|---|
| Training code (GRPO + SFT) | Ready | Configs, smoke tests, checkpoint resume all wired. |
| Eval code (F1 + bootstrap CI + Groq judge) | Ready | Groq judge is optional. |
| Inference (predictor + batch + Gradio) | Ready | `<think>`-block stripping is in place. |
| **Data on disk** | **MISSING** | Zero parquet files. PhysioNet download + preprocess required. |
| HF Hub push script | Ready | `huggingface_hub` is transitive via `transformers`, but undeclared. |
| Gradio app `ADAPTER_REPO` env var | Default placeholder | Must be set before launch. |

Single blocker: **no data on disk.** Everything else is green.

---

## Wall-clock reality check

The README says "24-36 h across 2-3 sessions." The training-config audit estimates one GRPO run at **7-12 h** and one SFT run at **1.5-2 h** on the RTX 4080. The 24-36 h figure only makes sense as cumulative time across SFT baseline + GRPO + eval + sweep iterations, not as a single uninterrupted run.

Plan around **one overnight session**:
- SFT baseline ~2 h
- GRPO run ~10 h
- Eval on both ~30 min
- Total: ~12-13 h (one night, one morning)

---

## Pre-flight (do this before kicking off)

### P1. Get PhysioNet credentials (REQUIRED for notes)

The open-access MIMIC-III demo ships `NOTEEVENTS.csv` as a 95-byte header-only stub. SHA256 confirms this is intentional — PhysioNet does not release clinical notes without credentialing, even for the 100-patient demo. The structured tables (DIAGNOSES_ICD etc.) are public and already downloaded; only NOTEEVENTS needs the credentialed path.

Steps (free, but the CITI module is ~2 hours):
1. Register at https://physionet.org/register/
2. Verify email, then go to https://physionet.org/settings/credentialing/
3. Complete CITI "Data or Specimens Only Research" training (https://about.citiprogram.org/, link your CITI account to PhysioNet)
4. Sign the MIMIC-III Clinical Database data use agreement
5. Once credentialed, log in and download the credentialed version of NOTEEVENTS.csv from https://physionet.org/content/mimiciii-demo/1.4/
6. Replace `data/raw/mimic-iii-demo/NOTEEVENTS.csv` with the downloaded file
7. Verify: `python scripts/setup_data.py` should print `[ok] MIMIC-III demo: ... notes present` rather than `[warn]`

(Alternative: if credentialing is a blocker, you can credential for full MIMIC-III instead and get a much larger note corpus. Same agreement.)

### P2. Open-access data (already done)

These are already on disk from `data/raw/` so you can skip; commands recorded for reproducibility:

```powershell
# Open-access MIMIC-III demo structured tables (~11 MB zip, no auth):
curl -fsSL -o data/raw/mimic-demo.zip https://physionet.org/content/mimiciii-demo/get-zip/1.4/
# (extracts to data/raw/mimic-iii-demo/, contains all 26 CSVs except real NOTEEVENTS)

# GEM crosswalk (CMS 404'd, falls back to GitHub mirror automatically):
python scripts/download_gem.py
```

**Why GEM is load-bearing:** if the GEM file is missing or empty, `mimic_demo._load_gem()` silently returns an empty dict; ICD-9 codes pass through unconverted; `is_valid()` then rejects them all because ICD-9 is numeric-only and fails the ICD-10 regex; you end up with an empty training set and a confusing crash deep in TRL. The updated `download_gem.py` tries CMS first, falls back to a GitHub mirror, and refuses to write a file smaller than 100 KB.

### P3. Preprocess to parquet

```powershell
python -m clinical_grpo.data.preprocess --source mimic_demo --out data/processed/mimic.parquet
```

Produces `data/processed/mimic.parquet`, `mimic_val.parquet`, `mimic_test.parquet`.

Expected shape: ~80-90 train rows, ~10-15 val, ~10-15 test (the MIMIC-III demo is small — 100 patients).

### P4. Validate the parquets

```powershell
python scripts/validate_data.py data/processed/mimic.parquet
```

If this reports `n_empty_code_lists > 0` or fewer than 50 train rows, **stop and investigate before training.** The most likely cause is a missing or empty GEM file (re-run P2).

### P5. Install dependencies

```powershell
pip install -e ".[dev,app]"
```

`huggingface_hub` arrives transitively via `transformers`; no extra install required for `push_to_hub.py`.

### P6. GPU smoke tests (NON-NEGOTIABLE)

Both gated by `@pytest.mark.gpu` and require CUDA. Each runs one full training step against `Qwen3-0.6B` on a tiny synthetic dataset.

```powershell
pytest tests/test_smoke_sft.py -k one_step -x      # ~3-5 min
pytest tests/test_smoke_train.py -k one_step -x    # ~3-5 min
```

GRPO smoke asserts: adapter saves, all three reward names appear in TRL logs, adapter reloads, generation parses as JSON.
SFT smoke asserts: adapter saves, `trainer_state.json` has finite loss, adapter reloads.

**If either smoke fails, a 10-hour run will fail too — fix smoke before continuing.**

### P7. (Optional) Set up tracking

W&B is off by default (`report_to: []` in `configs/train.yaml`). To enable:

```powershell
$env:WANDB_API_KEY = "<your_key>"
# add --wandb configs/wandb.yaml to the train command in steps T1/T2
```

If you skip W&B, training writes loss + reward curves to `outputs/<run>/checkpoint-*/trainer_state.json` — readable but less convenient.

---

## Training (T)

### T1. SFT baseline first (~1.5-2 h)

The SFT baseline is the comparison anchor for the GRPO claim. Run it first so it's done before bed.

```powershell
python scripts/train_sft.py --config configs/train.yaml --hw configs/hardware/rtx4080.yaml
```

Produces `outputs/sft/adapter/` and checkpoint dirs every 100 steps.

**Effective config (verified from configs):**
- max_steps=500, save_steps=100, lr=2e-5
- per_device_batch=2, grad_accum=8, fp16, seq_len=2048
- 4-bit base + LoRA rank 16 + `gradient_checkpointing="unsloth"`

Sanity-check the first checkpoint at ~step 100 (~20 min in):
- `outputs/sft/checkpoint-100/trainer_state.json` should show a falling `loss`.

### T2. GRPO run (~7-12 h)

```powershell
python scripts/train.py --config configs/train.yaml --hw configs/hardware/rtx4080.yaml
```

Produces `outputs/grpo/adapter/` and `outputs/grpo/checkpoint-*/`.

**Effective config:**
- max_steps=1000, save_steps=100, lr=5e-6, beta=0.04, num_generations=8
- Same batch/precision/seq settings as SFT
- Three reward functions logged separately by TRL: `exact_match_reward`, `chapter_partial_reward`, `format_validity_reward`

Watch the first ~50 steps. If you see any of these, kill and investigate:
- All three reward streams flat at zero → format collapse (model not emitting JSON)
- `format_validity_reward` at -0.5 for every step → JSON parser failure (check `parse_completion` is stripping `<think>` blocks)
- `exact_match_reward` rising while `chapter_partial_reward` falls → reward hacking on chapter credit

### T3. Recovery if a session interrupts

Checkpoint resume **is wired** (verified at `train_grpo.py:102` and `train_sft.py:92`):

```powershell
python scripts/train.py --config configs/train.yaml --hw configs/hardware/rtx4080.yaml --resume_from outputs/grpo/checkpoint-500
```

Same flag works for `train_sft.py`. The trainer rebuilds optimizer/scheduler state from the checkpoint directory.

---

## Eval (E)

### E1. Eval both adapters on the test split

```powershell
python scripts/eval.py --adapter outputs/sft/adapter  --split test --out outputs/sft/eval_test.json
python scripts/eval.py --adapter outputs/grpo/adapter --split test --out outputs/grpo/eval_test.json
```

Each produces a JSON with:
- `micro_code_f1` (precision/recall/F1, tp/fp/fn)
- `chapter_f1`
- `micro_code_f1_bootstrap` (mean + 95% CI, 1000 resamples, seed=42) — **always computed, no flag needed**

### E2. (Optional) Groq LLM judge

```powershell
$env:GROQ_API_KEY = "<your_key>"
python scripts/eval.py --adapter outputs/grpo/adapter --split test --judge groq --out outputs/grpo/eval_test_judged.json
```

Uses `llama-3.3-70b-versatile`. Free tier is 1000 req/day, ~1 eval run = ~N test samples in calls. Results are cached at `outputs/groq_cache/` keyed by sha256 of (pred, gold) pairs — re-runs are free.

If you skip this, the deterministic F1 + chapter F1 + bootstrap CI are still produced.

### E3. Compare GRPO vs SFT

Manual at this stage. Open both JSON reports side by side. The headline claim of the project is that GRPO's `micro_code_f1` exceeds SFT's by a margin larger than the overlap of their bootstrap 95% CIs. If they overlap, GRPO didn't beat the baseline cleanly and that's a finding to write up honestly.

---

## Deployment (D, optional)

### D1. Batch inference on the test set

```powershell
python scripts/batch_infer.py --adapter outputs/grpo/adapter --input data/processed/mimic_test.parquet --out outputs/grpo/predictions.parquet
```

Adds a `predicted_codes` column. Malformed JSON rows get `[]` (already tested — see `tests/test_inference.py::test_predict_returns_empty_on_malformed_json`).

### D2. Push to HuggingFace Hub

```powershell
huggingface-cli login   # one-time; token in ~/.cache/huggingface/token
python scripts/push_to_hub.py --adapter outputs/grpo/adapter --repo-id <your-user>/clinical-grpo-qwen3-4b-icd10
```

Add `--private` if you want it gated. The script calls `create_repo(..., exist_ok=True)` so re-pushes overwrite.

### D3. Launch Gradio app locally

```powershell
$env:ADAPTER_REPO = "<your-user>/clinical-grpo-qwen3-4b-icd10"  # or a local path; Unsloth accepts both
cd app
python app.py
```

The default `ADAPTER_REPO=TODO/...` placeholder will fail with a clear "model not found" error — set the env var first.

---

## Verification checklist (end-to-end)

Before declaring Phase 3 done:

- [ ] `data/processed/mimic{,_val,_test}.parquet` all exist and have >0 rows
- [ ] Both smoke tests pass on the RTX 4080
- [ ] `outputs/sft/adapter/` and `outputs/grpo/adapter/` both exist with `adapter_config.json`
- [ ] `outputs/sft/eval_test.json` and `outputs/grpo/eval_test.json` both readable and contain finite F1 numbers
- [ ] GRPO's micro F1 either beats SFT (write that up) or doesn't (write that up honestly)

---

## Known minor gaps (do not block Phase 3)

These were flagged by the eval/inference audit but are not on the critical path:

1. **`huggingface_hub` is undeclared in `pyproject.toml`.** It arrives transitively via `transformers>=4.49`, so installs work, but explicit declaration is good hygiene. One-line fix when convenient.
2. **No integration test for `outputs/<run>/adapter/ → eval → JSON`.** Unit tests cover each piece with mocks; nothing exercises the full chain against a real Unsloth checkpoint. Worth adding after the first real run when you have a known-good adapter to assert against.
3. **App `ADAPTER_REPO` placeholder is `"TODO/..."`.** Intentional — meant to be set via env var. If you publish a Hub model ID worth defaulting to, swap it in `app/app.py:14`.

These can be folded into a small Phase 3.5 cleanup PR after the run lands.
