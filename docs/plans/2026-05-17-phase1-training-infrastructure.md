# Phase 1: Training Infrastructure Implementation Plan

> **For agentic workers:** Use dispatching-parallel-agents — all four tasks are fully independent. Each agent takes one task.

**Goal:** Make the training loop robust for a real 1000-step run: checkpoint resume, experiment tracking, data quality validation, and bootstrap confidence intervals on eval metrics.

**Architecture:** Four orthogonal additions — no shared state, no import cycles. Checkpoint resume is wired into existing training scripts. Bootstrap CI wraps the existing F1 functions. Data validation is a standalone script. W&B integration is config-only.

**Tech Stack:** TRL GRPOConfig/SFTConfig (resume), scipy/numpy (bootstrap), pandas (data validation), wandb (tracking).

---

## Task 1: Checkpoint Resume Support

**Files:**
- Modify: `src/clinical_grpo/training/train_grpo.py`
- Modify: `src/clinical_grpo/training/train_sft.py`
- Modify: `scripts/train.py`
- Modify: `scripts/train_sft.py`
- Create: `tests/test_checkpoint_resume.py`

TRL's `GRPOConfig` and `SFTConfig` both accept `resume_from_checkpoint: str | bool | None`. When set to a checkpoint directory path, training picks up from that step. We expose this through the CLI and through a `resume_from` key in the YAML config.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_checkpoint_resume.py
from pathlib import Path
from unittest.mock import MagicMock, patch
from clinical_grpo.training.train_grpo import load_config
import yaml, tempfile, os

def _write_yamls(tmp_path):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text("base_model: Qwen/Qwen3-4B-Instruct\nlora:\n  r: 16\n  alpha: 32\n  dropout: 0.0\n  bias: none\n  target_modules: [q_proj]\n")
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(f"model_config: {model_yaml}\noutput_dir: outputs/grpo\ngrpo:\n  max_steps: 100\n  beta: 0.04\nsft:\n  max_steps: 50\n")
    return train_yaml

def test_load_config_resume_key_absent_gives_none(tmp_path):
    cfg = load_config(_write_yamls(tmp_path))
    assert cfg.get("resume_from") is None

def test_load_config_resume_key_present(tmp_path):
    train_yaml = _write_yamls(tmp_path)
    content = train_yaml.read_text() + f"resume_from: {tmp_path}/checkpoint-100\n"
    train_yaml.write_text(content)
    cfg = load_config(train_yaml)
    assert cfg["resume_from"] == str(tmp_path / "checkpoint-100")

def test_train_grpo_passes_resume_to_grpo_config(tmp_path):
    """GRPOConfig is constructed with resume_from_checkpoint when cfg has it."""
    from clinical_grpo.training.train_grpo import train

    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    cfg = {
        "model": {"base_model": "Qwen/Qwen3-0.6B", "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none", "target_modules": ["q_proj"]}},
        "grpo": {"num_generations": 2, "beta": 0.04, "learning_rate": 5e-6, "lr_scheduler_type": "constant", "warmup_ratio": 0.0, "max_steps": 1, "save_steps": 1, "logging_steps": 1, "max_prompt_length": 64, "max_completion_length": 32, "temperature": 0.9, "top_p": 1.0},
        "rewards": {"chapter_weight": 0.3, "hallucination_penalty": 0.1, "format_invalid_penalty": 0.5},
        "dataset": {"train_path": str(tmp_path / "train.parquet")},
        "output_dir": str(tmp_path / "output"),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "use_gradient_checkpointing": False,
        "load_in_4bit": False,
        "fp16": False, "bf16": False, "dtype": None,
        "seed": 42, "report_to": [],
        "max_seq_length": 128,
        "resume_from": str(ckpt),
    }

    captured = {}
    OrigGRPOConfig = None

    def mock_grpo_config(**kwargs):
        captured["resume_from_checkpoint"] = kwargs.get("resume_from_checkpoint")
        raise SystemExit(0)  # stop before actual training

    with patch("clinical_grpo.training.train_grpo.GRPOConfig", side_effect=mock_grpo_config):
        try:
            train(cfg, max_steps_override=1)
        except SystemExit:
            pass

    assert captured["resume_from_checkpoint"] == str(ckpt)
```

- [ ] **Step 2: Run and confirm tests fail**

```bash
cd /home/user/ClinicalGRPO && python -m pytest tests/test_checkpoint_resume.py -v --tb=short 2>&1 | tail -15
```
Expected: 3 tests fail (2 pass trivially since `resume_from` key is absent, 1 fails on GRPOConfig not getting the arg).

- [ ] **Step 3: Add `resume_from` to `train_grpo.train()`**

In `src/clinical_grpo/training/train_grpo.py`, update the `GRPOConfig(...)` call — add one line:

```python
    grpo_cfg = GRPOConfig(
        output_dir=cfg["output_dir"],
        # ... existing fields ...
        seed=cfg.get("seed", 42),
        resume_from_checkpoint=cfg.get("resume_from"),   # ADD THIS LINE
    )
```

- [ ] **Step 4: Add `resume_from` to `train_sft.train_sft()`**

In `src/clinical_grpo/training/train_sft.py`, update the `SFTConfig(...)` call:

```python
    sft_cfg = SFTConfig(
        output_dir=output_dir,
        # ... existing fields ...
        seed=cfg.get("seed", 42),
        resume_from_checkpoint=cfg.get("resume_from"),   # ADD THIS LINE
        dataset_text_field="text",
        packing=False,
    )
```

- [ ] **Step 5: Expose `--resume-from` in `scripts/train.py`**

```python
def main(
    config: Path = Path("configs/train.yaml"),
    hw: Path = Path("configs/hardware/rtx4080.yaml"),
    max_steps: int | None = None,
    resume_from: str | None = None,
) -> None:
    cfg = load_config(config, hw)
    if resume_from is not None:
        cfg["resume_from"] = resume_from
    adapter = train(cfg, max_steps_override=max_steps)
    print(f"adapter saved to: {adapter}")
```

Apply the same pattern to `scripts/train_sft.py`.

- [ ] **Step 6: Run tests and verify all pass**

```bash
python -m pytest tests/test_checkpoint_resume.py -v 2>&1 | tail -10
```
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/clinical_grpo/training/train_grpo.py src/clinical_grpo/training/train_sft.py scripts/train.py scripts/train_sft.py tests/test_checkpoint_resume.py
git commit -m "feat: add checkpoint resume support to GRPO and SFT training"
```

---

## Task 2: Bootstrap Confidence Intervals for F1

**Files:**
- Create: `src/clinical_grpo/eval/bootstrap.py`
- Create: `tests/test_bootstrap.py`

Micro-code F1 on a test split of ~10–100 samples has high variance. Bootstrap resampling gives a 95% CI without parametric assumptions. The function resamples (pred, gold) pairs with replacement N times, computes F1 each time, and returns the 2.5th and 97.5th percentile.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bootstrap.py
from clinical_grpo.eval.bootstrap import bootstrap_f1_ci, BootstrapResult

def _perfect(n: int):
    codes = [["E11.9", "I10"]] * n
    return codes, codes

def _half_recall(n: int):
    gold = [["E11.9", "I10"]] * n
    pred = [["E11.9"]] * n      # recall = 0.5 always
    return pred, gold

def test_bootstrap_result_has_expected_fields():
    preds, golds = _perfect(10)
    r = bootstrap_f1_ci(preds, golds, n_resamples=50, seed=0)
    assert hasattr(r, "mean_f1")
    assert hasattr(r, "ci_low")
    assert hasattr(r, "ci_high")
    assert hasattr(r, "n_resamples")

def test_perfect_predictions_ci_near_one():
    preds, golds = _perfect(20)
    r = bootstrap_f1_ci(preds, golds, n_resamples=200, seed=42)
    assert r.mean_f1 == pytest.approx(1.0, abs=1e-9)
    assert r.ci_low > 0.95
    assert r.ci_high == pytest.approx(1.0, abs=1e-9)

def test_half_recall_ci_brackets_true_value():
    preds, golds = _half_recall(50)
    r = bootstrap_f1_ci(preds, golds, n_resamples=500, seed=42)
    # true F1 = 2*(0.5*1.0)/(0.5+1.0) ≈ 0.667; CI should contain it
    assert r.ci_low < 0.667 < r.ci_high

def test_ci_low_leq_mean_leq_ci_high():
    import random; random.seed(0)
    preds = [[f"E{i:02d}.9"] for i in range(30)]
    golds = [[f"E{i:02d}.9", "I10"] for i in range(30)]
    r = bootstrap_f1_ci(preds, golds, n_resamples=200, seed=1)
    assert r.ci_low <= r.mean_f1 <= r.ci_high

def test_n_resamples_stored():
    preds, golds = _perfect(5)
    r = bootstrap_f1_ci(preds, golds, n_resamples=77, seed=0)
    assert r.n_resamples == 77

import pytest
```

- [ ] **Step 2: Run and confirm failure**

```bash
python -m pytest tests/test_bootstrap.py -v --tb=short 2>&1 | tail -10
```
Expected: ImportError — `bootstrap` module doesn't exist.

- [ ] **Step 3: Implement `src/clinical_grpo/eval/bootstrap.py`**

```python
"""Bootstrap confidence intervals for micro-code F1.

No scipy dependency — uses pure numpy for reproducibility and minimal deps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from clinical_grpo.eval.f1 import micro_code_f1


@dataclass
class BootstrapResult:
    mean_f1: float
    ci_low: float
    ci_high: float
    n_resamples: int


def bootstrap_f1_ci(
    preds: list[list[str]],
    golds: list[list[str]],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap 95% CI for micro-code F1 over (pred, gold) sample pairs."""
    rng = random.Random(seed)
    n = len(preds)
    scores: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        ps = [preds[i] for i in idx]
        gs = [golds[i] for i in idx]
        scores.append(micro_code_f1(ps, gs).f1)
    scores.sort()
    alpha = 1.0 - ci
    lo = int(alpha / 2 * n_resamples)
    hi = int((1 - alpha / 2) * n_resamples) - 1
    mean_f1 = micro_code_f1(preds, golds).f1
    return BootstrapResult(
        mean_f1=mean_f1,
        ci_low=scores[lo],
        ci_high=scores[hi],
        n_resamples=n_resamples,
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_bootstrap.py -v 2>&1 | tail -10
```
Expected: 5 passed.

- [ ] **Step 5: Wire into `eval/run_eval.py`**

In `src/clinical_grpo/eval/run_eval.py`, import and add bootstrap CI to the report:

```python
from clinical_grpo.eval.bootstrap import bootstrap_f1_ci
```

After the existing `micro_code_f1` / `chapter_f1` calls, add:

```python
    ci = bootstrap_f1_ci(preds, golds, n_resamples=1000, seed=42)
    report["micro_code_f1_bootstrap"] = {
        "mean": ci.mean_f1,
        "ci_low_95": ci.ci_low,
        "ci_high_95": ci.ci_high,
    }
```

- [ ] **Step 6: Ruff + full test suite**

```bash
python -m ruff check src/clinical_grpo/eval/bootstrap.py tests/test_bootstrap.py
python -m pytest tests/ --ignore=tests/test_smoke_train.py --ignore=tests/test_smoke_sft.py -q 2>&1 | tail -3
```
Expected: 0 lint errors, 100 passed.

- [ ] **Step 7: Commit**

```bash
git add src/clinical_grpo/eval/bootstrap.py tests/test_bootstrap.py src/clinical_grpo/eval/run_eval.py
git commit -m "feat: add bootstrap 95% CI for micro-code F1 in eval"
```

---

## Task 3: Data Quality Validation Script

**Files:**
- Create: `scripts/validate_data.py`
- Create: `tests/test_validate_data.py`

Before launching a training run, check the processed parquet for common quality issues: empty summaries, invalid ICD-10 codes, extreme summary lengths, class imbalance in code chapters.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validate_data.py
import pandas as pd
import pytest
from scripts.validate_data import validate_parquet, ValidationReport

def _make_df(summaries, codes):
    return pd.DataFrame({"discharge_summary": summaries, "icd10_codes": codes})

def test_clean_data_returns_no_issues(tmp_path):
    df = _make_df(
        ["Patient with diabetes and hypertension.", "Heart failure case."],
        [["E11.9", "I10"], ["I50.9"]],
    )
    p = tmp_path / "clean.parquet"
    df.to_parquet(p)
    report = validate_parquet(p)
    assert report.n_rows == 2
    assert report.n_empty_summaries == 0
    assert report.n_invalid_codes == 0
    assert report.n_empty_code_lists == 0

def test_detects_empty_summaries(tmp_path):
    df = _make_df(["", "Real note."], [["I10"], ["E11.9"]])
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    report = validate_parquet(p)
    assert report.n_empty_summaries == 1

def test_detects_invalid_codes(tmp_path):
    df = _make_df(
        ["Note one.", "Note two."],
        [["41401", "I10"], ["E11.9"]],   # 41401 is ICD-9
    )
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    report = validate_parquet(p)
    assert report.n_invalid_codes == 1   # one invalid code across all rows

def test_detects_empty_code_lists(tmp_path):
    df = _make_df(["Note.", "Note."], [[], ["E11.9"]])
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    report = validate_parquet(p)
    assert report.n_empty_code_lists == 1

def test_summary_length_stats(tmp_path):
    df = _make_df(["Short.", "A " * 100], [["I10"], ["E11.9"]])
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    report = validate_parquet(p)
    assert report.summary_len_min < report.summary_len_max
    assert report.summary_len_mean > 0
```

- [ ] **Step 2: Run and confirm failure**

```bash
python -m pytest tests/test_validate_data.py -v --tb=short 2>&1 | tail -10
```
Expected: ImportError — `scripts.validate_data` not found.

- [ ] **Step 3: Implement `scripts/validate_data.py`**

```python
"""Validate a processed parquet file before training.

Usage: python scripts/validate_data.py data/processed/mimic.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from clinical_grpo.utils.icd10 import is_valid


@dataclass
class ValidationReport:
    path: str
    n_rows: int
    n_empty_summaries: int
    n_empty_code_lists: int
    n_invalid_codes: int
    summary_len_min: float
    summary_len_max: float
    summary_len_mean: float
    summary_len_p50: float
    summary_len_p95: float
    codes_per_row_mean: float
    codes_per_row_max: int


def validate_parquet(path: Path | str) -> ValidationReport:
    path = Path(path)
    df = pd.read_parquet(path)

    lens = df["discharge_summary"].str.len()
    n_empty_summaries = int((lens == 0).sum())

    n_empty_code_lists = int(df["icd10_codes"].map(len).eq(0).sum())

    n_invalid_codes = int(
        df["icd10_codes"]
        .map(lambda codes: sum(1 for c in codes if not is_valid(c)))
        .sum()
    )

    codes_per_row = df["icd10_codes"].map(len)

    return ValidationReport(
        path=str(path),
        n_rows=len(df),
        n_empty_summaries=n_empty_summaries,
        n_empty_code_lists=n_empty_code_lists,
        n_invalid_codes=n_invalid_codes,
        summary_len_min=float(lens.min()),
        summary_len_max=float(lens.max()),
        summary_len_mean=float(lens.mean()),
        summary_len_p50=float(lens.quantile(0.5)),
        summary_len_p95=float(lens.quantile(0.95)),
        codes_per_row_mean=float(codes_per_row.mean()),
        codes_per_row_max=int(codes_per_row.max()),
    )


def _print_report(r: ValidationReport) -> None:
    ok = "ok" if r.n_empty_summaries == 0 else "WARN"
    print(f"[{ok}] empty summaries: {r.n_empty_summaries}/{r.n_rows}")
    ok = "ok" if r.n_empty_code_lists == 0 else "WARN"
    print(f"[{ok}] empty code lists: {r.n_empty_code_lists}/{r.n_rows}")
    ok = "ok" if r.n_invalid_codes == 0 else "WARN"
    print(f"[{ok}] invalid ICD-10 codes: {r.n_invalid_codes} total")
    print(f"[info] summary length: min={r.summary_len_min:.0f} p50={r.summary_len_p50:.0f} p95={r.summary_len_p95:.0f} max={r.summary_len_max:.0f} chars")
    print(f"[info] codes/row: mean={r.codes_per_row_mean:.1f} max={r.codes_per_row_max}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet", type=Path)
    args = p.parse_args()
    r = validate_parquet(args.parquet)
    _print_report(r)
    if r.n_empty_summaries or r.n_empty_code_lists or r.n_invalid_codes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Make scripts importable**

Add an empty `scripts/__init__.py` so `from scripts.validate_data import ...` works in tests:

```bash
touch /home/user/ClinicalGRPO/scripts/__init__.py
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_validate_data.py -v 2>&1 | tail -10
```
Expected: 5 passed.

- [ ] **Step 6: Lint**

```bash
python -m ruff check scripts/validate_data.py tests/test_validate_data.py
```

- [ ] **Step 7: Commit**

```bash
git add scripts/validate_data.py scripts/__init__.py tests/test_validate_data.py
git commit -m "feat: add data quality validation script with tests"
```

---

## Task 4: Weights & Biases Experiment Tracking

**Files:**
- Create: `configs/wandb.yaml`
- Modify: `src/clinical_grpo/training/train_grpo.py`
- Modify: `scripts/train.py`
- Create: `tests/test_wandb_config.py`

W&B logging is already stubbed (`report_to: []`). This task adds a `configs/wandb.yaml` that sets the project/run name, and wires it into `load_config` so passing `--wandb configs/wandb.yaml` enables tracking without touching model or hardware configs.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wandb_config.py
import pytest
import tempfile
from pathlib import Path
from clinical_grpo.training.train_grpo import load_config


def _base_yamls(tmp_path):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text("base_model: Qwen/Qwen3-4B-Instruct\nlora:\n  r: 16\n  alpha: 32\n  dropout: 0.0\n  bias: none\n  target_modules: [q_proj]\n")
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(f"model_config: {model_yaml}\noutput_dir: outputs/grpo\ngrpo:\n  max_steps: 100\n  beta: 0.04\nreport_to: []\n")
    return train_yaml


def test_no_wandb_config_report_to_is_empty(tmp_path):
    cfg = load_config(_base_yamls(tmp_path))
    assert cfg["report_to"] == []


def test_wandb_config_sets_report_to_wandb(tmp_path):
    train_yaml = _base_yamls(tmp_path)
    wandb_yaml = tmp_path / "wandb.yaml"
    wandb_yaml.write_text("report_to: [wandb]\nwandb:\n  project: clinical-grpo\n  run_name: grpo-test\n")
    cfg = load_config(train_yaml, hw_cfg=None, extra_cfgs=[wandb_yaml])
    assert "wandb" in cfg["report_to"]
    assert cfg["wandb"]["project"] == "clinical-grpo"


def test_load_config_accepts_multiple_extra_cfgs(tmp_path):
    train_yaml = _base_yamls(tmp_path)
    extra1 = tmp_path / "e1.yaml"
    extra1.write_text("custom_key: value1\n")
    extra2 = tmp_path / "e2.yaml"
    extra2.write_text("custom_key: value2\n")
    cfg = load_config(train_yaml, extra_cfgs=[extra1, extra2])
    assert cfg["custom_key"] == "value2"   # last extra wins
```

- [ ] **Step 2: Run and confirm failure**

```bash
python -m pytest tests/test_wandb_config.py -v --tb=short 2>&1 | tail -10
```
Expected: 2-3 tests fail on `load_config` not accepting `extra_cfgs`.

- [ ] **Step 3: Update `load_config` signature in `train_grpo.py`**

```python
def load_config(
    train_cfg: Path | str,
    hw_cfg: Path | str | None = None,
    extra_cfgs: list[Path | str] | None = None,
) -> dict[str, Any]:
    with open(train_cfg) as f:
        cfg = yaml.safe_load(f)
    if hw_cfg:
        with open(hw_cfg) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    for extra in (extra_cfgs or []):
        with open(extra) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    model_cfg_path = Path(cfg.get("model_config", "configs/model.yaml"))
    with open(model_cfg_path) as f:
        cfg["model"] = yaml.safe_load(f)
    return cfg
```

- [ ] **Step 4: Expose `--wandb` in `scripts/train.py`**

```python
def main(
    config: Path = Path("configs/train.yaml"),
    hw: Path = Path("configs/hardware/rtx4080.yaml"),
    wandb: Path | None = None,
    max_steps: int | None = None,
    resume_from: str | None = None,
) -> None:
    extra = [wandb] if wandb is not None else []
    cfg = load_config(config, hw, extra_cfgs=extra)
    if resume_from is not None:
        cfg["resume_from"] = resume_from
    adapter = train(cfg, max_steps_override=max_steps)
    print(f"adapter saved to: {adapter}")
```

- [ ] **Step 5: Create `configs/wandb.yaml`**

```yaml
# W&B experiment tracking config. Pass via: scripts/train.py --wandb configs/wandb.yaml
# Requires: pip install wandb && wandb login

report_to: [wandb]

wandb:
  project: clinical-grpo
  run_name: null        # set to a string to name this run, else auto-generated
  tags: [grpo, icd10, qwen3-4b]
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_wandb_config.py -v 2>&1 | tail -10
```
Expected: 3 passed.

- [ ] **Step 7: Full suite + lint**

```bash
python -m ruff check src/clinical_grpo/training/train_grpo.py scripts/train.py tests/test_wandb_config.py
python -m pytest tests/ --ignore=tests/test_smoke_train.py --ignore=tests/test_smoke_sft.py -q 2>&1 | tail -3
```

- [ ] **Step 8: Commit**

```bash
git add src/clinical_grpo/training/train_grpo.py scripts/train.py configs/wandb.yaml tests/test_wandb_config.py
git commit -m "feat: add W&B experiment tracking config and --wandb CLI flag"
```

---

## Verification

After all four tasks, the full suite should show ~115 tests passing:
- 95 existing + 3 (checkpoint) + 5 (bootstrap) + 5 (validate) + 3 (wandb) = 111 minimum

```bash
python -m pytest tests/ --ignore=tests/test_smoke_train.py --ignore=tests/test_smoke_sft.py -q
python -m ruff check .
python -m mypy src/clinical_grpo/utils/ src/clinical_grpo/rewards/ src/clinical_grpo/prompts/ src/clinical_grpo/eval/f1.py src/clinical_grpo/eval/bootstrap.py src/clinical_grpo/data/dataset.py src/clinical_grpo/data/preprocess.py --ignore-missing-imports
```
