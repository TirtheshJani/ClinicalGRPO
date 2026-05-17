# Phase 2: Deployment Readiness Implementation Plan

> **For agentic workers:** All four tasks are fully independent — dispatch in parallel (dispatching-parallel-agents). Each agent owns one task. Strict TDD: failing test first, watch it fail, minimal implementation.

**Goal:** Make a trained adapter usable: clean inference API, batch inference script, HF Hub push, and hyperparameter sweep tooling.

**Architecture:** `inference.py` defines the public API other scripts call. `batch_infer.py` and `push_to_hub.py` are thin scripts. `sweep.py` is config-only (no GPU). All heavy imports inside method bodies.

**Tech Stack:** Unsloth (lazy import), HuggingFace Hub, pandas, tyro.

---

## Task 1: Inference Module

**Files:**
- Create: `src/clinical_grpo/inference.py`
- Create: `tests/test_inference.py`

Single class `ICD10Predictor` that loads a LoRA adapter and exposes `predict(summary: str) -> list[str]` and `predict_batch(summaries: list[str]) -> list[list[str]]`. Heavy imports inside `__init__`.

### Step 1 — Write failing tests

```python
# tests/test_inference.py
import json
from unittest.mock import MagicMock, patch
import pytest
from clinical_grpo.inference import ICD10Predictor


def _make_predictor(tmp_path, completions):
    """Build an ICD10Predictor with mocked Unsloth internals."""
    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "prompt"
    mock_tok.return_value = {"input_ids": MagicMock(shape=[1, 10])}
    mock_tok.decode.side_effect = completions
    mock_model.generate.return_value = MagicMock()
    mock_model.device = "cpu"

    mock_fl = MagicMock()
    mock_fl.from_pretrained.return_value = (mock_model, mock_tok)

    fake_unsloth = MagicMock()
    fake_unsloth.FastLanguageModel = mock_fl

    import sys
    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        predictor = ICD10Predictor(str(tmp_path / "adapter"), load_in_4bit=False)
    predictor._model = mock_model
    predictor._tokenizer = mock_tok
    return predictor


def test_predict_returns_list_of_str(tmp_path):
    completion = json.dumps({"codes": ["E11.9", "I10"]})
    p = _make_predictor(tmp_path, [completion])
    result = p.predict("Patient with diabetes.")
    assert isinstance(result, list)
    assert all(isinstance(c, str) for c in result)


def test_predict_parses_codes_from_completion(tmp_path):
    completion = json.dumps({"codes": ["E11.9", "I10"]})
    p = _make_predictor(tmp_path, [completion])
    assert p.predict("note") == ["E11.9", "I10"]


def test_predict_returns_empty_on_malformed_json(tmp_path):
    p = _make_predictor(tmp_path, ["not json"])
    assert p.predict("note") == []


def test_predict_batch_returns_list_per_summary(tmp_path):
    completions = [
        json.dumps({"codes": ["E11.9"]}),
        json.dumps({"codes": ["I10"]}),
    ]
    p = _make_predictor(tmp_path, completions)
    results = p.predict_batch(["note one", "note two"])
    assert len(results) == 2
    assert results[0] == ["E11.9"]
    assert results[1] == ["I10"]


def test_adapter_path_stored(tmp_path):
    p = _make_predictor(tmp_path, [])
    assert p.adapter_path == str(tmp_path / "adapter")
```

### Step 2 — Run and confirm ImportError
```bash
python -m pytest tests/test_inference.py -v --tb=short 2>&1 | tail -8
```

### Step 3 — Implement `src/clinical_grpo/inference.py`

```python
"""ICD-10 inference wrapper for a trained LoRA adapter."""

from __future__ import annotations

from clinical_grpo.prompts.templates import build_chat
from clinical_grpo.rewards.composite import parse_completion


class ICD10Predictor:
    def __init__(
        self,
        adapter_path: str,
        max_new_tokens: int = 512,
        load_in_4bit: bool = True,
        max_seq_length: int = 2048,
    ) -> None:
        from unsloth import FastLanguageModel
        self.adapter_path = adapter_path
        self._max_new_tokens = max_new_tokens
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer

    def _generate(self, summary: str) -> str:
        text = self._tokenizer.apply_chat_template(
            build_chat(summary), tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
        )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def predict(self, summary: str) -> list[str]:
        return parse_completion(self._generate(summary)) or []

    def predict_batch(self, summaries: list[str]) -> list[list[str]]:
        return [self.predict(s) for s in summaries]
```

### Step 4 — Run tests (all 5 must pass)
```bash
python -m pytest tests/test_inference.py -v 2>&1 | tail -10
```

### Step 5 — Lint + full suite
```bash
python -m ruff check src/clinical_grpo/inference.py tests/test_inference.py
python -m pytest tests/ --ignore=tests/test_smoke_train.py --ignore=tests/test_smoke_sft.py -q 2>&1 | tail -2
```

### Step 6 — Commit
```bash
git add src/clinical_grpo/inference.py tests/test_inference.py
git commit -m "feat: add ICD10Predictor inference wrapper"
```

---

## Task 2: Batch Inference Script

**Files:**
- Create: `scripts/batch_infer.py`
- Create: `tests/test_batch_infer.py`

Reads a parquet with `discharge_summary` column, runs `ICD10Predictor.predict_batch`, writes predictions to an output parquet with `predicted_codes` column.

### Step 1 — Write failing tests

```python
# tests/test_batch_infer.py
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_batch_infer_adds_predicted_codes_column(tmp_path):
    from unittest.mock import MagicMock, patch
    from scripts.batch_infer import run_batch_infer

    df = pd.DataFrame({
        "discharge_summary": ["Diabetes note.", "Heart failure note."],
        "icd10_codes": [["E11.9"], ["I50.9"]],
    })
    in_path = tmp_path / "in.parquet"
    df.to_parquet(in_path)
    out_path = tmp_path / "out.parquet"

    mock_predictor = MagicMock()
    mock_predictor.predict_batch.return_value = [["E11.9"], ["I50.9"]]

    run_batch_infer(mock_predictor, in_path, out_path)

    result = pd.read_parquet(out_path)
    assert "predicted_codes" in result.columns
    assert len(result) == 2


def test_batch_infer_preserves_existing_columns(tmp_path):
    from unittest.mock import MagicMock
    from scripts.batch_infer import run_batch_infer

    df = pd.DataFrame({
        "discharge_summary": ["Note."],
        "icd10_codes": [["I10"]],
        "subject_id": [42],
    })
    in_path = tmp_path / "in.parquet"
    df.to_parquet(in_path)
    out_path = tmp_path / "out.parquet"

    mock_predictor = MagicMock()
    mock_predictor.predict_batch.return_value = [["I10"]]

    run_batch_infer(mock_predictor, in_path, out_path)
    result = pd.read_parquet(out_path)
    assert "subject_id" in result.columns
    assert "discharge_summary" in result.columns


def test_batch_infer_cli_help():
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/batch_infer.py", "--help"],
        capture_output=True, text=True, cwd="/home/user/ClinicalGRPO"
    )
    assert r.returncode == 0
    assert "adapter" in r.stdout.lower()
```

### Step 2 — Run and confirm failure
```bash
python -m pytest tests/test_batch_infer.py -v --tb=short 2>&1 | tail -8
```

### Step 3 — Implement `scripts/batch_infer.py`

```python
"""Run batch ICD-10 inference on a parquet file.

Usage: python scripts/batch_infer.py --adapter outputs/grpo/adapter --input data/processed/mimic_test.parquet --out predictions.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tyro

from clinical_grpo.inference import ICD10Predictor


def run_batch_infer(predictor: ICD10Predictor, input_path: Path, out_path: Path) -> None:
    df = pd.read_parquet(input_path)
    preds = predictor.predict_batch(df["discharge_summary"].tolist())
    df["predicted_codes"] = preds
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")


def main(
    adapter: Path,
    input: Path = Path("data/processed/mimic_test.parquet"),
    out: Path = Path("outputs/predictions.parquet"),
    load_in_4bit: bool = True,
) -> None:
    predictor = ICD10Predictor(str(adapter), load_in_4bit=load_in_4bit)
    run_batch_infer(predictor, input, out)


if __name__ == "__main__":
    tyro.cli(main)
```

### Step 4 — Run tests
```bash
python -m pytest tests/test_batch_infer.py -v 2>&1 | tail -10
```

### Step 5 — Lint + full suite
```bash
python -m ruff check scripts/batch_infer.py tests/test_batch_infer.py
python -m pytest tests/ --ignore=tests/test_smoke_train.py --ignore=tests/test_smoke_sft.py -q 2>&1 | tail -2
```

### Step 6 — Commit
```bash
git add scripts/batch_infer.py tests/test_batch_infer.py
git commit -m "feat: add batch inference script"
```

---

## Task 3: HuggingFace Hub Push Script

**Files:**
- Create: `scripts/push_to_hub.py`
- Create: `tests/test_push_to_hub.py`

Uploads the adapter directory to HF Hub using `huggingface_hub`. Accepts `--adapter`, `--repo-id`, `--private` flag. No dependency on inference module — just file upload.

### Step 1 — Write failing tests

```python
# tests/test_push_to_hub.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_push_cli_help():
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/push_to_hub.py", "--help"],
        capture_output=True, text=True, cwd="/home/user/ClinicalGRPO"
    )
    assert r.returncode == 0
    assert "repo" in r.stdout.lower()


def test_push_to_hub_calls_upload_folder(tmp_path):
    from unittest.mock import patch, MagicMock
    from scripts.push_to_hub import push_adapter

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"base_model": "Qwen/Qwen3-4B-Instruct"}')

    mock_api = MagicMock()
    with patch("scripts.push_to_hub.HfApi", return_value=mock_api):
        push_adapter(str(adapter_dir), "user/clinical-grpo-test", private=True)

    mock_api.create_repo.assert_called_once()
    mock_api.upload_folder.assert_called_once()
    call_kwargs = mock_api.upload_folder.call_args.kwargs
    assert call_kwargs["repo_id"] == "user/clinical-grpo-test"
    assert call_kwargs["folder_path"] == str(adapter_dir)


def test_push_raises_on_missing_adapter(tmp_path):
    from scripts.push_to_hub import push_adapter
    import pytest
    with pytest.raises(FileNotFoundError):
        push_adapter(str(tmp_path / "nonexistent"), "user/repo")
```

### Step 2 — Run and confirm failure
```bash
python -m pytest tests/test_push_to_hub.py -v --tb=short 2>&1 | tail -8
```

### Step 3 — Implement `scripts/push_to_hub.py`

```python
"""Push a trained LoRA adapter to HuggingFace Hub.

Usage: python scripts/push_to_hub.py --adapter outputs/grpo/adapter --repo-id your-user/clinical-grpo-qwen3-4b
"""

from __future__ import annotations

from pathlib import Path

import tyro
from huggingface_hub import HfApi


def push_adapter(adapter_path: str, repo_id: str, private: bool = False) -> None:
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(adapter_dir),
        repo_type="model",
        commit_message="Upload ClinicalGRPO LoRA adapter",
    )
    print(f"Pushed {adapter_dir} → https://huggingface.co/{repo_id}")


def main(
    adapter: Path,
    repo_id: str,
    private: bool = False,
) -> None:
    push_adapter(str(adapter), repo_id, private=private)


if __name__ == "__main__":
    tyro.cli(main)
```

### Step 4 — Run tests
```bash
python -m pytest tests/test_push_to_hub.py -v 2>&1 | tail -10
```

### Step 5 — Lint + full suite
```bash
python -m ruff check scripts/push_to_hub.py tests/test_push_to_hub.py
python -m pytest tests/ --ignore=tests/test_smoke_train.py --ignore=tests/test_smoke_sft.py -q 2>&1 | tail -2
```

### Step 6 — Commit
```bash
git add scripts/push_to_hub.py tests/test_push_to_hub.py
git commit -m "feat: add HuggingFace Hub push script"
```

---

## Task 4: Hyperparameter Sweep

**Files:**
- Create: `configs/sweep.yaml`
- Create: `scripts/sweep.py`
- Create: `tests/test_sweep.py`

Sweep over GRPO hyperparameters (beta, learning_rate, num_generations) by generating config variants, running training with `max_steps` override, and logging results to a CSV.

### Step 1 — Write failing tests

```python
# tests/test_sweep.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_sweep_config_parses(tmp_path):
    import yaml
    sweep_path = Path("configs/sweep.yaml")
    assert sweep_path.exists(), "configs/sweep.yaml must exist"
    data = yaml.safe_load(sweep_path.read_text())
    assert "sweep" in data
    assert "params" in data["sweep"]


def test_generate_configs_returns_one_per_combination():
    from scripts.sweep import generate_configs
    params = {
        "grpo.beta": [0.01, 0.04],
        "grpo.learning_rate": [1e-5, 5e-6],
    }
    configs = generate_configs(params)
    assert len(configs) == 4  # 2 * 2


def test_generate_configs_sets_override_values():
    from scripts.sweep import generate_configs
    params = {"grpo.beta": [0.01]}
    configs = generate_configs(params)
    assert configs[0]["grpo.beta"] == 0.01


def test_apply_override_sets_nested_key():
    from scripts.sweep import apply_override
    cfg = {"grpo": {"beta": 0.04, "lr": 5e-6}}
    apply_override(cfg, "grpo.beta", 0.01)
    assert cfg["grpo"]["beta"] == 0.01
    assert cfg["grpo"]["lr"] == 5e-6   # untouched


def test_sweep_cli_help():
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/sweep.py", "--help"],
        capture_output=True, text=True, cwd="/home/user/ClinicalGRPO"
    )
    assert r.returncode == 0
```

### Step 2 — Run and confirm failure
```bash
python -m pytest tests/test_sweep.py -v --tb=short 2>&1 | tail -8
```

### Step 3 — Create `configs/sweep.yaml`

```yaml
# GRPO hyperparameter sweep configuration.
# Run: python scripts/sweep.py --config configs/train.yaml --sweep configs/sweep.yaml --max-steps 50

sweep:
  params:
    grpo.beta: [0.01, 0.04, 0.1]
    grpo.learning_rate: [1.0e-5, 5.0e-6]
    grpo.num_generations: [4, 8]
  max_steps: 50        # short run per config for sweep
  output_csv: outputs/sweep_results.csv
```

### Step 4 — Implement `scripts/sweep.py`

```python
"""GRPO hyperparameter sweep.

Usage: python scripts/sweep.py --config configs/train.yaml --hw configs/hardware/rtx4080.yaml --sweep configs/sweep.yaml
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any

import yaml
import tyro

from clinical_grpo.training.train_grpo import load_config


def apply_override(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def generate_configs(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(params.keys())
    combos = list(itertools.product(*[params[k] for k in keys]))
    return [{k: v for k, v in zip(keys, combo)} for combo in combos]


def main(
    config: Path = Path("configs/train.yaml"),
    hw: Path = Path("configs/hardware/rtx4080.yaml"),
    sweep: Path = Path("configs/sweep.yaml"),
) -> None:
    sweep_cfg = yaml.safe_load(sweep.read_text())["sweep"]
    overrides = generate_configs(sweep_cfg["params"])
    max_steps = sweep_cfg.get("max_steps", 50)
    out_csv = Path(sweep_cfg.get("output_csv", "outputs/sweep_results.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    from clinical_grpo.training.train_grpo import train

    results = []
    for i, override in enumerate(overrides):
        cfg = load_config(config, hw)
        for k, v in override.items():
            apply_override(cfg, k, v)
        run_dir = out_csv.parent / f"sweep_{i:03d}"
        cfg["output_dir"] = str(run_dir)
        print(f"\n[sweep {i+1}/{len(overrides)}] {override}")
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
    print(f"\nSweep complete. Results: {out_csv}")


if __name__ == "__main__":
    tyro.cli(main)
```

### Step 5 — Run tests
```bash
python -m pytest tests/test_sweep.py -v 2>&1 | tail -10
```

### Step 6 — Lint + full suite
```bash
python -m ruff check scripts/sweep.py configs/sweep.yaml tests/test_sweep.py
python -m pytest tests/ --ignore=tests/test_smoke_train.py --ignore=tests/test_smoke_sft.py -q 2>&1 | tail -2
```

### Step 7 — Commit
```bash
git add scripts/sweep.py configs/sweep.yaml tests/test_sweep.py
git commit -m "feat: add hyperparameter sweep script and config"
```

---

## Verification

After all four tasks:

```bash
python -m pytest tests/ --ignore=tests/test_smoke_train.py --ignore=tests/test_smoke_sft.py -q
# Expected: ~131 passed (114 + 5 + 3 + 3 + 5)

python -m ruff check .
# Expected: 0 errors

python scripts/batch_infer.py --help
python scripts/push_to_hub.py --help
python scripts/sweep.py --help
# Expected: all exit 0
```
