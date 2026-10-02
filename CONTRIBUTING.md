# Contributing

Thanks for your interest in ClinicalGRPO. Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install   # optional, runs ruff and basic hygiene hooks on commit
```

## Before opening a pull request

Run the same checks as CI (`.github/workflows/ci.yml`):

```bash
ruff check .
python -m mypy src/clinical_grpo/utils/ src/clinical_grpo/rewards/ \
  src/clinical_grpo/prompts/ src/clinical_grpo/eval/f1.py \
  src/clinical_grpo/data/dataset.py src/clinical_grpo/data/preprocess.py \
  --ignore-missing-imports
pytest tests/ -q --ignore=tests/test_smoke_train.py
```

Changes to data processing, rewards, prompts, evaluation or ICD-10 utilities should come with a test that fails before the change and passes after it. The training loops in `src/clinical_grpo/training/` are covered by the GPU smoke tests instead; if you touch them, run `pytest tests/test_smoke_train.py -k one_step -x` on a CUDA machine and say so in the pull request.

Treat the `{"codes": [...]}` output contract in `src/clinical_grpo/prompts/templates.py` and the parser in `src/clinical_grpo/rewards/composite.py` as one unit: change both together or rewards silently collapse to penalties.

## Data

Never commit MIMIC-III data, other credentialed clinical data, or model outputs that contain note text. Use Synthea or hand-written synthetic examples in tests and issues. `data/` and `outputs/` are gitignored.

## Style

Ruff handles linting (line length 100). Keep pull requests focused, with a short description of what changed and how you verified it.
