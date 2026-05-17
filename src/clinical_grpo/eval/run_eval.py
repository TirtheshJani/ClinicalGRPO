"""Evaluation harness: load adapter, generate, score with F1 (+ optional Groq judge).

Writes a JSON report to outputs/<run>/eval_<split>.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from clinical_grpo.data.dataset import load_grpo_dataset
from clinical_grpo.eval.f1 import chapter_f1, micro_code_f1
from clinical_grpo.prompts.templates import build_chat  # noqa: F401 (re-exported contract)
from clinical_grpo.rewards.composite import parse_completion


def generate_predictions(adapter_path: Path, dataset, max_new_tokens: int = 512) -> list[str]:
    """Run greedy generation against the dataset and return raw completions."""
    from unsloth import FastLanguageModel

    # Adapter dir carries the base model id in its config; let Unsloth read it.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    completions: list[str] = []
    for row in dataset:
        text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0
        )
        completions.append(
            tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        )
    return completions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-stem", type=Path, default=Path("data/processed/mimic"))
    p.add_argument("--judge", choices=["none", "groq"], default="none")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    data_path = args.data_stem.parent / f"{args.data_stem.name}_{args.split}.parquet"
    ds = load_grpo_dataset(data_path)
    completions = generate_predictions(args.adapter, ds)
    preds = [parse_completion(c) or [] for c in completions]
    golds = list(ds["gold_codes"])

    report = {
        "adapter": str(args.adapter),
        "split": args.split,
        "n": len(preds),
        "micro_code_f1": asdict(micro_code_f1(preds, golds)),
        "chapter_f1": asdict(chapter_f1(preds, golds)),
    }

    if args.judge == "groq":
        from clinical_grpo.eval.groq_judge import GroqJudge

        judge = GroqJudge()
        verdicts = [judge.score(g, p) for g, p in zip(golds, preds)]
        report["judge"] = {
            "matched": sum(v.matched for v in verdicts),
            "near_matches": sum(v.near_matches for v in verdicts),
            "spurious": sum(v.spurious for v in verdicts),
            "missed": sum(v.missed for v in verdicts),
        }

    out_path = args.out or args.adapter.parent / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
