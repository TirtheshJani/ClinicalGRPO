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
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
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
