"""ZeroGPU Gradio Space: discharge summary -> ICD-10 JSON.

Deploy to Hugging Face Spaces with hardware=zero-gpu. The `@spaces.GPU`
decorator allocates a GPU only for the duration of the call.
"""

from __future__ import annotations

import json
import os

import gradio as gr

ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "TODO/clinical-grpo-qwen3-4b-icd10")

try:
    import spaces  # type: ignore
except ImportError:  # local dev without the spaces package
    class _NoSpaces:
        @staticmethod
        def GPU(fn):
            return fn

    spaces = _NoSpaces()  # type: ignore


_pipeline = None


def _load():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_REPO,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    _pipeline = (model, tokenizer)
    return _pipeline


@spaces.GPU
def predict(summary: str) -> str:
    from clinical_grpo.prompts.templates import build_chat
    from clinical_grpo.rewards.composite import parse_completion

    model, tokenizer = _load()
    text = tokenizer.apply_chat_template(
        build_chat(summary), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    codes = parse_completion(completion) or []
    return json.dumps({"raw": completion, "parsed_codes": codes}, indent=2)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=20, label="Discharge summary"),
    outputs=gr.Code(language="json", label="Predicted ICD-10 codes"),
    title="ClinicalGRPO",
    description="Qwen3-4B-Instruct fine-tuned with GRPO for ICD-10-CM coding.",
)


if __name__ == "__main__":
    demo.launch()
