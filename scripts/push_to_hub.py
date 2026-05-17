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


def main(adapter: Path, repo_id: str, private: bool = False) -> None:
    push_adapter(str(adapter), repo_id, private=private)


if __name__ == "__main__":
    tyro.cli(main)
