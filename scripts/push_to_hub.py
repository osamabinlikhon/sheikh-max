#!/usr/bin/env python3
"""
Push trained LoRA adapters to Hugging Face Hub
"""

from huggingface_hub import HfApi
import os

def push_to_hub(model_path, repo_name, token=None):
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"Model pushed to {repo_name}")

if __name__ == "__main__":
    # Example usage
    push_to_hub("outputs/checkpoint-60", "osamabinlikhon/sheikh-max-lora")