#!/usr/bin/env python3
"""
Push trained LoRA adapters or merged model to Hugging Face Hub.
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo


def push_to_hub(model_path, repo_name, token=None, private=False):
    """
    Push a model folder to Hugging Face Hub.
    
    Args:
        model_path: Local path to the model folder
        repo_name: HuggingFace repo ID (username/repo-name)
        token: HuggingFace API token (uses HF_TOKEN env var if not provided)
        private: Whether to create a private repo
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set and no token provided")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_name, token=token, private=private, exist_ok=True)
        print(f"✅ Repository ready: {repo_name}")
    except Exception as e:
        print(f"⚠️ Could not create repo (may already exist): {e}")
    
    # Upload the folder
    print(f"📤 Uploading {model_path} to {repo_name}...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"✅ Model pushed to https://huggingface.co/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Push Sheikh-Max model to HuggingFace Hub")
    parser.add_argument("--model-path", type=str, default="./results/final", help="Local model path")
    parser.add_argument("--repo-name", type=str, default="OsamaBinLikhon/sheikh-max-lora", help="HuggingFace repo ID")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or use HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    args = parser.parse_args()
    
    push_to_hub(args.model_path, args.repo_name, args.token, args.private)


if __name__ == "__main__":
    main()