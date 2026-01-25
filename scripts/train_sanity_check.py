#!/usr/bin/env python3
"""Sanity checks to run before training on a GPU instance.
Checks:
- Torch availability and CUDA
- GPU memory
- Training arguments have fp16 True and bf16 False
"""
import shutil
import subprocess
import sys
import json

from train_config import get_default_training_kwargs


def check_torch():
    try:
        import torch
        print("torch version:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                # Query basic GPU info (nvidia-smi may not be present on some systems)
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.free --format=csv,noheader,nounits"], stderr=subprocess.DEVNULL)
                print(out.decode())
            except Exception:
                print("nvidia-smi not available or failed; continuing")
        return True
    except Exception as e:
        print("Torch not available or import failed:", e)
        return False


def check_training_kwargs():
    cfg = get_default_training_kwargs()
    print("Default training kwargs:")
    print(json.dumps(cfg, indent=2))
    assert cfg.get("fp16", False) is True, "fp16 must be True for T4 compatibility"
    assert cfg.get("bf16", False) is False, "bf16 must be disabled on T4"
    print("Training kwargs are safe for T4 (fp16=True, bf16=False)")


if __name__ == "__main__":
    ok = check_torch()
    check_training_kwargs()
    if not ok:
        print("Warning: Torch/CUDA not available on this machine. Training must be run on GPU-enabled instance.")
        sys.exit(2)
    print("Sanity check passed. You're ready to start training on a GPU-enabled instance.")
