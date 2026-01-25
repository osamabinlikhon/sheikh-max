#!/usr/bin/env python3
"""Run the standard Sheikh-Max pipeline steps in order or selectively.
Usage:
    python scripts/run_pipeline.py --steps setup prepare-data train test
"""
import subprocess
import argparse
import sys

STEPS = {
    "setup": "bash scripts/setup_env.sh",
    "prepare-data": "python scripts/prepare_data.py --sample --output ./data/sample",
    "train": "python scripts/train_sheikh_max.py",
    "test": "pytest -q",
    "inference": "python scripts/inference.py",
    "push": "python scripts/push_to_hub.py --model-path ./results/final --repo-name OsamaBinLikhon/sheikh-max-lora",
}


def run(cmd: str):
    print(f"> {cmd}")
    return subprocess.call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", choices=STEPS.keys(), default=["prepare-data", "train", "test"], help="Pipeline steps to run")
    args = parser.parse_args()

    for step in args.steps:
        rc = run(STEPS[step])
        if rc != 0:
            print(f"❌ Step '{step}' failed with exit code {rc}")
            sys.exit(rc)

    print("✅ Pipeline completed successfully")


if __name__ == "__main__":
    main()
