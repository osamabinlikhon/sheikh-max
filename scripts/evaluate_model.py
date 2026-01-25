#!/usr/bin/env python3
"""Evaluate a trained Sheikh-Max model on a prepared dataset.
This script runs inference and computes simple metrics:
- percent of outputs that include interleaved thinking (<think> tags)
- average length of thinking content
- (Optional) saves a JSON report
"""
import argparse
import json
import os
import re
from pathlib import Path


def extract_thinking(response: str) -> str:
    m = re.search(r"<think>(.*?)</think>", response, flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def load_dataset_from_disk(path: str):
    from datasets import load_from_disk
    return load_from_disk(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OsamaBinLikhon/sheikh-max", help="Model path or HF ID")
    parser.add_argument("--dataset", type=str, default="./data/sample", help="Prepared dataset path")
    parser.add_argument("--output", type=str, default="./results/eval_report.json", help="Output report path")
    parser.add_argument("--no-unsloth", action="store_true", help="Don't use Unsloth even if available")
    args = parser.parse_args()

    ds = load_dataset_from_disk(args.dataset)
    print(f"Loaded dataset with {len(ds)} samples")

    # Try to load Unsloth if available and not disabled
    model = None
    tokenizer = None
    if not args.no_unsloth:
        try:
            from unsloth import FastLanguageModel
            import torch
            model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=512, dtype=torch.float16, load_in_4bit=True)
            FastLanguageModel.for_inference(model)
            print("Loaded model with Unsloth")
        except Exception as e:
            print("Unsloth not available or failed to load model:", e)
            tokenizer = None

    if not tokenizer:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto" if torch.cuda.is_available() else None)
            print("Loaded model with transformers")
        except Exception as e:
            print("Could not load model for evaluation:", e)
            return

    metrics = {
        "total": 0,
        "with_think": 0,
        "avg_think_len": 0.0,
        "samples": []
    }

    for i, example in enumerate(ds):
        # Expect format similar to prepare_data.create_sample_dataset()
        # Build a user prompt from the first user message
        messages = example.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            continue
        prompt = user_msgs[-1]["content"]

        # Simple apply: reuse tokenizer.chat_template methods if available
        try:
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt")
            device = "cuda" if hasattr(model, "device") and str(model.device).startswith("cuda") else "cpu"
            inputs = inputs.to(device)
            outputs = model.generate(input_ids=inputs, max_new_tokens=256, use_cache=True)
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=False)
        except Exception as e:
            print("Generation failed for sample", i, e)
            continue

        thinking = extract_thinking(response)
        metrics["total"] += 1
        if thinking:
            metrics["with_think"] += 1
            metrics["avg_think_len"] += len(thinking)
        metrics["samples"].append({"prompt": prompt, "thinking": thinking, "response": response})

    if metrics["total"]:
        metrics["avg_think_len"] = metrics["avg_think_len"] / metrics["total"]
        metrics["think_rate"] = metrics["with_think"] / metrics["total"]
    else:
        metrics["avg_think_len"] = 0.0
        metrics["think_rate"] = 0.0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Evaluation complete. Report saved to:", args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
