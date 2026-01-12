#!/usr/bin/env python3
"""
Data preparation script for Sheikh-Max training.
Loads datasets and formats them with interleaved thinking (<think> tags).
"""

import argparse
import json
import os
from datasets import load_dataset, Dataset


def format_bespoke_stratos(example):
    """
    Format Bespoke-Stratos dataset for Sheikh-Max training.
    Expected format: conversations with reasoning traces.
    """
    messages = []
    
    # Handle different possible formats
    if "conversations" in example:
        for conv in example["conversations"]:
            role = conv.get("role", conv.get("from", "user"))
            content = conv.get("content", conv.get("value", ""))
            
            # Map roles
            if role in ["human", "user"]:
                messages.append({"role": "user", "content": content})
            elif role in ["gpt", "assistant", "model"]:
                # Check if content has thinking/reasoning
                if "<think>" in content:
                    # Already has think tags
                    think_start = content.find("<think>") + len("<think>")
                    think_end = content.find("</think>")
                    thinking = content[think_start:think_end].strip()
                    response = content[think_end + len("</think>"):].strip()
                    messages.append({
                        "role": "assistant",
                        "thinking": thinking,
                        "content": response
                    })
                else:
                    messages.append({"role": "assistant", "content": content})
            elif role == "system":
                messages.append({"role": "system", "content": content})
    
    return {"messages": messages}


def format_openthoughts(example):
    """
    Format OpenThoughts dataset for Sheikh-Max training.
    OpenThoughts typically has 'thought' and 'response' fields.
    """
    messages = []
    
    if "instruction" in example or "question" in example:
        user_content = example.get("instruction", example.get("question", ""))
        messages.append({"role": "user", "content": user_content})
    
    if "thought" in example or "reasoning" in example:
        thinking = example.get("thought", example.get("reasoning", ""))
        response = example.get("response", example.get("answer", example.get("output", "")))
        messages.append({
            "role": "assistant",
            "thinking": thinking,
            "content": response
        })
    elif "response" in example or "output" in example:
        response = example.get("response", example.get("output", ""))
        messages.append({"role": "assistant", "content": response})
    
    return {"messages": messages}


def format_generic_instruct(example):
    """
    Format generic instruction-following datasets.
    Adds placeholder thinking for datasets without reasoning traces.
    """
    messages = []
    
    # Handle system prompt
    if "system" in example and example["system"]:
        messages.append({"role": "system", "content": example["system"]})
    
    # Handle instruction/input
    instruction = example.get("instruction", example.get("input", example.get("question", "")))
    if instruction:
        messages.append({"role": "user", "content": instruction})
    
    # Handle output/response
    output = example.get("output", example.get("response", example.get("answer", "")))
    if output:
        messages.append({"role": "assistant", "content": output})
    
    return {"messages": messages}


def load_and_format_dataset(dataset_name, split="train", format_type="auto", max_samples=None):
    """
    Load a dataset from HuggingFace and format it for Sheikh-Max training.
    
    Args:
        dataset_name: HuggingFace dataset ID
        split: Dataset split to load
        format_type: Format type ('bespoke', 'openthoughts', 'generic', 'auto')
        max_samples: Maximum number of samples to load
    
    Returns:
        Dataset with 'messages' column
    """
    print(f"📥 Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Auto-detect format
    if format_type == "auto":
        sample = dataset[0]
        if "conversations" in sample:
            format_type = "bespoke"
        elif "thought" in sample or "reasoning" in sample:
            format_type = "openthoughts"
        else:
            format_type = "generic"
        print(f"🔍 Auto-detected format: {format_type}")
    
    # Apply formatting
    format_funcs = {
        "bespoke": format_bespoke_stratos,
        "openthoughts": format_openthoughts,
        "generic": format_generic_instruct
    }
    
    format_func = format_funcs.get(format_type, format_generic_instruct)
    formatted_dataset = dataset.map(format_func, remove_columns=dataset.column_names)
    
    # Filter out empty messages
    formatted_dataset = formatted_dataset.filter(lambda x: len(x["messages"]) >= 2)
    
    print(f"✅ Formatted {len(formatted_dataset)} samples")
    return formatted_dataset


def create_sample_dataset():
    """Create a sample dataset for testing."""
    samples = [
        {
            "messages": [
                {"role": "system", "content": "You are Sheikh-Max, an AI that thinks step-by-step before coding."},
                {"role": "user", "content": "Write a Python function to check if a string is a palindrome."},
                {
                    "role": "assistant",
                    "thinking": "A palindrome reads the same forwards and backwards. I need to:\n1. Handle edge cases (empty string, single char)\n2. Compare the string with its reverse\n3. Consider case sensitivity and spaces",
                    "content": "```python\ndef is_palindrome(s: str) -> bool:\n    # Remove spaces and convert to lowercase\n    s = s.replace(' ', '').lower()\n    return s == s[::-1]\n\n# Examples\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('hello'))    # False\n```"
                }
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain how binary search works."},
                {
                    "role": "assistant",
                    "thinking": "Binary search is a divide-and-conquer algorithm. I should explain:\n1. The prerequisite (sorted array)\n2. The process of halving the search space\n3. Time complexity O(log n)",
                    "content": "Binary search is an efficient algorithm for finding an element in a **sorted** array.\n\n**How it works:**\n1. Start with the middle element\n2. If target equals middle, we're done\n3. If target < middle, search left half\n4. If target > middle, search right half\n5. Repeat until found or search space is empty\n\n**Time Complexity:** O(log n) - much faster than linear search O(n)"
                }
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a recursive function to calculate Fibonacci numbers."},
                {
                    "role": "assistant",
                    "thinking": "Fibonacci: F(n) = F(n-1) + F(n-2), with F(0)=0, F(1)=1.\nRecursive approach is simple but inefficient (O(2^n)).\nI should mention memoization for optimization.",
                    "content": "```python\ndef fibonacci(n: int) -> int:\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# With memoization for efficiency\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci_memo(n: int) -> int:\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    return fibonacci_memo(n - 1) + fibonacci_memo(n - 2)\n```"
                }
            ]
        }
    ]
    return Dataset.from_list(samples)


def save_dataset(dataset, output_path):
    """Save dataset to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dataset.save_to_disk(output_path)
    print(f"💾 Dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Sheikh-Max training")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "bespoke", "openthoughts", "generic"], help="Dataset format")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to load")
    parser.add_argument("--output", type=str, default="./data/prepared", help="Output path")
    parser.add_argument("--sample", action="store_true", help="Create sample dataset for testing")
    args = parser.parse_args()
    
    if args.sample:
        print("🧪 Creating sample dataset...")
        dataset = create_sample_dataset()
    elif args.dataset:
        dataset = load_and_format_dataset(
            args.dataset,
            split=args.split,
            format_type=args.format,
            max_samples=args.max_samples
        )
    else:
        print("❌ Please provide --dataset or use --sample")
        return
    
    if dataset:
        # Show sample
        print("\n📋 Sample formatted data:")
        print(json.dumps(dataset[0], indent=2))
        
        save_dataset(dataset, args.output)
        print(f"\n✅ Dataset ready for training!")
        print(f"   Load with: Dataset.load_from_disk('{args.output}')")


if __name__ == "__main__":
    main()
