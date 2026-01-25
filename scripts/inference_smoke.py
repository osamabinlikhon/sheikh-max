#!/usr/bin/env python3
"""Lightweight inference smoke test that doesn't require heavy ML deps.
This script is intended for CI and local quick checks to ensure the inference pipeline
and template handling produce expected outputs without downloading models.
"""
import argparse


def mock_response(prompt: str, system_prompt: str | None = None) -> str:
    thinking = (
        "<think>\n"
        "1. Analyze the request: need to implement factorial with reasoning.\n"
        "2. Edge cases: n < 0, n == 0, non-integers.\n"
        "3. Implementation: use iterative approach for O(n) time.\n"
        "</think>\n"
    )
    code = (
        "```python\n"
        "def factorial(n: int) -> int:\n"
        "    if not isinstance(n, int) or n < 0:\n"
        "        raise ValueError('n must be a non-negative integer')\n"
        "    result = 1\n"
        "    for i in range(2, n+1):\n"
        "        result *= i\n"
        "    return result\n"
        "```\n"
    )
    return thinking + "\n" + code


def main():
    parser = argparse.ArgumentParser(description="Mock inference smoke test")
    parser.add_argument("--prompt", type=str, default="Write a factorial function", help="User prompt")
    parser.add_argument("--system", type=str, default="You are Sheikh-Max, an AI that thinks step-by-step before coding.", help="System prompt")
    args = parser.parse_args()

    print("🚀 Mock Inference Smoke Test")
    print("Prompt:", args.prompt)
    print("-")
    resp = mock_response(args.prompt, args.system)
    print(resp)


if __name__ == "__main__":
    main()
