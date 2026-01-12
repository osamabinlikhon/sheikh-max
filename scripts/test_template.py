#!/usr/bin/env python3
"""
Test script for Sheikh-Max chat template.
Verifies that the template correctly formats messages with <think> tags.
Can run without GPU.
"""

import os
from jinja2 import Template

# Load the template
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "templates", "sheikh_chat_template.jinja")

def load_template():
    """Load the Sheikh-Max chat template."""
    with open(TEMPLATE_PATH, "r") as f:
        return Template(f.read())

def test_basic_conversation():
    """Test basic user-assistant conversation."""
    template = load_template()
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]
    result = template.render(
        messages=messages,
        bos_token="<|begin|>",
        eos_token="<|end|>",
        add_generation_prompt=False
    )
    print("=== Test 1: Basic Conversation ===")
    print(result)
    print()
    assert "### Instruction:" in result
    assert "### Response:" in result
    assert "What is 2 + 2?" in result
    assert "The answer is 4." in result
    print("✅ Test 1 passed!\n")

def test_thinking_tags():
    """Test that <think> tags are properly included."""
    template = load_template()
    messages = [
        {"role": "user", "content": "Write a Python function to calculate factorial."},
        {
            "role": "assistant",
            "thinking": "I need to implement factorial. I can use recursion or iteration. Let me use recursion with a base case of n <= 1.",
            "content": "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```"
        }
    ]
    result = template.render(
        messages=messages,
        bos_token="<|begin|>",
        eos_token="<|end|>",
        add_generation_prompt=False
    )
    print("=== Test 2: Thinking Tags ===")
    print(result)
    print()
    assert "<think>" in result
    assert "</think>" in result
    assert "recursion" in result
    assert "def factorial" in result
    print("✅ Test 2 passed!\n")

def test_system_message():
    """Test system message handling."""
    template = load_template()
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]
    result = template.render(
        messages=messages,
        bos_token="<|begin|>",
        eos_token="<|end|>",
        add_generation_prompt=False
    )
    print("=== Test 3: System Message ===")
    print(result)
    print()
    assert "### System:" in result
    assert "helpful coding assistant" in result
    print("✅ Test 3 passed!\n")

def test_generation_prompt():
    """Test that generation prompt is added correctly."""
    template = load_template()
    messages = [
        {"role": "user", "content": "Explain recursion."}
    ]
    result = template.render(
        messages=messages,
        bos_token="<|begin|>",
        eos_token="<|end|>",
        add_generation_prompt=True
    )
    print("=== Test 4: Generation Prompt ===")
    print(result)
    print()
    # Should end with ### Response: for the model to continue
    assert result.strip().endswith("### Response:")
    print("✅ Test 4 passed!\n")

def test_multi_turn_with_thinking():
    """Test multi-turn conversation with thinking."""
    template = load_template()
    messages = [
        {"role": "system", "content": "You are Sheikh-Max, an AI that thinks before coding."},
        {"role": "user", "content": "What is a prime number?"},
        {
            "role": "assistant",
            "thinking": "The user is asking for a definition. I should explain clearly.",
            "content": "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself."
        },
        {"role": "user", "content": "Write code to check if a number is prime."},
        {
            "role": "assistant",
            "thinking": "I need to write a function that checks divisibility. I'll iterate from 2 to sqrt(n) for efficiency.",
            "content": "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"
        }
    ]
    result = template.render(
        messages=messages,
        bos_token="<|begin|>",
        eos_token="<|end|>",
        add_generation_prompt=False
    )
    print("=== Test 5: Multi-turn with Thinking ===")
    print(result)
    print()
    # Count <think> tags - should be 2
    think_count = result.count("<think>")
    assert think_count == 2, f"Expected 2 <think> tags, got {think_count}"
    print("✅ Test 5 passed!\n")

def main():
    """Run all tests."""
    print("🧪 Testing Sheikh-Max Chat Template\n")
    print(f"Template path: {TEMPLATE_PATH}\n")
    
    test_basic_conversation()
    test_thinking_tags()
    test_system_message()
    test_generation_prompt()
    test_multi_turn_with_thinking()
    
    print("=" * 50)
    print("🎉 All tests passed! Template is working correctly.")
    print("=" * 50)

if __name__ == "__main__":
    main()
