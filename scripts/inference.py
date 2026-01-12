#!/usr/bin/env python3
"""
Inference script for Sheikh-Max model.
Loads the model and verifies interleaved thinking with <think> tags.
"""

import argparse
import os
import torch

# Default model path
DEFAULT_MODEL = "OsamaBinLikhon/sheikh-max"
MAX_SEQ_LENGTH = 512


def load_model(model_path=DEFAULT_MODEL, use_unsloth=True):
    """
    Load the Sheikh-Max model.
    
    Args:
        model_path: HuggingFace model ID or local path
        use_unsloth: Use Unsloth for optimized inference (requires GPU)
    
    Returns:
        tuple: (model, tokenizer)
    """
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=torch.float16,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            print(f"✅ Loaded model with Unsloth: {model_path}")
        except ImportError:
            print("⚠️ Unsloth not available, falling back to transformers")
            use_unsloth = False
    
    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"✅ Loaded model with transformers: {model_path}")
    
    # Load custom chat template
    template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "sheikh_chat_template.jinja")
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            tokenizer.chat_template = f.read()
        print(f"✅ Loaded custom chat template from {template_path}")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, system_prompt=None, max_new_tokens=512, temperature=0.7):
    """
    Generate a response from the model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User prompt
        system_prompt: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        str: Generated response
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(model, "device"):
        device = model.device
    inputs = inputs.to(device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=False)
    return response


def check_thinking(response):
    """Check if the response contains interleaved thinking."""
    has_think_open = "<think>" in response
    has_think_close = "</think>" in response
    
    if has_think_open and has_think_close:
        # Extract thinking content
        start = response.find("<think>") + len("<think>")
        end = response.find("</think>")
        thinking = response[start:end].strip()
        return True, thinking
    return False, None


def main():
    parser = argparse.ArgumentParser(description="Sheikh-Max Inference")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path or HF ID")
    parser.add_argument("--prompt", type=str, default="Write a Python function to calculate factorial with reasoning.", help="User prompt")
    parser.add_argument("--system", type=str, default="You are Sheikh-Max, an AI that thinks step-by-step before coding.", help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--no-unsloth", action="store_true", help="Don't use Unsloth")
    args = parser.parse_args()
    
    print(f"🚀 Sheikh-Max Inference")
    print(f"Model: {args.model}")
    print("-" * 50)
    
    model, tokenizer = load_model(args.model, use_unsloth=not args.no_unsloth)
    
    print(f"\n📝 Prompt: {args.prompt}\n")
    print("-" * 50)
    
    response = generate_response(
        model, tokenizer, args.prompt,
        system_prompt=args.system,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print("📤 Generated Response:")
    print(response)
    print("-" * 50)
    
    # Check for interleaved thinking
    has_thinking, thinking_content = check_thinking(response)
    if has_thinking:
        print("✅ Interleaved thinking detected!")
        print(f"💭 Thinking: {thinking_content[:200]}..." if len(thinking_content) > 200 else f"💭 Thinking: {thinking_content}")
    else:
        print("❌ No interleaved thinking found in response.")


if __name__ == "__main__":
    main()