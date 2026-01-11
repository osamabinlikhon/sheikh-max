#!/usr/bin/env python3
"""
Inference script for Sheikh-Max model
Loads the merged safetensors model and verifies interleaved thinking.
"""

from unsloth import FastLanguageModel
import torch

def load_model(model_path="sheikh-ai/mistral-7b-sheikh-chat-merged"):  # Update with your HF repo
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.7,
        min_p=0.1
    )
    return tokenizer.batch_decode(outputs)[0]

if __name__ == "__main__":
    model, tokenizer = load_model()
    response = generate_response(model, tokenizer, "Write a Python function to calculate factorial with reasoning.")
    print("Generated Response:")
    print(response)
    
    # Check for <think> tags
    if "<think>" in response and "</think>" in response:
        print("✅ Interleaved thinking detected!")
    else:
        print("❌ No interleaved thinking found.")