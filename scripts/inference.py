#!/usr/bin/env python3
"""
Inference script for Sheikh-Max model
"""

from unsloth import FastLanguageModel
import torch

def load_model(model_path="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", lora_path=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=True
    )
    if lora_path:
        model = FastLanguageModel.get_peft_model(model, r=16)
        model.load_adapter(lora_path)
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
    model, tokenizer = load_model(lora_path="outputs/checkpoint-60")
    response = generate_response(model, tokenizer, "Write a hello world in Python.")
    print(response)