#!/usr/bin/env python3
"""
Sheikh-Max Training Script
Fine-tunes Mistral-7B-Instruct-v0.2 with interleaved thinking on a dummy dataset.
Optimized for Google Colab T4 GPU with 4-bit quantization and QLoRA.
"""

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Set environment variable to enable expandable memory segments for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Explicitly clear memory from previous attempts if any models are still loaded ---
print("Clearing GPU memory from previous runs...")
torch.cuda.empty_cache()
import gc
gc.collect()

# -----------------------------------------------------------------------------------
# 1. Load the base model and tokenizer using Unsloth's optimized methods
model_id = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
print(f"Loading model and tokenizer with Unsloth: {model_id}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=512,  # Consistent with dataset tokenization max_length
    dtype=torch.float16,  # Use float16 for T4 compatibility
    load_in_4bit=True,  # Unsloth handles 4-bit quantization optimally
)

# Set pad_token if not already set (often needed for generation tasks)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Assign the defined sheikh_chat_template to the tokenizer's chat_template attribute
sheikh_chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "[INST] <<SYS>>\n{{ message['content'] }}\n<</SYS>>\n"
    "{% elif message['role'] == 'user' %}"
    "[INST] {{ message['content'] }} [/INST]"
    "{% elif message['role'] == 'assistant' %}"
    "{% if message['thinking'] %}"
    "<think>{{ message['thinking'] }}</think>"
    "{% endif %}"
    "{{ message['content'] }} "  # Added space for separation if thinking and content both exist
    "{% endif %}"
    "{% endfor %}"
    "{{ eos_token }}"
)
print("Assigning custom chat template to tokenizer...")
tokenizer.chat_template = sheikh_chat_template

# 2. Prepare dummy dataset
print("Preparing dummy dataset...")
dummy_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant that thinks step-by-step."},
            {"role": "user", "content": "Explain the concept of recursion to a 10-year-old using a simple analogy."},
            {"role": "assistant", "thinking": "I need to find a relatable analogy for a 10-year-old, like Russian nesting dolls or a story within a story, to explain recursion. I'll describe the process of breaking down a problem.", "content": "Imagine you have a big treasure chest. Inside it, there's a smaller treasure chest, and inside that, an even smaller one, until you get to the tiny chest with the actual treasure! Recursion is like that: solving a problem by breaking it into smaller, identical versions of itself, until you reach the smallest, easiest version to solve directly. Then, you build the solution back up."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is the capital of Japan?"},
            {"role": "assistant", "thinking": "The user is asking for a basic geographical fact. I need to retrieve the capital city of Japan.", "content": "The capital of Japan is Tokyo."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What are the main steps in making a good cup of coffee?"},
            {"role": "assistant", "thinking": "I should list the key stages: bean selection, grinding, brewing method, and water quality. I'll make sure to mention the importance of each step.", "content": "Making a great cup of coffee involves several key steps: First, start with high-quality, freshly roasted beans. Second, grind them just before brewing to the appropriate coarseness for your chosen method. Third, use clean, filtered water at the right temperature (around 195-205°F or 90-96°C). Finally, choose your preferred brewing method (like pour-over, French press, or espresso) and follow its specific instructions carefully for the best extraction."}
        ]
    }
]

dataset_raw = Dataset.from_list(dummy_data)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer.apply_chat_template(
        examples["messages"],
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset_raw.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})

# 3. Initialize a LoraConfig for QLoRA
print("Initializing LoraConfig parameters...")
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_bias = "none"
lora_target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Apply PEFT adapters
print("Preparing model for training with PEFT adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_r,
    target_modules=lora_target_modules,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias=lora_bias,
)

# Set the Hugging Face repository name
hub_model_name = "mistral-7b-sheikh-chat"
hf_username = os.getenv("HF_USERNAME", "sheikh-ai")
hub_model_id = f"{hf_username}/{hub_model_name}"

# 4. Configure TrainingArguments
print("Configuring TrainingArguments...")
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_steps=500,
    push_to_hub=True,
    report_to="wandb",
    save_strategy="epoch",
    hub_model_id=hub_model_id,
    hub_private_repo=False,
    remove_unused_columns=True,
    gradient_checkpointing=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    hub_token=os.environ.get("HF_TOKEN"),
)

# 5. Initialize the SFTTrainer
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_arguments,
    train_dataset=tokenized_dataset,
    max_seq_length=512,
)

# 6. Start the fine-tuning process
print("Starting fine-tuning...")
trainer.train()

# 7. Push the fine-tuned model and tokenizer to the Hugging Face Hub
print(f"Pushing model to Hugging Face Hub: {hub_model_id}...")
trainer.push_to_hub()

# 8. Merge and push the full model in safetensors format
print("Merging LoRA adapters and pushing full model in safetensors...")
model.push_to_hub_merged(
    hub_model_id + "-merged",
    tokenizer,
    save_method="merged_16bit",
    token=os.environ.get("HF_TOKEN")
)

print("Training complete! Model pushed to HF.")