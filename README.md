---
license: apache-2.0
base_model: unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit
tags:
  - unsloth
  - qwen2.5
  - coder
  - reasoning
  - chain-of-thought
  - bangladesh
  - sheikh-max
language:
  - en
  - bn
---

# 🚀 Sheikh-Max: The Interleaved Thinking Coder

<div align="center">

![Sheikh-Max Banner](https://img.shields.io/badge/Model-Sheikh--Max-green?style=for-the-badge&logo=android)
![Powered By](https://img.shields.io/badge/Powered_By-Unsloth-blue?style=for-the-badge)
![GPU](https://img.shields.io/badge/Optimized_For-Colab_T4-orange?style=for-the-badge)

**A Reasoning-First Coding Model optimized for constrained hardware.**

[Model on HuggingFace](https://huggingface.co/OsamaBinLikhon/sheikh-max) | [Open in Colab](https://colab.research.google.com/)

</div>

---

## 📖 About The Project

**Sheikh-Max** is a fine-tuned version of the powerful **Qwen 2.5 Coder 7B**, specifically engineered to exhibit **"Interleaved Thinking"**. Unlike standard models that rush to an answer, Sheikh-Max is trained to **plan, reason, and critique** its own logic inside `<think>` tags before generating code.

This project proves that you don't need H100 GPUs to build frontier-class intelligence. Sheikh-Max was trained entirely on a free **Google Colab T4 GPU (15GB VRAM)** using **Unsloth**.

### 🇧🇩 বাংলাদেশী ডেভেলপারদের জন্য (For Bangladeshi Developers)
শেখ-ম্যাক্স (Sheikh-Max) একটি বিশেষ কোডিং মডেল যা উত্তর দেওয়ার আগে চিন্তাভাবনা (Reasoning) করতে পারে। আমরা Qwen 2.5 Coder-কে ফাইন-টিউন করেছি যাতে এটি কোড লেখার আগে `<think>` ট্যাগের মধ্যে প্ল্যানিং করে নেয়। এটি সম্পূর্ণভাবে **Colab T4 GPU**-তে ট্রেইন করা হয়েছে, যা প্রমাণ করে যে রিসোর্স কম থাকলেও বুদ্ধিমান মডেল বানানো সম্ভব।

---

## ✨ Key Features

*   **🧠 Interleaved Thinking:** Generates detailed reasoning traces inside `<think>...</think>` before coding.
*   **⚡ T4 Optimized:** Uses 4-bit quantization (QLoRA) to run efficiently on 15GB VRAM cards (Tesla T4, RTX 3060/4060).
*   **💻 Superior Coding:** Inherits the SOTA coding capabilities of Qwen 2.5 Coder.
*   **🚀 Fast Training:** Built with [Unsloth](https://github.com/unslothai/unsloth) for 2x faster training and 60% less memory usage.

---

## �️ Building the Workspace

To set up and train Sheikh-Max from scratch, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/osamabinlikhon/sheikh-max.git
cd sheikh-max
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Project Structure
```
sheikh-max/
├── notebooks/          # Colab notebooks for training
├── data/               # Dataset processing scripts
├── templates/          # Jinja2 chat templates
├── outputs/            # Training checkpoints
├── scripts/            # Utility scripts (push_to_hub.py, inference.py)
├── requirements.txt    # Python dependencies
├── AGENTS.md           # Project guidelines
└── README.md           # This file
```

### 4. Train the Model
- Open `notebooks/train_sheikh_max.ipynb` in Google Colab.
- Mount your Google Drive and adjust paths if necessary.
- Run all cells to fine-tune the model.
- The notebook handles:
  - Installing libraries
  - Loading the base Qwen 2.5 Coder model
  - Preparing the dataset with interleaved thinking format
  - Fine-tuning with QLoRA on T4 GPU
  - Testing the model for `<think>` tags

### 5. Push to Hugging Face
After training, use the script to upload your LoRA adapters:
```bash
python scripts/push_to_hub.py
```

### 6. Inference
Test your trained model:
```bash
python scripts/inference.py
```

---

## �💡 Example Output

**User:** "Write a Python script to verify if a number is prime using recursion."

**Sheikh-Max:**
```xml
<think>
1.  **Analyze the Request:** The user wants a Python script to check for prime numbers using *recursion*.
2.  **Edge Cases:** Need to handle numbers less than 2 (not prime).
3.  **Recursive Logic:**
    - Base case: If divisor * divisor > n, it's prime.
    - Check: If n % divisor == 0, not prime.
    - Recursive step: Check (n, divisor + 1).
4.  **Optimization:** I should verify inputs to ensure they are integers.
</think>
```

```python
def is_prime_recursive(n, divisor=2):
    if n < 2:
        return False
    if divisor * divisor > n:
        return True
    if n % divisor == 0:
        return False
    return is_prime_recursive(n, divisor + 1)
```

---

## 🛠️ How to Use (Inference)

You need `unsloth` to run this model efficiently.

### 1. Installation
```bash
pip install unsloth
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
```

### 2. Python Code
```python
from unsloth import FastLanguageModel

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "OsamaBinLikhon/sheikh-max", # Load from HF
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Define the prompt with the specific template
prompt = """### Instruction:
Write a Python function to connect to a SQLite database.

### Response:
"""

# Generate
inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
print(tokenizer.batch_decode(outputs)[0])
```

---

## 📊 Training Details

*   **Base Model:** `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`
*   **Dataset:** `Bespoke-Stratos-17k` / `OpenThoughts` (Reasoning Traces).
*   **Hardware:** Google Colab Tesla T4 (1 GPU).
*   **Technique:** QLoRA (Rank 16, Alpha 16).
*   **Optimizer:** AdamW 8-bit.

---

## 🔄 CI/CD with GitHub Actions

Sheikh-Max uses GitHub Actions for automated workflows from idea to production.

### Quickstart
1. **Push Code:** Triggers CI (linting, testing, building).
2. **PR Review:** Automated checks ensure quality.
3. **Release:** Publish to PyPI and HF on release.

### Workflows
- **CI/CD (`python-ci.yml`)**: Lints, tests, builds package.
- **Release (`release.yml`)**: Validates and deploys on release.
- **Package Publish (`publish-package.yml`)**: Publishes to PyPI.

### Understanding
- **Continuous Integration:** Every push runs tests.
- **Continuous Deployment:** Releases auto-deploy.
- **vs GitHub Apps:** Actions automate repos; Apps extend GitHub.

See `.github/workflows/` for details.

---

## 🤝 Contributing & Acknowledgements

This model is a community effort to bring reasoning capabilities to smaller hardware.

*   **Developed by:** [OsamaBinLikhon](https://github.com/osamabinlikhon)
*   **Powered by:** [Unsloth AI](https://unsloth.ai)
*   **Base Model:** [Qwen Team](https://github.com/QwenLM/Qwen2.5)

---
<div align="center">
  Made with ❤️ in Bangladesh 🇧🇩
</div>
