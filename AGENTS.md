# 🤖 Sheikh-Max: Project Context & Guidelines

**As-salamu Alaykum, Agent!** 👋
Welcome to the **Sheikh-Max** repository. This file is your guide to understanding our mission, tech stack, and constraints. Since we are fine-tuning a Large Language Model (LLM) on limited hardware, please follow the guidelines below carefully.

## 🎯 Mission (লক্ষ্য)
Amader main goal holo **Qwen 2.5 Coder 7B** model-ta ke fine-tune kora jate she "Interleaved Thinking" (Reasoning + Coding) capability achieve korte pare.
*   **Target Hardware:** Google Colab **Tesla T4 GPU** (15GB VRAM).
*   **Output Style:** Model should think inside `<think>...</think>` tags before providing code.

---

## 🛠 Tech Stack (কি দিয়ে বানাচ্ছি)
Agent bhai, code generate korar shomoy nicher library gula mathay rakhben:

-   **Framework:** `Unsloth` (Must use this for T4 optimization).
-   **Base Model:** `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`
-   **Libraries:** `transformers`, `peft`, `trl`, `bitsandbytes`, `wandb`.
-   **Training Method:** QLoRA (Rank 16, 4-bit quantization).
-   **Tracking:** Weights & Biases (W&B).

---

## 📂 Project Structure (ফোল্ডার স্ট্রাকচার)
```text
sheikh-max/
├── notebooks/          # Colab notebooks (Main training logic ekhane)
├── data/               # Dataset processing scripts (Bespoke-Stratos/OpenThoughts)
├── templates/          # Jinja2 chat templates for correct formatting
├── outputs/            # Training checkpoints (LoRA adapters)
├── scripts/            # Utility python scripts (push_to_hub.py, inference.py)
├── AGENTS.md           # You are reading this!
└── README.md           # Public facing documentation
```

---

## ⚠️ Context for AI Agents (এজেন্টদের জন্য বিশেষ নির্দেশনা)

Apni jodi kono code suggest koren ba refactor koren, nicher **Constraint** gula obosshoy follow korben:

1.  **Memory Constraint (Very Important):**
    *   Amader kache matro **15GB VRAM** ache.
    *   ❌ **Don't** suggest full fine-tuning or loading model in 16-bit without quantization.
    *   ✅ **Always** use `load_in_4bit=True` and `gradient_checkpointing=True`.
    *   ✅ Optimizer hisebe always `adamw_8bit` use korben.

2.  **Training Arguments:**
    *   Batch size `2` er beshi baraben na (OOM khabo nahole).
    *   `gradient_accumulation_steps` bariye balance korun (e.g., 4 or 8).
    *   **No BF16:** T4 GPU `bfloat16` support kore na. Always `fp16 = True` rakben.

3.  **Data Formatting:**
    *   Make sure dataset format specifically follows:
        ```text
        ### Instruction: ...
        ### Response: <think>...</think> ...
        ```
    *   Jinja template chara train korle `<think>` tag kaj korbe na.

---

## 📝 Coding Conventions (কোডিং স্টাইল)

*   **Language:** Python (Mainly).
*   **Comments:** Code er moddhe logic bujhanor jonno short comments likhen. (English or Banglish is fine).
*   **Error Handling:** GPU OOM (Out of Memory) handle korar jonno `torch.cuda.empty_cache()` use kora jete pare jodi dorkar hoy.

---

## 🚀 Development Workflow

Jodi natun kono feature add korte hoy:

1.  **Data Prep:** `dataset` load kore age `sheikh_chat_template` apply korte hobe.
2.  **Training:** `Unsloth` er `SFTTrainer` diye train shuru hobe.
3.  **Inference Test:** Training sheshe obosshoy `<think>` tag check korte hobe.
4.  **Push:** Finally, `GGUF` ba `LoRA` adapters HuggingFace e push hobe.

> **Note to Agent:** Bhai, code dewar shomoy "Unsloth" er latest documentation follow koiren. HuggingFace er standard trainer use korle T4 e slow hobe.

---
*Happy Coding! Let's make Sheikh-Max intelligent!* 🚀
