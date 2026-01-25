"""Training configuration utilities for Sheikh-Max.
Centralizes defaults so they can be reused by training, sanity checks, and CI.
"""
from transformers import TrainingArguments
import os


def get_default_training_kwargs():
    """Return a dict of default training kwargs suitable for T4 (QLoRA/4bit)"""
    return {
        "output_dir": os.environ.get("RESULTS_DIR", "./results"),
        "per_device_train_batch_size": int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", 2)),
        "gradient_accumulation_steps": int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 4)),
        "optim": os.environ.get("OPTIMIZER", "adamw_8bit"),
        "logging_steps": int(os.environ.get("LOGGING_STEPS", 10)),
        "learning_rate": float(os.environ.get("LEARNING_RATE", 2e-4)),
        "fp16": True,   # Use FP16 on T4 (bf16 not supported)
        "bf16": False,  # Explicitly disable bf16
        "max_steps": int(os.environ.get("MAX_STEPS", 500)),
        "push_to_hub": bool(os.environ.get("HF_TOKEN")),
        "report_to": "wandb" if os.environ.get("WANDB_API_KEY") else "none",
        "save_strategy": os.environ.get("SAVE_STRATEGY", "steps"),
        "save_steps": int(os.environ.get("SAVE_STEPS", 100)),
        "hub_model_id": os.environ.get("HUB_MODEL_ID", None),
        "hub_private_repo": False,
        "remove_unused_columns": True,
        "gradient_checkpointing": True,
        "warmup_ratio": float(os.environ.get("WARMUP_RATIO", 0.03)),
        "lr_scheduler_type": os.environ.get("LR_SCHEDULER_TYPE", "cosine"),
    }


def build_training_arguments(**overrides) -> TrainingArguments:
    """Construct a Transformers TrainingArguments with defaults merged with overrides."""
    kwargs = get_default_training_kwargs()
    kwargs.update(overrides or {})
    # Ensure bf16 is explicitly False unless the user overrides it to True (not recommended for T4)
    if "bf16" not in kwargs:
        kwargs["bf16"] = False
    return TrainingArguments(**kwargs)
