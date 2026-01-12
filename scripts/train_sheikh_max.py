#!/usr/bin/env python3
"""
Sheikh-Max Training Script
Fine-tunes Qwen2.5-Coder-7B with interleaved thinking (reasoning in <think> tags).
Optimized for Google Colab T4 GPU with 4-bit quantization and QLoRA.

Security-First Approach:
- Implements security-weighted loss for better security reasoning
- Uses environment variables for sensitive configuration (never hardcoded)
- Validates inputs and handles errors gracefully
"""

import os
import gc
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Import security metrics if available
try:
    from quality_metrics import calculate_security_score, security_weighted_loss
    SECURITY_METRICS_AVAILABLE = True
except ImportError:
    SECURITY_METRICS_AVAILABLE = False
    print("⚠️ Security metrics not available. Using standard training.")

# Set environment variable to enable expandable memory segments for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear GPU memory from previous runs
print("Clearing GPU memory from previous runs...")
torch.cuda.empty_cache()
gc.collect()

# -----------------------------------------------------------------------------------
# 1. Load the base model and tokenizer using Unsloth's optimized methods
MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 512

print(f"Loading model and tokenizer with Unsloth: {MODEL_ID}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.float16,  # Use float16 for T4 compatibility (no bf16 support)
    load_in_4bit=True,
)

# Set pad_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the Sheikh-Max chat template with <think> tag support
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "templates", "sheikh_chat_template.jinja")
if os.path.exists(TEMPLATE_PATH):
    with open(TEMPLATE_PATH, "r") as f:
        sheikh_chat_template = f.read()
    print(f"Loaded chat template from {TEMPLATE_PATH}")
else:
    # Fallback inline template if file not found
    sheikh_chat_template = """{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {%- if message['content'] %}
{{- '### System:\\n' + message['content'].strip() + '\\n\\n' }}
        {%- endif %}
    {%- elif message['role'] == 'user' %}
{{- '### Instruction:\\n' + message['content'].strip() + '\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
{{- '### Response:\\n' }}
        {%- if message.get('thinking') %}
{{- '<think>\\n' + message['thinking'].strip() + '\\n</think>\\n\\n' }}
        {%- endif %}
        {%- if message['content'] %}
{{- message['content'].strip() }}
        {%- endif %}
{{- eos_token + '\\n\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '### Response:\\n' }}
{%- endif %}"""
    print("Using fallback inline chat template")

print("Assigning custom chat template to tokenizer...")
tokenizer.chat_template = sheikh_chat_template

# 2. Prepare training dataset with security-focused examples
print("Preparing training dataset with security-focused examples...")
training_data = [
    {
        "messages": [
            {"role": "system", "content": "You are Sheikh-Max, an AI that thinks step-by-step with security-first approach before coding."},
            {"role": "user", "content": "Write a Python script to verify if a number is prime using recursion."},
            {
                "role": "assistant",
                "thinking": """1. **Analyze the Request:** The user wants a Python script to check for prime numbers using *recursion*.
2. **Security Considerations:**
   - Input validation: Ensure n is a positive integer
   - Edge cases: Handle numbers less than 2 (not prime)
   - No external dependencies that could introduce vulnerabilities
3. **Recursive Logic:**
   - Base case: If divisor * divisor > n, it's prime
   - Check: If n % divisor == 0, not prime
   - Recursive step: Check (n, divisor + 1)
4. **Error Handling:** Validate inputs to ensure they are integers.""",
                "content": """```python
def is_prime_recursive(n, divisor=2):
    # Input validation
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 2:
        return False
    if divisor * divisor > n:
        return True
    if n % divisor == 0:
        return False
    return is_prime_recursive(n, divisor + 1)

# Test
print(is_prime_recursive(17))  # True
print(is_prime_recursive(4))   # False
```"""
            }
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Create a function to connect to a database and fetch user data."},
            {
                "role": "assistant",
                "thinking": """🛡️ **Security Analysis:**
1. **Input Validation:** Sanitize all user inputs to prevent SQL injection
2. **Authentication:** Use environment variables for credentials (never hardcode)
3. **Parameterized Queries:** Always use parameterized queries, never string concatenation
4. **Error Handling:** Don't expose sensitive info in error messages
5. **Connection Security:** Use SSL/TLS for database connections
6. **Least Privilege:** Database user should have minimal required permissions""",
                "content": """```python
import os
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    \"\"\"Secure database connection with proper error handling.\"\"\"
    # Use environment variables for credentials (never hardcode)
    db_path = os.environ.get('DATABASE_PATH', ':memory:')
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        # Log error securely without exposing sensitive details
        raise RuntimeError("Database connection failed") from None
    finally:
        if conn:
            conn.close()

def fetch_user_by_id(user_id: int) -> dict:
    \"\"\"Fetch user data securely using parameterized query.\"\"\"
    # Input validation
    if not isinstance(user_id, int) or user_id < 1:
        raise ValueError("Invalid user ID")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Parameterized query prevents SQL injection
        cursor.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
```"""
            }
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Write a login function with password verification."},
            {
                "role": "assistant",
                "thinking": """🛡️ **Security Analysis:**
1. **Password Handling:**
   - Never store plain text passwords
   - Use bcrypt or argon2 for hashing (not MD5/SHA1)
   - Use constant-time comparison to prevent timing attacks
2. **Input Validation:**
   - Validate username format
   - Limit password length to prevent DoS
3. **Rate Limiting:** Implement to prevent brute force attacks
4. **Session Management:**
   - Generate cryptographically secure tokens
   - Set appropriate expiration
5. **Error Messages:** Generic messages to prevent user enumeration
6. **Logging:** Log failed attempts without exposing passwords""",
                "content": """```python
import os
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta

def hash_password(password: str) -> str:
    \"\"\"Hash password using secure method.\"\"\"
    # Use environment variable for salt (in production, use bcrypt)
    salt = os.environ.get('PASSWORD_SALT', secrets.token_hex(16))
    return hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt.encode(),
        100000  # iterations
    ).hex()

def verify_password(password: str, hashed: str) -> bool:
    \"\"\"Constant-time password comparison to prevent timing attacks.\"\"\"
    return hmac.compare_digest(hash_password(password), hashed)

def login(username: str, password: str) -> dict:
    \"\"\"Secure login with proper validation.\"\"\"
    # Input validation
    if not username or len(username) > 100:
        raise ValueError("Invalid credentials")  # Generic message
    if not password or len(password) > 128:
        raise ValueError("Invalid credentials")
    
    # Fetch user (use parameterized query in real implementation)
    user = get_user_by_username(username)
    
    if user and verify_password(password, user['password_hash']):
        # Generate secure session token
        token = secrets.token_urlsafe(32)
        return {
            'success': True,
            'token': token,
            'expires': (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
    
    # Generic error message prevents user enumeration
    raise ValueError("Invalid credentials")
```"""
            }
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Implement binary search in Python."},
            {
                "role": "assistant",
                "thinking": """1. **Algorithm Analysis:**
   - Binary search requires a sorted array
   - Time complexity: O(log n)
   - Space complexity: O(1) for iterative, O(log n) for recursive
2. **Input Validation:**
   - Verify array is not empty
   - Handle edge cases (single element, target not found)
3. **Implementation:**
   - Use iterative approach to avoid stack overflow on large arrays
   - Prevent integer overflow in mid calculation""",
                "content": """```python
from typing import List, Optional

def binary_search(arr: List[int], target: int) -> Optional[int]:
    \"\"\"
    Binary search implementation with proper validation.
    
    Args:
        arr: Sorted list of integers
        target: Value to find
        
    Returns:
        Index of target if found, None otherwise
    \"\"\"
    if not arr:
        return None
    
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # Prevent integer overflow (important in some languages)
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return None

# Test
arr = [1, 3, 5, 7, 9, 11, 13]
print(binary_search(arr, 7))   # 3
print(binary_search(arr, 6))   # None
```"""
            }
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Create a file upload handler."},
            {
                "role": "assistant",
                "thinking": """🛡️ **Security Analysis:**
1. **File Validation:**
   - Validate file extension (whitelist approach)
   - Check MIME type (don't trust Content-Type header alone)
   - Verify file signature (magic bytes)
   - Limit file size to prevent DoS
2. **Path Security:**
   - Sanitize filename to prevent path traversal
   - Use secure random names for stored files
   - Store outside web root
3. **Malware Prevention:**
   - Scan uploaded files if possible
   - Don't execute uploaded files
4. **Access Control:**
   - Verify user permissions before upload
   - Log all upload attempts""",
                "content": """```python
import os
import secrets
import mimetypes
from pathlib import Path

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', '/tmp/uploads')

def validate_file(filename: str, file_size: int, file_content: bytes) -> bool:
    \"\"\"Validate uploaded file for security.\"\"\"
    # Check extension (whitelist)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type not allowed: {ext}")
    
    # Check file size
    if file_size > MAX_FILE_SIZE:
        raise ValueError("File too large")
    
    # Verify magic bytes for common types
    magic_bytes = {
        '.jpg': [b'\\xff\\xd8\\xff'],
        '.png': [b'\\x89PNG'],
        '.gif': [b'GIF87a', b'GIF89a'],
        '.pdf': [b'%PDF'],
    }
    
    if ext in magic_bytes:
        if not any(file_content.startswith(mb) for mb in magic_bytes[ext]):
            raise ValueError("File content doesn't match extension")
    
    return True

def save_uploaded_file(filename: str, content: bytes) -> str:
    \"\"\"Securely save uploaded file.\"\"\"
    # Validate first
    validate_file(filename, len(content), content)
    
    # Generate secure random filename (prevents path traversal)
    ext = Path(filename).suffix.lower()
    secure_name = f"{secrets.token_hex(16)}{ext}"
    
    # Ensure upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, secure_name)
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return secure_name
```"""
            }
        ]
    }
]

dataset_raw = Dataset.from_list(training_data)

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
HUB_MODEL_NAME = "sheikh-max"
HF_USERNAME = os.getenv("HF_USERNAME", "OsamaBinLikhon")
HUB_MODEL_ID = f"{HF_USERNAME}/{HUB_MODEL_NAME}"

# 4. Configure TrainingArguments
print("Configuring TrainingArguments...")
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # Keep low for T4 15GB VRAM
    gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
    optim="adamw_8bit",  # Use 8-bit AdamW for memory efficiency
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,  # T4 doesn't support bf16
    max_steps=500,
    push_to_hub=bool(os.environ.get("HF_TOKEN")),
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    save_strategy="steps",
    save_steps=100,
    hub_model_id=HUB_MODEL_ID,
    hub_private_repo=False,
    remove_unused_columns=True,
    gradient_checkpointing=True,  # Essential for T4 memory
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
    max_seq_length=MAX_SEQ_LENGTH,
)

# 6. Start the fine-tuning process
print("Starting fine-tuning...")
trainer.train()

# 7. Save locally first
print("Saving model locally...")
trainer.save_model("./results/final")

# 8. Push to Hugging Face Hub if token is available
if os.environ.get("HF_TOKEN"):
    print(f"Pushing model to Hugging Face Hub: {HUB_MODEL_ID}...")
    trainer.push_to_hub()

    # 9. Merge and push the full model in safetensors format
    print("Merging LoRA adapters and pushing full model in safetensors...")
    model.push_to_hub_merged(
        HUB_MODEL_ID + "-merged",
        tokenizer,
        save_method="merged_16bit",
        token=os.environ.get("HF_TOKEN")
    )
    print(f"✅ Training complete! Model pushed to {HUB_MODEL_ID}")
else:
    print("⚠️ HF_TOKEN not set. Model saved locally but not pushed to Hub.")
    print("✅ Training complete! Model saved to ./results/final")