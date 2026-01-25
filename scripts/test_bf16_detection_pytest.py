import re
import pathlib

# Dangerous patterns to detect (assignment or dtype usage), allow comments/mentions
PATTERNS = [
    r'\bbf16\s*=\s*True',           # bf16=True
    r"\bbf16\s*:\s*True",         # bf16: True (yaml/json)
    r"bfloat16",                     # any explicit 'bfloat16' dtype mention
    r"torch\.bfloat16",             # torch.bfloat16 dtype
]

EXCLUDE_FILES = {"scripts/test_bf16_detection_pytest.py"}
# Skip CI workflow files (they may contain detection patterns on purpose)
SKIP_DIRS = {'.github/workflows'}


def scan_paths(paths):
    bad = []
    for p in paths:
        for f in pathlib.Path(p).rglob('*'):
            if any(str(f).startswith(d) for d in SKIP_DIRS):
                continue
            if str(f) in EXCLUDE_FILES:
                continue
            if f.is_file() and f.suffix in {'.py', '.ipynb', '.md', '.yml', '.yaml'}:
                try:
                    txt = f.read_text(encoding='utf-8')
                except Exception:
                    continue
                for pat in PATTERNS:
                    if re.search(pat, txt, flags=re.IGNORECASE):
                        bad.append((str(f), pat))
                        break
    return bad


def test_no_bf16_in_repo():
    # Scan critical locations: scripts, notebooks, and workflows
    bad = scan_paths(['scripts', 'notebooks', '.github/workflows'])
    assert not bad, f"Found bf16/bfloat references in files: {bad}"


def test_trainingarguments_fp16_set():
    # Ensure TrainingArguments in training script sets fp16=True and does not set bf16=True
    train_file = pathlib.Path('scripts/train_sheikh_max.py')
    txt = train_file.read_text()

    # Extract TrainingArguments(...) block
    m = re.search(r'TrainingArguments\((.*?)\)\s*,', txt, re.S | re.M)
    if not m:
        # fallback: look for TrainingArguments( ... ) up to two hundred lines
        m = re.search(r'TrainingArguments\((.*?)\)\s*\n', txt, re.S | re.M)

    assert m, "Could not find TrainingArguments(...) in scripts/train_sheikh_max.py"
    block = m.group(1)

    assert 'fp16=True' in block.replace(' ', ''), "TrainingArguments should set fp16=True"
    assert 'bf16=True' not in block.replace(' ', ''), "TrainingArguments must not set bf16=True"
