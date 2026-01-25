#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Creating virtual environment and installing dependencies..."
python -m venv .venv || true
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt || true
if [ -f dev-requirements.txt ]; then
  pip install -r dev-requirements.txt || true
fi

# Create .env from sample if it doesn't exist
if [ ! -f .env ] && [ -f .env.example ]; then
  echo "Creating .env from .env.example (you should edit it with your keys)..."
  cp .env.example .env
fi

# Source .env if present (do not commit .env to the repo)
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -o allexport
  # shellcheck disable=SC1090
  source .env
  set +o allexport
  echo "Loaded environment variables from .env"
else
  echo "No .env file found. Create .env from .env.example and add your WANDB_API_KEY and HF_TOKEN."
fi

echo "✅ Environment ready. Activate with: source .venv/bin/activate"
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
PY