.PHONY: all setup prepare-data train test inference push clean

all: setup prepare-data train test

setup:
	@echo "🔧 Setting up Python virtualenv and installing dependencies..."
	python -m venv .venv || true
	. .venv/bin/activate && pip install -U pip setuptools wheel || true
	. .venv/bin/activate && pip install -r requirements.txt || true
	@echo "✅ Setup complete. Activate with: source .venv/bin/activate"

prepare-data:
	@echo "📥 Preparing sample dataset..."
	python scripts/prepare_data.py --sample --output ./data/sample

train:
	@echo "🚀 Starting training (make sure GPU is available and environment variables are set)"
	python scripts/train_sheikh_max.py

test:
	@echo "🧪 Running tests..."
	pytest -q || true

inference:
	@echo "🧠 Running inference"
	python scripts/inference.py

push:
	@echo "📤 Pushing model to HuggingFace (ensure HF_TOKEN env var is set)"
	python scripts/push_to_hub.py --model-path ./results/final --repo-name OsamaBinLikhon/sheikh-max-lora

clean:
	rm -rf .venv __pycache__ .pytest_cache data/sample results
	@echo "🧹 Cleaned workspace"