# =====================================================================
# Makefile for Food101 Project (portable version)
# =====================================================================
# This Makefile is designed to:
# 1. Activate a Conda environment
# 2. Launch MLflow UI
# 3. Run training scripts
# =====================================================================

# -----------------------
# Configuration
# -----------------------
# Conda environment name (can be overridden by ENV variable)
CONDA_ENV ?= food101mini
PYTHON ?= python

# -----------------------
# Activate conda environment
# -----------------------
# This target only prints instructions. Use it once per session.
activate:
	@echo "To activate the conda environment run:"
	@echo "  conda activate $(CONDA_ENV)"

# -----------------------
# Launch MLflow UI
# -----------------------
# Make sure the environment is activated before running.
ui:
	@echo "Launching MLflow UI..."
	$(PYTHON) -m mlflow ui
	@echo "MLflow UI running at http://127.0.0.1:5000"

# -----------------------
# Train the model
# -----------------------
# Make sure the environment is activated before running.
train:
	@echo "Starting training..."
	$(PYTHON) -m scripts.train
	@echo "Training finished!"
