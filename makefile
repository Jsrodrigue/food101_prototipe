# =====================================================================
# Makefile for Food101 Project
# =====================================================================
# This Makefile is designed to:
# 1. Activate the conda environment 'food101mini'
# 2. Launch MLflow UI
# 3. Run training scripts
# =====================================================================

# Path to the MLflow SQLite database
MLFLOW_DB := C:/Users/Juan/Desktop/food101Mini/mlruns/mlflow.db

# -----------------------
# Activate conda environment
# -----------------------
# This target only activates the environment.
# Use it once per Git Bash session.
activate:
	@echo "Activating conda environment 'food101mini'..."
	. /c/Users/Juan/anaconda3/etc/profile.d/conda.sh && conda activate food101mini && echo "Environment activated!"


# -----------------------
# Launch MLflow UI
# -----------------------
# Make sure the environment is activated before running.
ui:
	@echo "Launching MLflow UI..."
	mkdir -p C:/Users/Juan/Desktop/food101Mini/mlruns
	mlflow ui --backend-store-uri sqlite:///$(MLFLOW_DB)
	@echo "MLflow UI running at http://127.0.0.1:5000"

# -----------------------
# Train the model
# -----------------------
# Make sure the environment is activated before running.
train:
	@echo "Starting training..."
	python -m scripts.train
	@echo "Training finished!"
