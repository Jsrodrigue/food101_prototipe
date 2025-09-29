


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

ui-aws:
	mlflow ui --backend-store-uri sqlite:///mlruns_aws/mlflow.db --default-artifact-root ./mlruns/artifacts
