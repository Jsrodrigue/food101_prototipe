

# -----------------------
# Launch MLflow UI
# -----------------------
ui:
	@echo use mlflow ui --backend-store-uri "file:///C:/Users/Juan/Desktop/food101Mini/mlflow" --default-artifact-root "C:/Users/Juan/Desktop/food101Mini/mlflow/artifacts"


# -----------------------
# Launch MLflow UI AWS
# -----------------------
ui-aws:
	@echo "Launching MLflow UI (AWS)..."
	call conda activate $(CONDA_ENV) && ^
	python -m mlflow ui --backend-store-uri ./mlruns_aws --default-artifact-root ./mlruns_aws/artifacts
	@echo "MLflow UI AWS running at http://127.0.0.1:5000"

# -----------------------
# Train the model
# -----------------------
train:
	@echo "Starting training..."
	python -m scripts.train $(ARGS)
	@echo "Training finished!"

