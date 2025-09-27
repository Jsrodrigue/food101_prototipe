import random
from pathlib import Path

import mlflow.pytorch
import numpy as np
import torch
from mlflow.models.signature import infer_signature


def log_pytorch_model_mlflow(model, train_dataloader, model_name="best_model"):
    """
    Logs a PyTorch model to MLflow with proper input_example and signature.

    Args:
        model (torch.nn.Module): The trained PyTorch model
        train_dataloader (DataLoader): Dataloader to take example input
        model_name (str): Name under which the model will be saved in MLflow
    """
    # Take a single batch from the dataloader
    example_input = next(iter(train_dataloader))[0][:1]  # tensor of batch size 1

    # Convert tensor to numpy for MLflow
    example_input_np = example_input.cpu().numpy()

    # Run model in no-grad mode
    with torch.no_grad():
        example_output = model(example_input)
    example_output_np = example_output.cpu().numpy()

    # Infer signature
    signature = infer_signature(example_input_np, example_output_np)

    # Log model in MLflow
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name=model_name,  # updated parameter instead of artifact_path
        input_example=example_input_np,
        signature=signature,
    )
    print(f"[INFO] Model logged to MLflow as '{model_name}'")


def save_model(model, target_dir, model_name):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Use .pth or .pt"
    save_path = target_dir / model_name
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to {save_path}")


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
