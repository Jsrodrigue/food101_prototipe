import os
import random
import shutil
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import torch


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


def upload_folder_to_s3(local_folder: Path, bucket: str, s3_prefix: str):
    """
    Upload all files in local_folder recursively to S3 under s3_prefix.
    """
    s3 = boto3.client("s3")
    for file_path in local_folder.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_folder)
            s3_key = f"{s3_prefix}/{relative_path.as_posix()}"
            print(f"[UPLOAD] {file_path} -> s3://{bucket}/{s3_key}")
            s3.upload_file(str(file_path), bucket, s3_key)


def log_hyperparams_mlflow(cfg, loss_fn):
    mlflow.log_param("model_name", cfg.model.name)
    mlflow.log_param("pretrained", cfg.model.pretrained)
    mlflow.log_param("batch_size", cfg.train.batch_size)
    mlflow.log_param("optimizer_lr", cfg.train.optimizer.lr)
    mlflow.log_param("scheduler_type", cfg.train.scheduler.type)
    mlflow.log_param("train_subset_percentage", cfg.train.subset_percentage)
    mlflow.log_param("unfreeze_layers", cfg.train.unfreeze_layers)
    mlflow.log_param("version", cfg.model.version)
    mlflow.log_param("epochs", cfg.train.epochs)
    mlflow.log_param("unfreeze_layers", cfg.train.unfreeze_layers)
    mlflow.log_param("loss_fn", type(loss_fn).__name__)
    mlflow.log_param("augmentation", cfg.train.augmentation)


def upload_folder_to_s3(local_folder: Path, bucket: str, s3_prefix: str):
    """Upload entire folder to S3"""
    s3 = boto3.client("s3")
    for path in local_folder.rglob("*"):
        if path.is_file():
            key = f"{s3_prefix}/{path.relative_to(local_folder)}"
            s3.upload_file(str(path), bucket, key)
            print(f"[S3 Upload] {path} -> s3://{bucket}/{key}")


# -------------------- LOGGING & PLOTTING -------------------- #


def log_loss_curve(results, filename="loss_curve.png"):
    """
    Plot and save the training and validation loss curve.

    Args:
        results (dict): Dictionary containing training and validation metrics.
                        Expected keys: "train_loss", "val_loss".
        filename (str): Name of the output PNG file where the plot will be saved.

    Returns:
        Path: Path object pointing to the saved PNG file
    """
    plt.figure(figsize=(10, 5))
    plt.plot(results["train_loss"], label="train_loss")
    plt.plot(results["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return Path(filename)

# -------------------- LOG LOSS CURVE -------------------- #
def log_loss_curve_mlflow(plot_path: Path, artifact_path="plots"):
    """
    Log a loss curve plot to MLflow safely.
    
    Args:
        plot_path: Path object pointing to local plot file (PNG, PDF, etc.).
        artifact_path: Folder name inside MLflow run to store the plot.
    
    Notes:
        - Ensures artifact is uploaded before deleting local file.
        - Prints the MLflow run ID and artifact URI for reference.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("No active MLflow run. Call mlflow.start_run() first.")

    # Log artifact to MLflow
    mlflow.log_artifact(str(plot_path), artifact_path=artifact_path)

    # Get final artifact URI
    artifact_uri = mlflow.get_artifact_uri(f"{artifact_path}/{plot_path.name}")
    run_id = mlflow.active_run().info.run_id
    print(f"[INFO] Plot logged in MLflow run {run_id}")
    print(f"[INFO] Artifact URI: {artifact_uri}")

    # Remove local temporary file
    plot_path.unlink()

def log_pytorch_model_state_dict_mlflow(model, model_name="model", overwrite=False):
    """
    Log only the state_dict of a PyTorch model to MLflow.

    This avoids storing the full model object, keeping artifacts lighter
    and making it easier to manage multiple runs.

    Args:
        model (torch.nn.Module): The PyTorch model instance to log.
        model_name (str): Name for the model artifact in MLflow (no special chars).
        overwrite (bool): If True, deletes any previous artifact with the same name in this run.
    """

    # Get the current MLflow run's artifact URI
    artifact_uri = mlflow.get_artifact_uri()
    target_dir = os.path.join(artifact_uri, model_name)

    # Remove previous artifact if overwrite is True
    if overwrite and os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Save state_dict to a temporary file
    state_path = Path(f"{model_name}_state_dict.pth")
    torch.save(model.state_dict(), state_path)

    # Log the state_dict as an artifact in MLflow
    mlflow.log_artifact(str(state_path), artifact_path=model_name)

    # Remove the temporary local file
    state_path.unlink()



def update_best_model(model, val_loss, best_loss):
    """
    Save the best model to MLflow if the validation loss improves.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        val_loss (float): Current epoch's validation loss.
        best_loss (float): Current best validation loss seen so far.
        mlflow_artifact_dir (str or Path): Directory for storing additional artifacts in MLflow.

    Returns:
        float: Updated best validation loss (either unchanged or updated if improvement).
    """
    if val_loss < best_loss:
        best_loss = val_loss
        log_pytorch_model_state_dict_mlflow(
            model, model_name="model", overwrite=True
        )
    return best_loss
