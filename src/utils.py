import os
import random
import shutil
from pathlib import Path

import boto3
import mlflow.pytorch
import numpy as np
import torch


def log_pytorch_model_mlflow(
    model, model_name="model", artifact_dir=None, overwrite=False
):
    """
    Log a PyTorch model to MLflow.

    Args:
        model: The PyTorch model instance to log.
        model_name (str): Name for the model in MLflow (must not contain '/', ':', '.', '%', '"', "'").
        artifact_dir (Path|str|None): Optional folder with extra artifacts to log alongside the model.
        overwrite (bool): If True, deletes any previous model with the same name in this run before logging.
    """
    # MLflow run's artifact root directory (usually something like "mlruns/<run_id>/artifacts")
    artifact_uri = mlflow.get_artifact_uri()
    target_dir = os.path.join(artifact_uri, model_name)

    # If overwrite is enabled and a model with this name already exists, delete it
    if overwrite and os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Log the model to MLflow under the given name (MLflow ≥ 2.16 expects "name" instead of "artifact_path")
    mlflow.pytorch.log_model(model, name=model_name)

    # If extra artifacts are provided, log them into a separate subfolder in MLflow
    if artifact_dir:
        mlflow.log_artifacts(str(artifact_dir), artifact_path=f"{model_name}_artifacts")


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
    mlflow.log_param("train_subset_percentage", cfg.train.scheduler.type)
    mlflow.log_param("unfreeze_layers", cfg.train.unfreeze_layers)
    mlflow.log_param("version", cfg.model.version)
    mlflow.log_param("unfreeze_layers", cfg.train.unfreeze_layers)
    mlflow.log_param("loss_fn", type(loss_fn).__name__)
    if hasattr(cfg.train, "augmentation"):
        mlflow.log_param("augmentation", cfg.train.augmentation)


def upload_folder_to_s3(local_folder: Path, bucket: str, s3_prefix: str):
    """Upload entire folder to S3"""
    s3 = boto3.client("s3")
    for path in local_folder.rglob("*"):
        if path.is_file():
            key = f"{s3_prefix}/{path.relative_to(local_folder)}"
            s3.upload_file(str(path), bucket, key)
            print(f"[S3 Upload] {path} -> s3://{bucket}/{key}")
