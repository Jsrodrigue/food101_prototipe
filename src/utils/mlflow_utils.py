import datetime
import json
import os
import shutil
from pathlib import Path

import mlflow
import torch
from hydra.utils import get_original_cwd


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


def update_best_model(model, val_metrics, best_loss, cfg, epoch):
    """
    Save the best model to MLflow, log metrics, and store informative artifacts.

    Args:
        model (torch.nn.Module): The PyTorch model.
        val_metrics (dict): Validation metrics of the current epoch (must include 'loss').
        best_loss (float): Current best validation loss.
        cfg: Configuration object containing hyperparameters.
        epoch (int): Current epoch number.

    Returns:
        tuple: (updated best validation loss, bool indicating if it's a new best).
    """
    is_new_best = False

    if val_metrics["loss"] < best_loss:
        best_loss = val_metrics["loss"]
        is_new_best = True

        # --- Log model state dict ---
        log_pytorch_model_state_dict_mlflow(model, model_name="model", overwrite=True)

        # --- Log best metrics to MLflow ---
        for key, value in val_metrics.items():
            mlflow.log_metric(f"best_val_{key}", value)

        # --- Save informative artifact: metrics + hyperparameters ---
        artifact_info = {
            "best_epoch": epoch,
            "metrics": val_metrics,
            "hyperparameters": {
                "model_name": cfg.model.name,
                "version": cfg.model.version,
                "pretrained": cfg.model.pretrained,
                "optimizer_lr": cfg.train.optimizer.lr,
                "batch_size": cfg.train.batch_size,
                "scheduler_type": cfg.train.scheduler.type,
                "unfreeze_layers": cfg.train.unfreeze_layers,
                "augmentation": cfg.train.augmentation,
                "epochs": cfg.train.epochs,
                "subset_percentage": cfg.train.subset_percentage,
                "loss_fn": (
                    str(cfg.train.loss_fn) if hasattr(cfg.train, "loss_fn") else None
                ),
            },
        }

        # Save JSON locally
        json_path = Path("best_model_info.json")
        with open(json_path, "w") as f:
            json.dump(artifact_info, f, indent=4)

        # Log JSON as MLflow artifact
        mlflow.log_artifact(str(json_path), artifact_path="best_model_info")
        json_path.unlink()

        print(
            f"[INFO] New best model saved at epoch {epoch} with val_loss={best_loss:.4f}"
        )

    return best_loss, is_new_best


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
    print(f"[INFO] Plot URI: {artifact_uri}")

    # Remove local temporary file
    plot_path.unlink()


def setup_mlflow(cfg):
    """
    Setup MLflow tracking directory and experiment based on configuration.

    Args:
        cfg: Hydra / OmegaConf configuration containing MLflow settings.

    Returns:
        mlflow_dir (Path): Path to the MLflow tracking directory.
        run_name (str): Name of the current run.
    """
    # Determine MLflow directory
    mlflow_dir = Path(get_original_cwd()) / (
        cfg.outputs.local.mlflow.path
        if cfg.outputs.mode == "local"
        else cfg.outputs.aws.mlflow.path
    )
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_dir.resolve().as_uri())

    # Determine experiment name
    exp_name = (
        cfg.outputs.local.mlflow.experiment_name
        if cfg.outputs.mode == "local"
        else cfg.outputs.aws.mlflow.experiment_name
    )
    mlflow.set_experiment(exp_name)

    # Determine run name
    run_name = cfg.outputs.run_name or datetime.datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )

    return mlflow_dir, run_name
