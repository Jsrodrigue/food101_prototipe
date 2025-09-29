import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
from torch import optim
from tqdm import tqdm

from .metrics import compute_metrics
from .utils import log_hyperparams_mlflow, save_model, set_seed, upload_folder_to_s3


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=5, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model=None, save_path=None):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if model and save_path:
                save_model(model, save_path.parent, save_path.name)
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True


def train_step(model, dataloader, loss_fn, optimizer, device, metrics_list=None):
    """
    Perform one training epoch.
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y = y.long()

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs)
        all_labels.append(y)

    avg_loss = total_loss / len(dataloader)
    preds = torch.cat(all_preds).argmax(dim=1)
    labels = torch.cat(all_labels)
    metrics_dict = compute_metrics(labels, preds, metrics_list or ["accuracy"])
    metrics_dict["loss"] = avg_loss
    return metrics_dict


def test_step(model, dataloader, loss_fn, device, metrics_list=None):
    """
    Perform one validation/test epoch.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.long()

            outputs = model(X)
            loss = loss_fn(outputs, y)

            total_loss += loss.item()
            all_preds.append(outputs)
            all_labels.append(y)

    avg_loss = total_loss / len(dataloader)
    preds = torch.cat(all_preds).argmax(dim=1)
    labels = torch.cat(all_labels)
    metrics_dict = compute_metrics(labels, preds, metrics_list or ["accuracy"])
    metrics_dict["loss"] = avg_loss
    return metrics_dict


def train_mlflow(model, train_loader, val_loader, optimizer, loss_fn, cfg, device=None):
    """
    Train a PyTorch model and log metrics, plots, and models to MLflow.

    All artifacts are stored within the MLflow run folder.
    """
    set_seed(cfg.train.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set scheduler if defined
    scheduler = None
    if cfg.train.scheduler.type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    elif cfg.train.scheduler.type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.scheduler.step_size,
            gamma=cfg.train.scheduler.gamma,
        )

    # Setup MLflow
    mlflow_dir = Path(cfg.outputs.local.mlflow.path) if cfg.outputs.mode == "local" else Path(cfg.outputs.aws.mlflow.path)
    exp_name = cfg.outputs.local.mlflow.experiment_name if cfg.outputs.mode == "local" else cfg.outputs.aws.mlflow.experiment_name
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlflow_dir.resolve()}")
    mlflow.set_experiment(exp_name)

    run_name = cfg.outputs.run_name or f"run_{int(time.time())}"

    # Initialize early stopping
    best_model_path = None  # will store model inside MLflow artifact_path
    early_stopper = EarlyStopping(patience=cfg.train.early_stop_patience, verbose=True)

    # Prepare metrics storage
    results = {f"train_{m}": [] for m in cfg.train.metrics + ["loss"]}
    results.update({f"val_{m}": [] for m in cfg.train.metrics + ["loss"]})

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        log_hyperparams_mlflow(cfg, loss_fn)

        for epoch in tqdm(range(cfg.train.epochs), desc="Training"):
            # Train and validate
            train_metrics = train_step(model, train_loader, loss_fn, optimizer, device, cfg.train.metrics)
            val_metrics = test_step(model, val_loader, loss_fn, device, cfg.train.metrics)

            # Scheduler step
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            # Log metrics to MLflow
            for key in train_metrics:
                mlflow.log_metric(f"train_{key}", train_metrics[key], step=epoch)
                mlflow.log_metric(f"val_{key}", val_metrics[key], step=epoch)
                results[f"train_{key}"].append(train_metrics[key])
                results[f"val_{key}"].append(val_metrics[key])

            # Print metrics
            train_str = ", ".join([f"{k} {v:.4f}" for k, v in train_metrics.items()])
            val_str = ", ".join([f"{k} {v:.4f}" for k, v in val_metrics.items()])
            print(f"Epoch {epoch+1}: Train: {train_str}")
            print(f"        Val: {val_str}\n")

            # Early stopping, save best model inside MLflow run
            if val_metrics["loss"] < (early_stopper.best_loss or float("inf")):
                best_model_path = "checkpoints/best_model.pth"
                mlflow.pytorch.log_model(model, artifact_path="checkpoints")
            early_stopper(val_metrics["loss"])

            if early_stopper.early_stop:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

        # Plot and log loss curve inside MLflow
        plt.figure(figsize=(10, 5))
        plt.plot(results["train_loss"], label="train_loss")
        plt.plot(results["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plot_path = Path("loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")

    # Upload to S3 if using AWS
    if cfg.outputs.mode == "aws":
        upload_folder_to_s3(
            mlflow_dir, cfg.dataset.aws.s3_bucket, f"{cfg.outputs.aws.s3_prefix}/mlruns"
        )

    return results
