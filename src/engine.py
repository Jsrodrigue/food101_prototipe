# src/train_engine.py

import torch
from torch import optim
from pathlib import Path
import mlflow
from tqdm import tqdm
import datetime
from hydra.utils import get_original_cwd
from .metrics import compute_metrics
from .utils import (
    log_hyperparams_mlflow,
    set_seed,
    upload_folder_to_s3,
    log_loss_curve, 
    update_best_model,
    log_loss_curve_mlflow
)


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

    def __call__(self, val_loss: float):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True


# -------------------- TRAIN & EVAL -------------------- #

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


def eval_one_epoch(model, dataloader, loss_fn, device, metrics_list=None):
    """
    Evaluation step (used for validation or test).
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



# -------------------- MAIN TRAIN MLflow -------------------- #

def train_mlflow(model, train_loader, val_loader, optimizer, loss_fn, cfg, device=None):
    """
    Train a PyTorch model and log metrics, plots, and best model to MLflow.
    """
    set_seed(cfg.train.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Scheduler setup ---
    scheduler = None
    if cfg.train.scheduler.type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    elif cfg.train.scheduler.type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.scheduler.step_size,
            gamma=cfg.train.scheduler.gamma,
        )

    # --- MLflow setup ---
    mlflow_dir = Path(get_original_cwd()) / (cfg.outputs.local.mlflow.path if cfg.outputs.mode == "local" else cfg.outputs.aws.mlflow.path)
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_dir.resolve().as_uri())  # <-- cambio principal
    exp_name = cfg.outputs.local.mlflow.experiment_name if cfg.outputs.mode == "local" else cfg.outputs.aws.mlflow.experiment_name
    mlflow.set_experiment(exp_name)
    run_name = cfg.outputs.run_name or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    early_stopper = EarlyStopping(patience=cfg.train.early_stop_patience, verbose=True)
    best_model_loss = float("inf")

    # --- Metrics storage ---
    results = {f"train_{m}": [] for m in cfg.train.metrics + ["loss"]}
    results.update({f"val_{m}": [] for m in cfg.train.metrics + ["loss"]})

    with mlflow.start_run(run_name=run_name):
        log_hyperparams_mlflow(cfg, loss_fn)

        for epoch in tqdm(range(cfg.train.epochs), desc="Training"):
            # --- TRAIN & VALIDATION ---
            train_metrics = train_step(model, train_loader, loss_fn, optimizer, device, cfg.train.metrics)
            val_metrics = eval_one_epoch(model, val_loader, loss_fn, device, cfg.train.metrics)

            # --- Scheduler step ---
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            # --- Log metrics ---
            for key in train_metrics:
                mlflow.log_metric(f"train_{key}", train_metrics[key], step=epoch)
                mlflow.log_metric(f"val_{key}", val_metrics[key], step=epoch)
                results[f"train_{key}"].append(train_metrics[key])
                results[f"val_{key}"].append(val_metrics[key])

            # --- Print metrics ---
            train_str = ", ".join([f"{k} {v:.4f}" for k, v in train_metrics.items()])
            val_str = ", ".join([f"{k} {v:.4f}" for k, v in val_metrics.items()])
            print(f"Epoch {epoch+1}: Train: {train_str}")
            print(f"        Val: {val_str}\n")

            # --- Early stopping & best model logging ---
            best_model_loss = update_best_model(model, val_metrics["loss"], best_model_loss)
            early_stopper(val_metrics["loss"])
            if early_stopper.early_stop:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

        # --- Log loss curve ---
        plot_path = log_loss_curve(results)
        log_loss_curve_mlflow(plot_path, artifact_path="plots")

    # --- Upload to S3 if AWS ---
    if cfg.outputs.mode == "aws":
        upload_folder_to_s3(
            mlflow_dir, cfg.dataset.aws.s3_bucket, f"{cfg.outputs.aws.s3_prefix}/mlruns"
        )

    return results
