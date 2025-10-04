# src/train_engine.py

import json
from pathlib import Path

import mlflow
import torch
from torch import optim
from tqdm import tqdm

from .utils.metrics import compute_metrics
from .utils.mlflow_utils import (
    log_hyperparams_mlflow,
    log_loss_curve_mlflow,
    update_best_model,
    setup_mlflow
)
from .utils.plot_utils import log_loss_curve
from .utils.s3_utils import upload_folder_to_s3
from .utils.seed_utils import set_seed
from .utils.eval_utils import eval_one_epoch


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


# -------------------- TRAIN STEP -------------------- #


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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=cfg.train.scheduler.patience
        )
    elif cfg.train.scheduler.type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.scheduler.step_size,
            gamma=cfg.train.scheduler.gamma,
        )

    # --- MLflow setup ---
    mlflow_dir, run_name = setup_mlflow(cfg)


    early_stopper = EarlyStopping(patience=cfg.train.early_stop_patience, verbose=True)
    best_model_loss = float("inf")
    best_model_epoch = -1

    # --- Metrics storage ---
    results = {f"train_{m}": [] for m in cfg.train.metrics + ["loss"]}
    results.update({f"val_{m}": [] for m in cfg.train.metrics + ["loss"]})

    with mlflow.start_run(run_name=run_name):
        log_hyperparams_mlflow(cfg, loss_fn)

        for epoch in tqdm(range(cfg.train.epochs), desc="Training"):
            # --- TRAIN & VALIDATION ---
            train_metrics = train_step(
                model, train_loader, loss_fn, optimizer, device, cfg.train.metrics
            )
            val_metrics = eval_one_epoch(
                model, val_loader, loss_fn, device, cfg.train.metrics
            )

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
            best_model_loss, is_new_best = update_best_model(
                model, val_metrics, best_model_loss, cfg, epoch
            )
            if is_new_best:
                best_model_epoch = epoch

            early_stopper(val_metrics["loss"])
            if early_stopper.early_stop:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

        # --- Add epochs info ---
        results["epochs"] = list(range(len(results["train_loss"])))
        results["best_epoch"] = best_model_epoch

        # --- Save results JSON ---
        json_path = Path("training_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        mlflow.log_artifact(str(json_path), artifact_path="metrics")
        json_path.unlink()

        # --- Log loss curve ---
        plot_path = log_loss_curve(results)
        log_loss_curve_mlflow(plot_path, artifact_path="plots")

    # --- Upload to S3 if AWS ---
    if cfg.outputs.mode == "aws":
        upload_folder_to_s3(
            mlflow_dir, cfg.dataset.aws.s3_bucket, f"{cfg.outputs.aws.s3_prefix}/mlruns"
        )

    return results
