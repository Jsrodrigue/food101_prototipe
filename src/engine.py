from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
from torch import optim
from tqdm.auto import tqdm

from .metrics import compute_metrics
from .utils import (
    log_hyperparams_mlflow,
    log_pytorch_model_mlflow,
    save_model,
    set_seed,
    upload_folder_to_s3,
)


class EarlyStopping:
    """Stop training if validation loss doesn't improve after patience epochs."""

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
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    # for X, y in dataloader:
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.long()  # <-- asegurar tipo correcto
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
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.inference_mode():
        # for X, y in dataloader:
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y = y.long()  # <-- asegurar tipo correcto
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
    set_seed(cfg.train.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Scheduler
    scheduler = None
    if cfg.train.scheduler.type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )
    elif cfg.train.scheduler.type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.scheduler.step_size,
            gamma=cfg.train.scheduler.gamma,
        )

    # MLflow setup
    if cfg.outputs.mode == "aws":
        mlflow_dir = Path(cfg.outputs.aws.mlflow.path)
        exp_name = cfg.outputs.aws.mlflow.experiment_name
    else:
        mlflow_dir = Path(cfg.outputs.local.mlflow.path)
        exp_name = cfg.outputs.local.mlflow.experiment_name
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:///{mlflow_dir}")
    mlflow.set_experiment(exp_name)

    # Artifacts
    artifacts_dir = mlflow_dir /cfg.outputs.run_name / "artifacts"
    plots_dir = mlflow_dir /cfg.outputs.run_name / "plots"
    checkpoints_dir = artifacts_dir / "checkpoints"
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = checkpoints_dir / f"{cfg.model.name}_best.pth"
    early_stopper = EarlyStopping(patience=cfg.train.early_stop_patience, verbose=True)

    results = {f"train_{m}": [] for m in cfg.train.metrics + ["loss"]}
    results.update({f"test_{m}": [] for m in cfg.train.metrics + ["loss"]})

    with mlflow.start_run(run_name=cfg.outputs.run_name):
        log_hyperparams_mlflow(cfg, loss_fn)
        for epoch in tqdm(range(cfg.train.epochs), desc="Training"):
            train_metrics = train_step(
                model, train_loader, loss_fn, optimizer, device, cfg.train.metrics
            )
            val_metrics = test_step(
                model, val_loader, loss_fn, device, cfg.train.metrics
            )

            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            for key in train_metrics:
                mlflow.log_metric(f"train_{key}", train_metrics[key], step=epoch)
                mlflow.log_metric(f"val_{key}", val_metrics[key], step=epoch)
                results[f"train_{key}"].append(train_metrics[key])
                results[f"test_{key}"].append(val_metrics[key])

            train_str = ", ".join([f"{k} {v:.4f}" for k, v in train_metrics.items()])
            val_str = ", ".join([f"{k} {v:.4f}" for k, v in val_metrics.items()])
            print(f"Epoch {epoch+1}: Train: {train_str}")
            print(f"        Val: {val_str}\n")

            early_stopper(val_metrics["loss"], model, save_path=best_model_path)
            if early_stopper.early_stop:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

        if best_model_path.exists():
            log_pytorch_model_mlflow(
                model, model_name="best_model", artifact_dir=artifacts_dir
            )

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(results["train_loss"], label="train_loss")
    plt.plot(results["test_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plots_dir / "loss_curve.png")
    plt.close()

    # Upload to S3 if AWS
    if cfg.outputs.mode == "aws":
        upload_folder_to_s3(
            mlflow_dir, cfg.dataset.aws.s3_bucket, f"{cfg.outputs.aws.s3_prefix}/mlruns"
        )

    return results
