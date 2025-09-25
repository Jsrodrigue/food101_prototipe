"""
Engine module for PyTorch training and evaluation with MLflow tracking.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
from torch import optim
from tqdm.auto import tqdm
import yaml
from omegaconf import OmegaConf

from .utils import save_model, set_seed
from .metrics import compute_metrics  # calcula métricas desde logits y labels


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after patience epochs."""
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

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
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
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
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


def train_mlflow(model, train_dataloader, test_dataloader, optimizer, loss_fn, cfg, device=None):
    """Train a model using cfg and dataloaders, log metrics to MLflow, save best model locally."""
    set_seed(cfg.train.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Scheduler
    scheduler = None
    if cfg.train.scheduler.type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    elif cfg.train.scheduler.type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.scheduler.step_size, gamma=cfg.train.scheduler.gamma)

    # Directories
    run_dir = Path(cfg.outputs.local.path)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)

    # MLflow
    mlflow_db_path = Path(cfg.outputs.mlflow.path) / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment(cfg.outputs.mlflow.experiment_name)

    early_stopper = EarlyStopping(patience=cfg.train.early_stop_patience, verbose=True)
    best_model_path = run_dir / f"{cfg.model.name}_best.pth"

    results = {f"train_{m}": [] for m in cfg.train.metrics + ["loss"]}
    results.update({f"test_{m}": [] for m in cfg.train.metrics + ["loss"]})

    with mlflow.start_run(run_name=cfg.outputs.run_name):
        mlflow.log_params(OmegaConf.to_container(cfg.train, resolve=True))
        for epoch in tqdm(range(cfg.train.epochs), desc="Training"):
            train_metrics = train_step(model, train_dataloader, loss_fn, optimizer, device, cfg.train.metrics)
            test_metrics = test_step(model, test_dataloader, loss_fn, device, cfg.train.metrics)

            # Scheduler step
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_metrics["loss"])
                else:
                    scheduler.step()

            # Log metrics
            for key in train_metrics:
                mlflow.log_metric(f"train_{key}", train_metrics[key], step=epoch)
                mlflow.log_metric(f"test_{key}", test_metrics[key], step=epoch)
                results[f"train_{key}"].append(train_metrics[key])
                results[f"test_{key}"].append(test_metrics[key])

            # Print 
            train_str = ", ".join([f"{k} {v:.4f}" for k, v in train_metrics.items()])
            test_str = ", ".join([f"{k} {v:.4f}" for k, v in test_metrics.items()])

            print(f"Epoch {epoch+1}: Train: {train_str}")
            print(f"                 Test: {test_str}\n")


            # Early stopping
            early_stopper(test_metrics["loss"], model, save_path=best_model_path)
            if early_stopper.early_stop:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

        # Save best model to MLflow
        if best_model_path.exists():
            mlflow.pytorch.log_model(model, "best_model",
                                     input_example=next(iter(train_dataloader))[0][:1].cpu().numpy())

    # Plots
    plt.figure(figsize=(10,5))
    plt.plot(results["train_loss"], label="train_loss")
    plt.plot(results["test_loss"], label="test_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(plots_dir / "loss_curve.png"); plt.close()

    return results
