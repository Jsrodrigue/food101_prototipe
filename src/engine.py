"""
Engine module for PyTorch training and evaluation with MLflow tracking.
"""
import torch
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch
from .utils import save_model, set_seed
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from omegaconf import OmegaConf

# -----------------------------
# EarlyStopping class
# -----------------------------
class EarlyStopping:
    """
    Stops training if validation loss doesn't improve after a patience number of epochs.
    """
    def __init__(self, patience=5, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model=None, save_path=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            if save_path and model:
                save_model(model, save_path.parent, save_path.name)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if save_path and model:
                save_model(model, save_path.parent, save_path.name)

# -----------------------------
# Training step
# -----------------------------
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Trains model for one epoch and returns avg loss and accuracy."""
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_acc += (preds == y).sum().item() / len(y)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc


# -----------------------------
# Evaluation step
# -----------------------------
def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates model for one epoch and returns avg loss and accuracy."""
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_acc += (preds == y).sum().item() / len(y)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc


# -----------------------------
# MLflow training function
# -----------------------------

def train_mlflow(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device = torch.device("cpu"),
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    params: dict = None,
    config: dict = None,
    outputs_dir: str = "outputs",
    mlflow_dir: str = "mlruns",
    experiment_name: str = "foodmini_experiments",
    seed: int = 42,
    early_stop_patience: int = 5,
    save_name: str = "best_model.pth",
    run_name: str = None
) -> dict:

    # -----------------------------
    # Seed for reproducibility
    # -----------------------------
    set_seed(seed)

    # -----------------------------
    # Local run folder and plots folder
    # -----------------------------
    run_dir = Path(outputs_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Save config locally
    if config:
        with open(run_dir / "config.yaml", "w") as f:
            cfg_dict = OmegaConf.to_container(config, resolve=True)
            yaml.safe_dump(cfg_dict, f)

    model.to(device)
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # -----------------------------
    # Configure MLflow
    # -----------------------------
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment(experiment_name)

    early_stopper = EarlyStopping(patience=early_stop_patience, verbose=True)
    best_model_path = run_dir / save_name

    # Start MLflow run using the provided run_name
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)

        # Log local folder path
        mlflow.log_param("local_path", str(run_dir))

        # Training loop
        for epoch in tqdm(range(epochs), desc="Training"):
            train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            }, step=epoch)

            if scheduler:
                mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

            # Save metrics for local plotting
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, "
                  f"test_acc={test_acc:.4f}")

            # Early stopping
            early_stopper(test_loss, model, save_path=best_model_path)
            if early_stopper.early_stop:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

        # Save best model to MLflow
        if best_model_path.exists():
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="best_model",
                input_example=next(iter(train_dataloader))[0][:1].cpu().numpy(),
                pip_requirements=["torch>=2.8.0", "torchvision>=0.23.0"]
            )

    # Save metrics and plots locally
    pd.DataFrame(results).to_csv(run_dir / "metrics.csv", index=False)

    plt.figure(figsize=(10,5))
    plt.plot(results["train_loss"], label="train_loss")
    plt.plot(results["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plots_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(results["train_acc"], label="train_acc")
    plt.plot(results["test_acc"], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(plots_dir / "acc_curve.png")
    plt.close()

    return results
