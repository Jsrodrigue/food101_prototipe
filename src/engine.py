"""
Engine module for PyTorch training and evaluation with MLflow tracking.
"""
import torch
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch

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
    scheduler=None,
    params: Dict = None,
    dagshub_uri: str = None,  # if None, use local ./mlruns
    experiment_name: str = "foodmini_experiments",
    seed: int = 42
) -> Dict[str, List]:
    """
    Trains a PyTorch model while logging metrics and model to MLflow.

    Returns a dict with train/test loss and accuracy per epoch.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    model.to(device)
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Configure MLflow
    mlflow.set_tracking_uri(dagshub_uri if dagshub_uri else "file:./mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        if params:
            mlflow.log_params(params)

        for epoch in tqdm(range(epochs), desc="Training"):
            train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

            # Store results
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            }, step=epoch)

            if scheduler:
                mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        # Log final model
        example_input, _ = next(iter(train_dataloader))
        example_input = example_input[:1].cpu().numpy()
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            input_example=example_input,
            pip_requirements=["torch>=2.8.0", "torchvision>=0.23.0"]
        )

    return results
