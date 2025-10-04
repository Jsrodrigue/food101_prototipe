import torch
from .metrics import compute_metrics

def eval_one_epoch(model, dataloader, loss_fn, device, metrics_list=None):
    """
    Perform a full evaluation over a dataset for one epoch.
    
    This function can be used for validation or testing. It runs the model
    in evaluation mode over all batches of the provided dataloader, computes
    the loss and metrics, and returns a dictionary summarizing the results.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation or test dataset.
        loss_fn (callable): Loss function used to compute the loss.
        device (torch.device): Device on which to run the computations (CPU or GPU).
        metrics_list (list of str, optional): List of metric names to compute
            (e.g., ["accuracy", "f1"]). Defaults to ["accuracy"] if None.

    Returns:
        dict: Dictionary containing:
            - "loss": average loss over all batches
            - Other metrics as specified in metrics_list
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
