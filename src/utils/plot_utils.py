import matplotlib.pyplot as plt
from pathlib import Path

def log_loss_curve(results, filename="loss_curve.png"):
    """
    Plot and save the training and validation loss curve.

    Args:
        results (dict): Dictionary containing training and validation metrics.
                        Expected keys: "train_loss", "val_loss".
        filename (str): Name of the output PNG file where the plot will be saved.

    Returns:
        Path: Path object pointing to the saved PNG file
    """
    plt.figure(figsize=(10, 5))
    plt.plot(results["train_loss"], label="train_loss")
    plt.plot(results["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return Path(filename)
