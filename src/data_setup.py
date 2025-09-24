import os
import random
import zipfile
from pathlib import Path
from io import BytesIO
import requests

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count() 

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    train_transform: transforms.Compose,
    test_transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int = NUM_WORKERS,
    subset_percentage: float = 1.0,
    seed: int = 42
):
    """Creates train/test DataLoaders with optional subset sampling and reproducibility."""
    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    if subset_percentage < 1.0:
        random.seed(seed)
        train_size = int(len(train_data) * subset_percentage)
        test_size = int(len(test_data) * subset_percentage)
        train_indices = random.sample(range(len(train_data)), train_size)
        test_indices = random.sample(range(len(test_data)), test_size)
        train_data = Subset(train_data, train_indices)
        test_data = Subset(test_data, test_indices)

    class_names = train_data.dataset.classes if isinstance(train_data, Subset) else train_data.classes

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    return train_dataloader, test_dataloader, class_names


def download_and_extract(url: str, destination: str, overwrite: bool = False, timeout: int = 10):
    """Download a zip file from a URL and extract it to a destination folder."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    
    # Skip download if folder exists
    if any(destination.iterdir()) and not overwrite:
        print(f"Destination {destination} already contains files. Skipping download.")
        return

    print(f"Downloading from {url} ...")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    print("Download completed, extracting...")
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        zf.extractall(destination)
    print(f"Files extracted to {destination}")
