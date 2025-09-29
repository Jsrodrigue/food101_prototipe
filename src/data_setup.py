import os
import random
from pathlib import Path

from hydra.utils import get_original_cwd
from torch import cuda
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import TrivialAugmentWide

from .utils import set_seed

NUM_WORKERS = os.cpu_count()


def get_transforms(augmentation=None, base_transform=None):
    if base_transform is None:
        base_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = base_transform
    if augmentation:
        if augmentation.lower() == "trivialaugmentwide":
            train_transform = transforms.Compose([TrivialAugmentWide(), base_transform])
        else:
            raise ValueError(f"Unknown augmentation {augmentation}")
    test_transform = base_transform
    return train_transform, test_transform


def create_dataloader_from_folder(
    data_dir,
    batch_size,
    transform,
    subset_percentage=1.0,
    shuffle=True,
    seed=42,
    num_workers=NUM_WORKERS,
):
    """
    Create a DataLoader from an image folder.

    Args:
        data_dir (str): Path to the folder containing images organized by class.
        batch_size (int): Batch size for the DataLoader.
        transform (torchvision.transforms): Transformations to apply to images.
        subset_percentage (float, optional): Fraction of the dataset to use (0 < subset_percentage <= 1.0).
        shuffle (bool, optional): Whether to shuffle the data.
        seed (int, optional): Random seed for reproducibility.
        num_workers (int, optional): Number of worker processes for loading data.

    Returns:
        loader (DataLoader): PyTorch DataLoader for the dataset.
        class_names (list): List of class names in the dataset.
    """
    # Set random seeds for reproducibility using utils.set_seed
    set_seed(seed)

    # Create the dataset from the folder
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Optionally take a random subset of the dataset
    if 0 < subset_percentage < 1.0:
        indices = random.sample(
            range(len(dataset)), int(len(dataset) * subset_percentage)
        )
        dataset = Subset(dataset, indices)

    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if cuda.is_available() else False,
    )

    # Get class names
    class_names = (
        dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes
    )

    return loader, class_names


def create_dataloaders(
    train_dir,
    test_dir,
    batch_size,
    train_transform,
    test_transform,
    train_subset_percentage=1.0,
    seed=42,
    num_workers=NUM_WORKERS,
):
    """
    Create DataLoaders for training and testing/validation.

    Args:
        train_dir (str): Path to the training data folder.
        test_dir (str): Path to the test/validation data folder.
        batch_size (int): Batch size for both loaders.
        train_transform (torchvision.transforms): Transformations for training images.
        test_transform (torchvision.transforms): Transformations for test/validation images.
        train_subset_percentage (float, optional): Fraction of training data to use.
        seed (int, optional): Random seed for reproducibility.
        num_workers (int, optional): Number of worker processes.

    Returns:
        train_loader (DataLoader): DataLoader for training.
        test_loader (DataLoader): DataLoader for testing/validation.
        class_names (list): List of class names.
    """
    # Create the training DataLoader
    train_loader, class_names = create_dataloader_from_folder(
        train_dir,
        batch_size,
        transform=train_transform,
        subset_percentage=train_subset_percentage,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )

    # Create the test/validation DataLoader
    test_loader, _ = create_dataloader_from_folder(
        test_dir,
        batch_size,
        transform=test_transform,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )

    return train_loader, test_loader, class_names


def get_data_path(cfg):
    """
    Returns the dataset path depending on the execution mode (local or AWS).
    """
    if cfg.dataset.mode == "aws":
        # Instead of syncing every time, use tmp_data folder downloaded by train.py
        return Path(get_original_cwd()) / "tmp_data"
    else:
        return Path(get_original_cwd()) / cfg.dataset.path

        # Function to recursively download S3 folder to local

def download_s3_folder(bucket, prefix, local_path: Path, s3):
    """
    Recursively download all files from an S3 folder (prefix) to a local path.
    Skips files that already exist locally.

    Args:
        bucket (str): Name of the S3 bucket.
        prefix (str): The S3 folder prefix to download.
        local_path (Path): Local folder where files will be stored.
        s3 (boto3.client): Boto3 S3 client instance.
    """
    # Ensure the base local directory exists
    local_path.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    downloaded_count = 0
    skipped_count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue

        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Skip pseudo-directories
            if key.endswith("/"):
                continue

            # Create relative path inside local_path
            relative_path = Path("/".join(key.split("/")[1:]))  
            file_path = local_path / relative_path

            # Skip if file already exists
            if file_path.exists():
                skipped_count += 1
                continue

            # Ensure parent directories exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the file
            s3.download_file(bucket, key, str(file_path))
            downloaded_count += 1

    print(f"[INFO] S3 download completed. Downloaded {downloaded_count} files, skipped {skipped_count} existing files.")
