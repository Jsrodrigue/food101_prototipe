import os
import random

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import TrivialAugmentWide

from .utils import (
    set_seed,  # Make sure this function sets seeds for random, numpy, torch, etc.
)

NUM_WORKERS = os.cpu_count()


def get_transforms(augmentation=None, base_transform=None):
    if base_transform is None:
        base_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = base_transform
    if augmentation:
        if augmentation.lower() == "trivialaugmentwide":
            train_transform = transforms.Compose(
                [TrivialAugmentWide(), base_transform] 
            )
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
        pin_memory=True,
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
