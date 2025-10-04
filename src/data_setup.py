import os
import random
from pathlib import Path

import boto3
from torch import cuda
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .utils.s3_utils import download_s3_folder
from .utils.seed_utils import set_seed

NUM_WORKERS = os.cpu_count()



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



def prepare_data(cfg, get_train=True, get_val=True, get_test=True):
    """
    Prepares dataset paths according to the configuration.
    Supports AWS S3 or local datasets, with options to fetch only the desired sets.
    
    Args:
        cfg: Full configuration (Hydra / OmegaConf)
        get_train (bool): Whether to prepare/download the training set
        get_val (bool): Whether to prepare/download the validation set
        get_test (bool): Whether to prepare/download the test set
    
    Returns:
        dict: {'train_dir': Path or None, 'val_dir': Path or None, 'test_dir': Path or None}
              Paths to the prepared datasets. None if not requested.
    """

    paths = {'train_dir': None, 'val_dir': None, 'test_dir': None}
    
    if cfg.dataset.mode == "aws":
        base_dir = Path.cwd() / "data_s3"
        base_dir.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")

        if get_train:
            download_s3_folder(cfg.dataset.aws.s3_bucket, cfg.dataset.aws.train_prefix, base_dir, s3)
            paths['train_dir'] = base_dir/ 'dataset' / 'train'
        if get_val:
            download_s3_folder(cfg.dataset.aws.s3_bucket, cfg.dataset.aws.val_prefix, base_dir, s3)
            paths['val_dir'] = base_dir / 'dataset' / 'val'
        if get_test:
            download_s3_folder(cfg.dataset.aws.s3_bucket, cfg.dataset.aws.test_prefix, base_dir, s3)
            paths['test_dir'] = base_dir / 'dataset' / 'test'
        print(paths)
    else:
        if get_train:
            paths['train_dir'] = Path(cfg.dataset.train_dir)
        if get_val:
            paths['val_dir'] = Path(cfg.dataset.val_dir)
        if get_test:
            paths['test_dir'] = Path(cfg.dataset.test_dir)
    
    return paths
