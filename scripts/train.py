import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent))

import hydra
from omegaconf import DictConfig

import torch
from torch import nn, optim

from models import EfficientNetModel, MobileNetV2Model
from src.data_setup import create_dataloaders, download_and_extract
from src.engine import train_mlflow
from src.utils import set_seed

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):

    from hydra.utils import get_original_cwd
    import datetime

    # Generate timestamp to uniquely identify the run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Local output folder for this run
    local_dir = Path(get_original_cwd()) / cfg.outputs.local.path / f"run_{timestamp}"
    local_dir.mkdir(parents=True, exist_ok=True)

    # MLflow folder for this run
    mlflow_dir = Path(get_original_cwd()) / cfg.outputs.mlflow.path
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    # Define run name (same as local folder) to use in MLflow
    run_name = f"run_{timestamp}"

    # Update config paths to reflect current run directories
    cfg.outputs.local.path = str(local_dir)
    cfg.outputs.mlflow.path = str(mlflow_dir)

    # Dataset paths
    dataset_path = Path(get_original_cwd()) / cfg.dataset.path
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"

    # Download dataset if URL is provided
    if cfg.dataset.url:
        download_and_extract(cfg.dataset.url, str(dataset_path), overwrite=cfg.dataset.overwrite)

    from torchvision import transforms
    from torchvision.models import MobileNet_V2_Weights, EfficientNet_B0_Weights
    from torchvision.transforms import TrivialAugmentWide

    # Select pretrained weights according to model
    if cfg.model.name.lower() == "mobilenet":
        weights = MobileNet_V2_Weights.DEFAULT
    elif cfg.model.name.lower() == "efficientnet":
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")

    # Define transforms for training and testing
    base_transform = weights.transforms()
    train_transform = transforms.Compose([TrivialAugmentWide(), *base_transform.transforms]) if getattr(cfg.train, "augment", False) else base_transform
    test_transform = base_transform

    # Create dataloaders
    train_loader, test_loader, class_names = create_dataloaders(
        train_dir=str(train_dir),
        test_dir=str(test_dir),
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=cfg.train.batch_size,
        train_subset_percentage=cfg.train.subset_percentage,
        seed=cfg.train.seed
    )
    num_classes = len(class_names)

    #set seed 
    set_seed(cfg.train.seed)

    # Initialize model
    if cfg.model.name.lower() == "mobilenet":
        model = MobileNetV2Model(num_classes=num_classes, pretrained=cfg.model.pretrained)
    elif cfg.model.name.lower() == "efficientnet":
        model = EfficientNetModel(version=cfg.model.version, num_classes=num_classes, pretrained=cfg.model.pretrained)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = None
    if cfg.train.scheduler.type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    elif cfg.train.scheduler.type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start training and MLflow logging
    results = train_mlflow(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.train.epochs,
        device=device,
        scheduler=scheduler,
        params=cfg,
        config=cfg,
        outputs_dir=cfg.outputs.local.path,
        mlflow_dir=cfg.outputs.mlflow.path,
        experiment_name=cfg.outputs.mlflow.experiment_name,
        seed=cfg.train.seed,
        early_stop_patience=cfg.train.early_stop_patience,
        save_name=f"{cfg.model.name}_{cfg.model.version}_bs{cfg.train.batch_size}_lr{cfg.train.lr}.pth",
        run_name=run_name  # <-- Pass run name to MLflow
    )

    print("Training completed. Local outputs and MLflow logged separately.")

if __name__ == "__main__":
    main()
