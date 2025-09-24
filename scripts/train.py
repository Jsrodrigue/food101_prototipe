# train.py
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig

import torch
from torch import nn, optim

from models import EfficientNetModel, MobileNetV2Model
from src.data_setup import create_dataloaders, download_and_extract
from src.engine import train_mlflow

# -----------------------------
# Hydra entry point
# -----------------------------
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main training script using Hydra configuration.
    Downloads dataset if needed, creates dataloaders, selects model,
    optimizer, and calls the training engine with MLflow logging.
    """

    # -----------------------------
    # 0️⃣ Load environment variables
    # -----------------------------
    load_dotenv()  # loads .env if exists
    root_path = os.getenv("ROOT_PATH")
    if not root_path:
        raise ValueError("ROOT_PATH not defined in environment variables or .env file")

    # Build dataset path dynamically
    dataset_path = Path(root_path) / cfg.dataset.path  # relative path from YAML
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"

    # -----------------------------
    # 1️⃣ Prepare dataset
    # -----------------------------
    if cfg.dataset.url:
        download_and_extract(cfg.dataset.url, str(dataset_path), overwrite=cfg.dataset.overwrite)

    # -----------------------------
    # 2️⃣ Define transforms
    # -----------------------------
    from torchvision import transforms
    from torchvision.models import MobileNet_V2_Weights, EfficientNet_B0_Weights
    from torchvision.transforms import TrivialAugmentWide

    # Select pretrained weights depending on model
    if cfg.model.name.lower() == "mobilenet":
        weights = MobileNet_V2_Weights.DEFAULT
    elif cfg.model.name.lower() == "efficientnet":
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")

    # Base transform recommended by pretrained weights (resize + normalization)
    base_transform = weights.transforms()

    # Train transform: optionally add TrivialAugmentWide if augment=True
    if getattr(cfg.train, "augment", False):
        train_transform = transforms.Compose([
            TrivialAugmentWide(),       # automatic random augmentations
            *base_transform.transforms  # keep pretrained resize + normalization
        ])
    else:
        train_transform = base_transform

    # Test transform: always base transforms only
    test_transform = base_transform

    # -----------------------------
    # 3️⃣ Create dataloaders
    # -----------------------------
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

    # -----------------------------
    # 4️⃣ Select model
    # -----------------------------
    if cfg.model.name.lower() == "mobilenet":
        model = MobileNetV2Model(num_classes=num_classes, pretrained=cfg.model.pretrained)
    elif cfg.model.name.lower() == "efficientnet":
        model = EfficientNetModel(version=cfg.model.version, num_classes=num_classes, pretrained=cfg.model.pretrained)
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")

    # -----------------------------
    # 5️⃣ Optimizer, loss, scheduler
    # -----------------------------
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = None
    if cfg.train.scheduler.type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    elif cfg.train.scheduler.type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # -----------------------------
    # 6️⃣ Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 7️⃣ Train
    # -----------------------------
    results = train_mlflow(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.train.epochs,
        device=device,
        scheduler=scheduler,
        params=cfg,            # log config parameters in MLflow
        config=cfg,            # save Hydra config in outputs
        outputs_dir=cfg.outputs.local.path,
        mlflow_dir=cfg.outputs.mlflow.path,
        experiment_name=cfg.outputs.mlflow.experiment_name,
        seed=cfg.train.seed,
        early_stop_patience=cfg.train.early_stop_patience,
        save_name = f"{cfg.model.name}_{cfg.model.version}_bs{cfg.train.batch_size}_lr{cfg.train.lr}.pth"
    )

    print("Training completed. Metrics saved in outputs folder and MLflow logged.")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
