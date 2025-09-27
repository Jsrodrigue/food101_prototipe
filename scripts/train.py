import datetime
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch import nn, optim
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights

from models import EfficientNetModel, MobileNetV2Model
from src.data_setup import create_dataloaders, get_data_path, get_transforms
from src.engine import train_mlflow
from src.utils import set_seed


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if cfg.dataset.mode == "aws":
        #  Local direcotry for outputs and Mlflow
        local_dir = Path(get_original_cwd()) / "tmp_outputs" / f"run_{timestamp}"
        local_dir.mkdir(parents=True, exist_ok=True)
        mlflow_dir = Path(get_original_cwd()) / "tmp_mlflow"
        mlflow_dir.mkdir(exist_ok=True)
    else:
        local_dir = (
            Path(get_original_cwd()) / cfg.outputs.local.path / f"run_{timestamp}"
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        mlflow_dir = Path(get_original_cwd()) / cfg.outputs.local.mlflow.path
        mlflow_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"run_{timestamp}"

    cfg.outputs.run_name = run_name
    cfg.outputs.local.path = str(local_dir)
    cfg.outputs.local.mlflow.path = str(mlflow_dir)

    data_path = get_data_path(cfg)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if cfg.model.name.lower() == "mobilenet":
        base_transform = MobileNet_V2_Weights.DEFAULT.transforms()
    elif cfg.model.name.lower() == "efficientnet":
        base_transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")
    train_transform, val_transform = get_transforms(
        augmentation=cfg.train.augmentation, base_transform=base_transform
    )

    train_loader, val_loader, class_names = create_dataloaders(
        train_dir=str(train_dir),
        test_dir=str(val_dir),
        batch_size=cfg.train.batch_size,
        train_subset_percentage=cfg.train.subset_percentage,
        train_transform=train_transform,
        test_transform=val_transform,
        seed=cfg.train.seed,
    )
    set_seed(cfg.train.seed)

    num_classes = len(class_names)

    if cfg.model.name.lower() == "mobilenet":
        model = MobileNetV2Model(
            num_classes=num_classes, pretrained=cfg.model.pretrained
        )
    elif cfg.model.name.lower() == "efficientnet":
        model = EfficientNetModel(
            version=cfg.model.version,
            num_classes=num_classes,
            pretrained=cfg.model.pretrained,
        )
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")

    # Freeze backbone
    model.freeze_backbone()

    # Unfreeze last layers
    if cfg.train.unfreeze_layers > 0:
        model.unfreeze_backbone(cfg.train.unfreeze_layers)

    optimizer = optim.Adam(model.model.parameters(), lr=cfg.train.optimizer.lr)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = train_mlflow(
        model.model, train_loader, val_loader, optimizer, loss_fn, cfg, device
    )
    print("Training completed. Best model saved locally and logged in MLflow.")


if __name__ == "__main__":
    main()
