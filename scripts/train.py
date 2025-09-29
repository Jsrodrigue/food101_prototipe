import datetime
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights
import boto3

from models import EfficientNetModel, MobileNetV2Model
from src.data_setup import create_dataloaders, get_transforms, download_s3_folder, get_data_path
from src.engine import train_mlflow
from src.utils import set_seed

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}"
    cfg.outputs.run_name = run_name

    # ---------------- DATA PATHS ----------------
    if cfg.dataset.mode == "aws":
        # Temporary folder for S3 data
        tmp_data_dir = Path.cwd() / "tmp_data"
        tmp_data_dir.mkdir(parents=True, exist_ok=True)

        # Download training and validation folders from S3
        s3 = boto3.client("s3")
        download_s3_folder(cfg.dataset.aws.s3_bucket, cfg.dataset.aws.train_prefix, tmp_data_dir, s3)
        download_s3_folder(cfg.dataset.aws.s3_bucket, cfg.dataset.aws.val_prefix, tmp_data_dir, s3)

        data_path = tmp_data_dir / "dataset"
        mlflow_path = Path.cwd() / "mlruns_aws"
        mlflow_path.mkdir(parents=True, exist_ok=True)
    else:
        data_path = Path(cfg.dataset.path)
        mlflow_path = Path(cfg.outputs.local.mlflow.path)
        mlflow_path.mkdir(parents=True, exist_ok=True)

    train_dir = data_path / "train"
    val_dir = data_path / "val"

    # ---------------- TRANSFORMS ----------------
    if cfg.model.name.lower() == "mobilenet":
        base_transform = MobileNet_V2_Weights.DEFAULT.transforms()
    elif cfg.model.name.lower() == "efficientnet":
        base_transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")

    train_transform, val_transform = get_transforms(
        augmentation=cfg.train.augmentation,
        base_transform=base_transform
    )

    # ---------------- DATALOADERS ----------------
    train_loader, val_loader, class_names = create_dataloaders(
        train_dir=str(train_dir),
        test_dir=str(val_dir),
        batch_size=cfg.train.batch_size,
        train_transform=train_transform,
        test_transform=val_transform,
        train_subset_percentage=cfg.train.subset_percentage,
        seed=cfg.train.seed,
    )

    set_seed(cfg.train.seed)
    num_classes = len(class_names)

    # ---------------- MODEL ----------------
    if cfg.model.name.lower() == "mobilenet":
        model = MobileNetV2Model(num_classes=num_classes, pretrained=cfg.model.pretrained)
    else:
        model = EfficientNetModel(
            version=cfg.model.version,
            num_classes=num_classes,
            pretrained=cfg.model.pretrained,
        )

    model.freeze_backbone()
    if cfg.train.unfreeze_layers > 0:
        model.unfreeze_backbone(cfg.train.unfreeze_layers)

    # ---------------- OPTIMIZER & LOSS ----------------
    optimizer = optim.Adam(model.model.parameters(), lr=cfg.train.optimizer.lr)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- TRAIN ----------------
    print("[INFO] Starting training...")
    results = train_mlflow(model.model, train_loader, val_loader, optimizer, loss_fn, cfg, device)

    print("[INFO] Training completed. Best model saved and logged in MLflow.")
    print("[INFO] Sample of loss curve values:")
    print("Train loss:", results["train_loss"][:5])
    print("Val loss:", results["val_loss"][:5])

if __name__ == "__main__":
    main()
