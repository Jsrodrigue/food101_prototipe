import sys, datetime
from pathlib import Path
from omegaconf import DictConfig
import hydra
import torch
from torchvision.models import MobileNet_V2_Weights , EfficientNet_B0_Weights
from torch import nn, optim
from src.data_setup import create_dataloaders
from src.engine import train_mlflow
from src.utils import set_seed
from models import MobileNetV2Model, EfficientNetModel
from src.data_setup import get_transforms


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    local_dir = Path(get_original_cwd()) / cfg.outputs.local.path / f"run_{timestamp}"; local_dir.mkdir(parents=True, exist_ok=True)
    mlflow_dir = Path(get_original_cwd()) / cfg.outputs.mlflow.path; mlflow_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"run_{timestamp}"

    cfg.outputs.run_name = run_name
    cfg.outputs.local.path = str(local_dir)
    cfg.outputs.mlflow.path = str(mlflow_dir)
    cfg.outputs.run_name = run_name

    train_dir = Path(get_original_cwd()) / cfg.dataset.test_dir
    val_dir = Path(get_original_cwd()) / cfg.dataset.val_dir

    
    
    if cfg.model.name.lower() == "mobilenet":
        base_transform = MobileNet_V2_Weights.DEFAULT.transforms()
    elif cfg.model.name.lower() == "efficientnet":
        base_transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")
    train_transform, test_transform = get_transforms(augmentation=cfg.train.augmentation, base_transform=base_transform)
    
    train_loader, val_loader, class_names = create_dataloaders(
        train_dir=str(train_dir),
        test_dir=str(val_dir),
        batch_size=cfg.train.batch_size,
        train_subset_percentage=cfg.train.subset_percentage,
        train_transform=train_transform,
        test_transform=test_transform,
        seed=cfg.train.seed
    )
    set_seed(cfg.train.seed)

    num_classes = len(class_names)
    
    if cfg.model.name.lower() == "mobilenet":
        model = MobileNetV2Model(num_classes=num_classes, pretrained=cfg.model.pretrained)
    elif cfg.model.name.lower() == "efficientnet":
        model = EfficientNetModel(version=cfg.model.version, num_classes=num_classes, pretrained=cfg.model.pretrained)
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")
    

    optimizer = optim.Adam(model.model.parameters(), lr=cfg.train.optimizer.lr)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = train_mlflow(model.model, train_loader, val_loader, optimizer, loss_fn, cfg, device)
    print("Training completed. Best model saved locally and logged in MLflow.")

if __name__=="__main__":
    main()
