import datetime
import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights

from src.models import EfficientNetModel, MobileNetV2Model
from src.data_setup import create_dataloader_from_folder, prepare_data
from src.train_engine import train_mlflow
from src.utils.seed_utils import set_seed
from src.utils.model_utils import get_model_transforms

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}"
    cfg.outputs.run_name = run_name

    # ---------------- DATA PATHS ----------------
    data_paths = prepare_data(cfg, get_train=True, get_val=True, get_test=False)
    train_dir = data_paths['train_dir']
    val_dir = data_paths['val_dir']


    # ---------------- TRANSFORMS ----------------

    train_transform= get_model_transforms(
        model_name= cfg.model.name.lower(), 
        version= cfg.model.version, 
        augmentation=cfg.train.augmentation 
    )

    val_transform= get_model_transforms(
        model_name= cfg.model.name.lower(), 
        version= cfg.model.version, 
        augmentation=None 
    )

    # ---------------- DATALOADERS ----------------
   
    train_loader, class_names = create_dataloader_from_folder(
        data_dir=train_dir,
        batch_size=cfg.train.batch_size,
        transform=train_transform,
        subset_percentage=cfg.train.subset_percentage,
        seed=cfg.train.seed)

    val_loader, _ = create_dataloader_from_folder(
        data_dir=val_dir,
        batch_size=cfg.train.batch_size,
        transform=val_transform,
        subset_percentage=1,
        seed=cfg.train.seed)
    
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
    if cfg.train.loss_fn.lower()=="crossentropyloss":
        loss_fn = nn.CrossEntropyLoss()
    else: 
        raise ValueError(f"Unknown loss function: {cfg.train.loss_fn}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- TRAIN ----------------
    print("[INFO] Starting training...")
    train_mlflow(model.model, train_loader, val_loader, optimizer, loss_fn, cfg, device)

   
if __name__ == "__main__":
    main()
