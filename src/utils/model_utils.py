import torch
from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B2_Weights, MobileNet_V2_Weights

from src.models import EfficientNetModel, MobileNetV2Model


def load_model_from_run(
    state_dict_path,
    model_name: str,
    num_classes: int,
    version: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.nn.Module:
    """
    Load a trained model from a run folder.

    Args:
        state_dict_path (str | Path): Path to the saved model_state_dict.pth file.
        model_name (str): Model family name, e.g., "efficientnet" or "mobilenet".
        num_classes (int): Number of output classes for the classification head.
        version (str, optional): Specific version for EfficientNet (e.g., "b0", "b2").
        device (str, optional): Device to map the model to ("cpu" or "cuda").

    Returns:
        model (torch.nn.Module): Loaded model in evaluation mode on the given device.
    """
    if model_name.startswith("efficientnet"):
        if version is None:
            raise ValueError("EfficientNet requires a version (e.g., 'b0', 'b2').")
        model = EfficientNetModel(version=version, num_classes=num_classes)

    elif model_name.startswith("mobilenet"):
        model = MobileNetV2Model(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model family: {model_name}")

    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def get_model_transforms(model_name: str, version: str = None, augmentation: str = None):
    """
    Return the default torchvision transforms associated with a model,
    optionally adding augmentation.

    Args:
        model_name (str): Model family name, e.g., "efficientnet" or "mobilenet".
        version (str, optional): Version for EfficientNet (e.g., "b0", "b2").
        augmentation (str, optional): Name of augmentation to apply (e.g., "TrivialAugmentWide").

    Returns:
        transform (Callable): Transform pipeline.
    """
    # Get base transforms from torchvision weights
    if model_name.startswith("efficientnet"):
        if version is None:
            raise ValueError("EfficientNet requires a version (e.g., 'b0', 'b2').")
        version = version.lower()
        if version == "b0":
            transform = EfficientNet_B0_Weights.DEFAULT.transforms()
        elif version == "b2":
            transform = EfficientNet_B2_Weights.DEFAULT.transforms()
        else:
            raise ValueError(f"Unsupported EfficientNet version: {version}")
    elif model_name.startswith("mobilenet"):
        transform = MobileNet_V2_Weights.DEFAULT.transforms()
    else:
        raise ValueError(f"Unknown model family: {model_name}")

    if augmentation:
        if augmentation.lower() == "trivialaugmentwide":
            transform = transforms.Compose([TrivialAugmentWide(), transform])
        else:
            raise ValueError(f"Unknown augmentation {augmentation}")

    return transform