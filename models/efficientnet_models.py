import torch
import torch.nn as nn
from torchvision import models

class EfficientNetModel(nn.Module):
     """
    Wrapper for EfficientNet (B0 and B2) with support for transfer learning.

    Args:
        version (str): 'b0' or 'b2' to select EfficientNet version.
        num_classes (int): Number of output classes for classification.
        pretrained (bool): Whether to use ImageNet-pretrained weights.

    Methods:
        forward(x): Returns model predictions for input x.
        freeze_backbone(): Freeze all layers except the classifier.
        unfreeze_backbone(layers=None): Unfreeze all or selected layers.

    Usage example:
        model = EfficientNetModel(version='b0', num_classes=10)
        model.freeze_backbone()
        model.unfreeze_backbone(layers=3)
    """
     def __init__(self, version='b0', num_classes=10, pretrained=True):
        super().__init__()
        self.version = version.lower()
        
        if self.version == 'b0':
            self.model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
        elif self.version == 'b2':
            self.model = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported version: {version}. Use 'b0' or 'b2'.")
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

     def forward(self, x):
        return self.model(x)
    
     def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

     def unfreeze_backbone(self, layers=None):
        """
        Unfreeze backbone layers partially or fully.
        Args:
            layers: None = unfreeze all
                    int = last N layers
                    list = list of block indices to unfreeze
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = False  # Freeze all first
        
        if layers is None:
            # Unfreeze all
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            blocks = list(self.model.features.children())
            if isinstance(layers, int):
                blocks_to_unfreeze = blocks[-layers:]
            elif isinstance(layers, list):
                blocks_to_unfreeze = [blocks[i] for i in layers]
            else:
                raise ValueError("layers should be None, int, or list of indices.")
            
            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
