import torch
import torch.nn as nn
from torchvision import models

class MobileNetV2Model(nn.Module):
    """
Wrapper for MobileNetV2 with support for transfer learning.

Args:
    num_classes (int): Number of output classes for classification.
    pretrained (bool): Whether to use ImageNet-pretrained weights.

Methods:
    forward(x): Returns model predictions for input x.
    freeze_backbone(): Freeze all layers except the classifier.
    unfreeze_backbone(layers=None): Unfreeze all or selected layers.

Usage example:
    model = MobileNetV2Model(num_classes=10)
    model.freeze_backbone()
    model.unfreeze_backbone(layers=3)
"""

    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
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
        
        features = list(self.model.features.children())
        
        if layers is None:
            # Unfreeze all
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            if isinstance(layers, int):
                blocks_to_unfreeze = features[-layers:]
            elif isinstance(layers, list):
                blocks_to_unfreeze = [features[i] for i in layers]
            else:
                raise ValueError("layers should be None, int, or list of indices.")
            
            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
