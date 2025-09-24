"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from typing import List, Tuple

from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Predict on a target image with a target model
# Function created in: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set
def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    true_label: str = None,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
) -> dict:
    """Predicts on a target image and plots it with probability and optional true label.

    Returns a dictionary with predicted label and probability.
    """
    img = Image.open(image_path)

    # Transformation
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    
    model.to(device)
    model.eval()
    with torch.inference_mode():
        tensor_img = transform(img).unsqueeze(0).to(device)
        logits = model(tensor_img)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
    
    pred_class = class_names[pred_idx]
    pred_prob = probs.max().item()

    # Plot
    plt.figure()
    plt.imshow(img)
    title = f"Pred: {pred_class} | Prob: {pred_prob:.3f}"
    if true_label:
        title += f" | True: {true_label}"
    plt.title(title)
    plt.axis(False)
    plt.show()

    return {"pred_class": pred_class, "pred_prob": pred_prob, "true_label": true_label}
