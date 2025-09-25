import mlflow
import mlflow.pytorch
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# --- Ruta local de mlruns ---
mlruns_dir = Path(r"C:/Users/Juan/Desktop/food101Mini/mlruns")
mlflow.set_tracking_uri(f"file:///{mlruns_dir.as_posix()}")

# --- Nombre del experimento ---
mlflow.set_experiment("food101_experiment")

# --- Run ID correcto ---
run_id = "171acc4ab699412a8851ce84719d2f0a"

# --- Cargar modelo desde MLflow ---
model_uri = f"runs:/{run_id}/best_model"
model = mlflow.pytorch.load_model(model_uri)
model.eval()  # importante para evaluación/inferencia

# --- Imagen de prueba ---
img_path = Path(r"C:/Users/Juan/Desktop/food101Mini/data/dataset/train/carrot_cake/1003032.jpg")
image = Image.open(img_path).convert("RGB")

# --- Transformaciones EfficientNet (ImageNet) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

x = transform(image).unsqueeze(0)  # agregar dimensión batch

# --- Predicción ---
with torch.inference_mode():
    preds = model(x)
    predicted_class = preds.argmax(dim=1).item()

print("Predicted class index:", predicted_class)
