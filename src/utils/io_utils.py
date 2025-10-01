import torch 
from pathlib import Path

def save_model(model, target_dir, model_name):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Use .pth or .pt"
    save_path = target_dir / model_name
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to {save_path}")