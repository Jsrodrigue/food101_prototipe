from pathlib import Path
import json
from .model_utils import get_model_transforms
from typing import Tuple, Callable, Dict

def find_runs_in_dir(models_root: Path):
    """
    Discover model runs inside a given directory.

    This function assumes the following folder structure:
        <run_name>/artifacts/
            ├── best_model_info/best_model_info.json
            └── model/model_state_dict.pth

    Args:
        models_root (Path | str): Path to the root directory containing 
                                  one or more model families.

    Returns:
        list[dict]: A list of discovered runs. Each run is represented by:
            - "model_name": str, the model family name (parent folder of run)
            - "run_path": Path, the run directory path
            - "info": dict, metadata loaded from best_model_info.json
            - "state_dict": Path, path to model_state_dict.pth
    """
    models_root = Path(models_root)
    runs = []

    # Look only in <model>/<run>/artifacts
    for info_file in models_root.glob("*/*/artifacts/best_model_info/best_model_info.json"):
        state_dict_file = info_file.parent.parent / "model" / "model_state_dict.pth"
        if state_dict_file.exists():
            with open(info_file, "r") as f:
                info = json.load(f)
            run_dir = info_file.parent.parent.parent  # the run folder
            runs.append({
                "run_path": run_dir,
                "hyperparameters": info['hyperparameters'],
                "state_dict": state_dict_file
            })
    return runs



def collect_model_transforms(runs: list) -> Dict[Tuple[str, str], Callable]:
    """
    Collect the default torchvision transforms for a list of trained model runs.

    Each run is expected to have a structure like:
        {
            "run_path": ...,
            "hyperparameters": {
                "model_name": str,
                "version": str,
                ...
            },
            ...
        }

    Args:
        runs (list): List of run dictionaries, each containing 'hyperparameters'.

    Returns:
        Dict[Tuple[str, str], Callable]: Dictionary mapping (model_name, version)
            to the corresponding torchvision transforms for that model.
    """
    transforms_dict = {}
    for run in runs:
        hp = run.get("hyperparameters", {})
        model_name = hp.get("model_name")
        version = hp.get("version", None)
        if model_name is None:
            raise ValueError(f"Run {run.get('run_path')} missing 'model_name' in hyperparameters")

        # Use your existing get_model_transforms function
        transforms_dict[(model_name, version.lower() if version else None)] = get_model_transforms(
            model_name=model_name,
            version=version,
            augmentation=False
        )

    return transforms_dict