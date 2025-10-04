from pathlib import Path

import hydra
from omegaconf import DictConfig
from src.utils.runs_utils import find_runs_in_dir, collect_model_transforms
from src.data_setup import prepare_data

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):

    # ---------------- DATA PATHS ----------------
    test_dir = prepare_data(cfg, get_train=False, get_val=False, get_test=True)['test_dir']
    
    # -------------GET RUNS INFO -----------------
    runs = find_runs_in_dir(Path.cwd() / cfg.test.runs_dir)
    print(f"[INFO] Testing {len(runs)} models...")

    #--------------TRANSFORMS-----------------------

    # List of dictionaries with keys (model_name, version) y values the transform
    transforms_dicts = collect_model_transforms(runs)

    print(transforms_dicts)

        #------------ Test loader ------------------

if __name__ == "__main__":
    main()
