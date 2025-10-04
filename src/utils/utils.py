# src/utils.py
import argparse
import json
import os
import random
import shutil
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch














# -------------------- LOGGING & PLOTTING -------------------- #








def get_test_args():
    """
    Parse command-line arguments for testing models.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - architectures (list[str]): Architectures to test.
            - base_dir (str): Base folder containing model runs.
            - batch_size (int): Batch size for test loader.
            - device (str): Device to run models on ('cpu' or 'cuda').
            - save_results (bool): Whether to save test metrics.
    """
    parser = argparse.ArgumentParser(description="Test selected models")

    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["mobilenetv2", "efficientnetb0"],
        help="Architectures to test",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="selected_models/",
        help="Base directory for model runs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for test loader"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run models on",
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Whether to save test metrics"
    )

    return parser.parse_args()
