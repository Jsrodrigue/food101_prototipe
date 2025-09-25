import os
import random
import shutil
from pathlib import Path

from torchvision.datasets import Food101
from tqdm import tqdm


def prepare_food101_subset(
    root: str = "./data",
    subset_root: str = "./data_subset",
    zip_path: str = "./food101_subset",
    selected_classes=None,
    samples_per_class: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    delete_original: bool = True,
    seed: int = 42,
) -> tuple:
    """
    Create a balanced subset of Food101 with train/val/test splits and optional zip.
    Returns paths for train and test ready for DataLoader.
    """
    random.seed(seed)
    if selected_classes is None:
        selected_classes = [
            "sushi",
            "pizza",
            "steak",
            "hamburger",
            "ramen",
            "tacos",
            "pancakes",
            "lasagna",
            "ice_cream",
            "carrot_cake",
        ]
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    data_dir = Path(root) / "food-101"
    images_dir = data_dir / "images"

    # -------------------------
    # Step 1: Download if missing
    # -------------------------
    if not images_dir.exists():
        print("Images folder not found. Downloading Food101 dataset...")
        Food101(root=root, split="train", download=True)
        Food101(root=root, split="test", download=True)
        print("Download completed.")
    else:
        print("Images folder already exists, skipping download.")

    # -------------------------
    # Step 2: Create subset
    # -------------------------
    subset_root = Path(subset_root)
    splits = ["train", "val", "test"]
    for split in splits:
        for cls in selected_classes:
            (subset_root / split / cls).mkdir(parents=True, exist_ok=True)

    n_train = int(samples_per_class * train_ratio)
    n_val = int(samples_per_class * val_ratio)
    n_test = samples_per_class - n_train - n_val
    print(f"Per class: Train={n_train}, Val={n_val}, Test={n_test}")

    for cls in tqdm(selected_classes, desc="Processing classes"):
        class_path = images_dir / cls
        if not class_path.exists():
            print(f"Warning: folder {cls} does not exist, skipping.")
            continue

        all_images = os.listdir(class_path)
        if len(all_images) < samples_per_class:
            print(f"Warning: {cls} only has {len(all_images)} images, adjusting...")
            selected_images = all_images
            total = len(selected_images)
            n_train_actual = int(total * train_ratio)
            n_val_actual = int(total * val_ratio)
            n_test_actual = total - n_train_actual - n_val_actual
        else:
            selected_images = random.sample(all_images, samples_per_class)
            n_train_actual, n_val_actual, n_test_actual = n_train, n_val, n_test

        train_imgs = selected_images[:n_train_actual]
        val_imgs = selected_images[n_train_actual : n_train_actual + n_val_actual]
        test_imgs = selected_images[n_train_actual + n_val_actual :]

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            for img in split_imgs:
                shutil.copy2(class_path / img, subset_root / split_name / cls / img)

    print(f"Subset created in {subset_root}")

    # -------------------------
    # Step 3: Optional zip
    # -------------------------
    zip_path = Path(zip_path)
    if zip_path.with_suffix(".zip").exists():
        zip_path.with_suffix(".zip").unlink()
    shutil.make_archive(str(zip_path), "zip", subset_root)
    print(f"Subset zipped successfully at {zip_path}.zip")

    # -------------------------
    # Step 4: Delete original if requested
    # -------------------------
    if delete_original and data_dir.exists():
        shutil.rmtree(data_dir)
        print("Original Food101 dataset deleted.")

    # -------------------------
    # Step 5: Summary
    # -------------------------
    print("\nSubset summary:")
    for split in splits:
        print(f"\n{split.upper()}:")
        total_split = 0
        for cls in selected_classes:
            cls_path = subset_root / split / cls
            count = len(os.listdir(cls_path)) if cls_path.exists() else 0
            total_split += count
            print(f"  {cls}: {count}")
        print(f"  Total {split}: {total_split}")

    # Return paths for train and test
    return subset_root / "train", subset_root / "test"


if __name__ == "__main__":
    prepare_food101_subset()
