"""
food101_subset_generator.py

This script downloads the Food101 dataset using torchvision,
creates a balanced subset with 10 fixed valid classes (100 images per class),
splits into train/test (80/20), saves the subset in folders ready for ImageFolder,
compresses the subset into a ZIP, and deletes the original dataset to save space.
"""

import os
import random
import shutil
from tqdm import tqdm
from torchvision.datasets import Food101

# -------------------------------
# Parameters
# -------------------------------
root = './data'                # folder to download Food101
subset_root = './data_subset'  # folder for balanced subset
zip_path = './food101_subset'  # output zip path (without .zip)
# 10 valid Food101 classes
selected_classes = [
    "sushi", "pizza", "steak", "hamburger", "ramen",
    "tacos", "pancakes", "lasagna", "ice_cream", "carrot_cake"
]
samples_per_class = 100
train_ratio = 0.8
random.seed(42)

# -------------------------------
# Functions
# -------------------------------

def download_food101(root):
    print("Downloading Food101 dataset...")
    Food101(root=root, split='train', download=True, transform=None)
    print("Food101 downloaded")

def create_subset_from_folder(source_root, subset_root, selected_classes, samples_per_class, train_ratio):
    print("Creating balanced subset...")

    # Create train/test folders
    for split in ["train", "test"]:
        for class_name in selected_classes:
            os.makedirs(os.path.join(subset_root, split, class_name), exist_ok=True)

    # Copy images
    for class_name in tqdm(selected_classes, desc="Processing classes"):
        class_folder = os.path.join(source_root, class_name)
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Class folder '{class_folder}' does not exist")
        
        images = os.listdir(class_folder)
        
        if len(images) < samples_per_class:
            raise ValueError(f"Class '{class_name}' has fewer than {samples_per_class} images")
        
        # Sample images_per_class randomly
        selected_images = random.sample(images, samples_per_class)
        split_point = int(train_ratio * samples_per_class)
        train_images = selected_images[:split_point]
        test_images  = selected_images[split_point:]
        
        # Copy train images
        for img in train_images:
            shutil.copy(os.path.join(class_folder, img),
                        os.path.join(subset_root, "train", class_name, img))
        
        # Copy test images
        for img in test_images:
            shutil.copy(os.path.join(class_folder, img),
                        os.path.join(subset_root, "test", class_name, img))

    print("Balanced subset created at:", subset_root)

def zip_subset(subset_root, zip_path):
    if os.path.exists(zip_path + '.zip'):
        os.remove(zip_path + '.zip')
    shutil.make_archive(zip_path, 'zip', subset_root)
    print(f"Balanced subset zipped successfully at {zip_path}.zip")

def delete_food101(root):
    food101_root = os.path.join(root, "food-101")
    if os.path.exists(food101_root):
        shutil.rmtree(food101_root)
        print(f"Deleted original Food101 folder at {food101_root}")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Step 1: download dataset
    download_food101(root)

    # Step 2: create subset from the downloaded folder
    images_folder = os.path.join(root, "food-101", "images")
    create_subset_from_folder(images_folder, subset_root, selected_classes, samples_per_class, train_ratio)

    # Step 3: zip the subset
    zip_subset(subset_root, zip_path)

    # Step 4: delete original Food101 to save space
    delete_food101(root)
