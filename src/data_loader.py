"""
Crop Disease Detection - Data Loading and Preparation
======================================================
Scans the PlantVillage dataset directory, creates stratified
train/val/test splits, defines augmentation transforms, and
produces PyTorch DataLoaders with class-weighted sampling.
"""

import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


# -- Class Names and Mappings --------------------------------------------------

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
]

DISPLAY_NAMES = {
    'Pepper__bell___Bacterial_spot': 'Pepper Bell - Bacterial Spot',
    'Pepper__bell___healthy': 'Pepper Bell - Healthy',
    'Potato___Early_blight': 'Potato - Early Blight',
    'Potato___Late_blight': 'Potato - Late Blight',
    'Potato___healthy': 'Potato - Healthy',
    'Tomato_Bacterial_spot': 'Tomato - Bacterial Spot',
    'Tomato_Early_blight': 'Tomato - Early Blight',
    'Tomato_Late_blight': 'Tomato - Late Blight',
    'Tomato_Leaf_Mold': 'Tomato - Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato - Septoria Leaf Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato - Spider Mites',
    'Tomato__Target_Spot': 'Tomato - Target Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato - Yellow Leaf Curl Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato - Mosaic Virus',
    'Tomato_healthy': 'Tomato - Healthy',
}

CROP_MAP = {
    'Pepper__bell___Bacterial_spot': 'Pepper',
    'Pepper__bell___healthy': 'Pepper',
    'Potato___Early_blight': 'Potato',
    'Potato___Late_blight': 'Potato',
    'Potato___healthy': 'Potato',
    'Tomato_Bacterial_spot': 'Tomato',
    'Tomato_Early_blight': 'Tomato',
    'Tomato_Late_blight': 'Tomato',
    'Tomato_Leaf_Mold': 'Tomato',
    'Tomato_Septoria_leaf_spot': 'Tomato',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato',
    'Tomato__Target_Spot': 'Tomato',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato',
    'Tomato__Tomato_mosaic_virus': 'Tomato',
    'Tomato_healthy': 'Tomato',
}

IS_HEALTHY = {name: 'healthy' in name.lower() for name in CLASS_NAMES}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -- Dataset Scanning ----------------------------------------------------------

def scan_dataset(data_dir):
    """Walk the dataset directory and collect (filepath, label_index) pairs.

    Only processes the 15 known class folders, ignoring any nested
    duplicates or non-image files.
    """
    file_list = []
    labels = []
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    skipped = 0

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  Warning: class folder not found: {class_dir}")
            continue

        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                file_list.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])
            else:
                skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} non-image files")

    return file_list, labels


# -- Transforms ----------------------------------------------------------------

def get_transforms(mode='train', img_size=224):
    """Return torchvision transforms for train or val/test mode."""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.05
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# -- Splitting -----------------------------------------------------------------

def split_dataset(file_list, labels, train_ratio=0.70, val_ratio=0.15,
                  random_state=42):
    """Stratified 70/15/15 split into train, validation, and test sets."""
    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: train vs (val + test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_list, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state,
    )

    # Second split: val vs test (50/50 of the remaining 30%)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=random_state,
    )

    return (
        (train_files, train_labels),
        (val_files, val_labels),
        (test_files, test_labels),
    )


# -- Class Weights -------------------------------------------------------------

def compute_class_weights(labels, num_classes=15):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weights.append(total / (num_classes * count))
    return torch.FloatTensor(weights)


# -- PyTorch Dataset -----------------------------------------------------------

class PlantVillageDataset(Dataset):
    """Custom Dataset for PlantVillage images with lazy loading."""

    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


# -- DataLoader Factory --------------------------------------------------------

def get_data_loaders(data_dir, batch_size=32, img_size=224, random_state=42,
                     num_workers=0, pin_memory=None):
    """Main entry point: scan dataset, split, create DataLoaders.

    Args:
        pin_memory: If None, auto-detect (True for CUDA, False for MPS/CPU).

    Returns:
        train_loader, val_loader, test_loader, class_weights, dataset_stats
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    print("  Scanning dataset directory...")
    file_list, labels = scan_dataset(data_dir)
    total = len(file_list)
    print(f"  Found {total} images across {len(CLASS_NAMES)} classes")

    # Class distribution
    counts = Counter(labels)
    class_counts = {CLASS_NAMES[k]: v for k, v in sorted(counts.items())}
    print(f"  Smallest class: {min(counts.values())} images")
    print(f"  Largest class:  {max(counts.values())} images")
    print(f"  Imbalance ratio: {max(counts.values()) / max(min(counts.values()), 1):.1f}:1")

    # Split
    print("  Creating stratified 70/15/15 split...")
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = \
        split_dataset(file_list, labels, random_state=random_state)

    print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Transforms
    train_transform = get_transforms('train', img_size)
    eval_transform = get_transforms('eval', img_size)

    # Datasets
    train_dataset = PlantVillageDataset(train_files, train_labels, train_transform)
    val_dataset = PlantVillageDataset(val_files, val_labels, eval_transform)
    test_dataset = PlantVillageDataset(test_files, test_labels, eval_transform)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # Class weights (computed from training set only)
    class_weights = compute_class_weights(train_labels, num_classes=len(CLASS_NAMES))

    # Dataset stats for saving
    dataset_stats = {
        'total_images': total,
        'num_classes': len(CLASS_NAMES),
        'class_counts': class_counts,
        'train_size': len(train_files),
        'val_size': len(val_files),
        'test_size': len(test_files),
        'img_size': img_size,
        'batch_size': batch_size,
        'class_names': CLASS_NAMES,
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
    }

    return train_loader, val_loader, test_loader, class_weights, dataset_stats
