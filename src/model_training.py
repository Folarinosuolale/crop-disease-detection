"""
Crop Disease Detection - Model Training and Evaluation
=======================================================
EfficientNetB0 with transfer learning. Two-phase training:
Phase 1 trains only the classifier head (backbone frozen),
Phase 2 fine-tunes the full network with a smaller learning rate.
"""

import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from tqdm import tqdm
import joblib


# -- Model Architecture -------------------------------------------------------

def build_model(num_classes=15, pretrained=True):
    """Load EfficientNetB0 with ImageNet weights and replace the classifier head."""
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)

    # Freeze all backbone parameters
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier: EfficientNetB0 outputs 1280 features
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


def freeze_backbone(model):
    """Freeze all parameters in the feature extraction backbone."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model, num_blocks=3):
    """Unfreeze the last `num_blocks` of the backbone for fine-tuning.

    EfficientNetB0 has 9 feature blocks (indices 0-8).  Unfreezing only
    the last few keeps early layers (edges, textures) frozen and adapts
    the higher-level, domain-specific layers.  This is dramatically
    faster than unfreezing everything, especially on MPS.
    """
    # First freeze everything
    for param in model.features.parameters():
        param.requires_grad = False

    # Then unfreeze the last N blocks
    total_blocks = len(model.features)
    start = max(0, total_blocks - num_blocks)
    for i in range(start, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True


def count_parameters(model):
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {'total': total, 'trainable': trainable, 'frozen': frozen}


# -- Training Loop -------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for a single epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='  Training', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate on validation set. Returns loss, accuracy, predictions."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc='  Validating', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# -- Full Training Procedure ---------------------------------------------------

def train_model(model, train_loader, val_loader, class_weights, device,
                phase1_epochs=5, phase2_epochs=25,
                lr_head=1e-3, lr_finetune=1e-4, patience=7):
    """Two-phase training with early stopping.

    Phase 1: Train only the classifier head (backbone frozen).
    Phase 2: Unfreeze all layers and fine-tune with smaller LR.

    Returns:
        model: Trained model with best weights restored
        history: Dict of per-epoch metrics for both phases
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'phase_boundary': phase1_epochs,
        'lr': [],
    }

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    total_epoch = 0

    # ---- Phase 1: Head Only ----
    print(f"\n  Phase 1: Training classifier head ({phase1_epochs} epochs, lr={lr_head})")
    freeze_backbone(model)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head, weight_decay=1e-4,
    )

    for epoch in range(phase1_epochs):
        total_epoch += 1
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr_head)

        print(f"  Epoch {total_epoch:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    # ---- Phase 2: Full Fine-tuning ----
    print(f"\n  Phase 2: Fine-tuning full model (up to {phase2_epochs} epochs, lr={lr_finetune})")
    unfreeze_backbone(model)
    optimizer = AdamW(model.parameters(), lr=lr_finetune, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    for epoch in range(phase2_epochs):
        total_epoch += 1
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            improved = " *"
        else:
            epochs_no_improve += 1

        print(f"  Epoch {total_epoch:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.1e}{improved}")

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered at epoch {total_epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Restored best model (val_loss={best_val_loss:.4f})")

    history['total_epochs'] = total_epoch
    history['best_val_loss'] = best_val_loss

    return model, history


# -- Evaluation ----------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, loader, class_names, device):
    """Comprehensive evaluation on a dataset split.

    Returns dict with overall metrics, per-class metrics, and confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc='  Evaluating', leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Per-class metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    results = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': {},
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
    }

    for class_name in class_names:
        if class_name in report:
            results['per_class'][class_name] = {
                'precision': float(report[class_name]['precision']),
                'recall': float(report[class_name]['recall']),
                'f1': float(report[class_name]['f1-score']),
                'support': int(report[class_name]['support']),
            }

    return results


# -- Model Persistence ---------------------------------------------------------

def save_model(model, artifacts, path='models/'):
    """Save the trained model and preprocessing artifacts."""
    os.makedirs(path, exist_ok=True)

    # Save model state dict (PyTorch best practice)
    torch.save(model.state_dict(), os.path.join(path, 'best_model.pt'))

    # Save artifacts (class names, transforms config, etc.)
    joblib.dump(artifacts, os.path.join(path, 'artifacts.pkl'))

    print(f"  Model saved to {os.path.join(path, 'best_model.pt')}")
    print(f"  Artifacts saved to {os.path.join(path, 'artifacts.pkl')}")


def load_model(path='models/', num_classes=15, device='cpu'):
    """Load a saved model and artifacts."""
    model = build_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(
        os.path.join(path, 'best_model.pt'),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    artifacts = joblib.load(os.path.join(path, 'artifacts.pkl'))

    return model, artifacts
