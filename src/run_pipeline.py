import os
import sys
import json
import time

import torch
import pandas as pd

# Ensure src is importable when running as script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.data_loader import (
    get_data_loaders, CLASS_NAMES, DISPLAY_NAMES, CROP_MAP, IS_HEALTHY,
)
from src.model_training import (
    build_model, count_parameters, train_model, evaluate_model,
    save_model,
)
from src.explainability import generate_gradcam_samples


def get_device():
    """Auto-detect the best available device."""
    if torch.backends.mps.is_available():
        try:
            # Test MPS with a small tensor to verify it works
            t = torch.zeros(1, device='mps')
            del t
            return torch.device('mps')
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def run_full_pipeline(data_path=None, batch_size=32, phase1_epochs=5,
                      phase2_epochs=10, patience=5):
    """Execute the complete crop disease detection pipeline."""

    start_time = time.time()

    print("=" * 60)
    print("CROP DISEASE DETECTION - FULL PIPELINE")
    print("=" * 60)

    # Detect device
    device = get_device()
    print(f"Using device: {device}")

    if data_path is None:
        data_path = os.path.join(ROOT, 'data', 'PlantVillage')

    models_dir = os.path.join(ROOT, 'models')
    assets_dir = os.path.join(ROOT, 'assets', 'gradcam_samples')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # ==================================================================
    # [1/7] Load & Prepare Data
    # ==================================================================
    print("\n" + "=" * 60)
    print("[1/7] Loading and preparing dataset...")
    print("=" * 60)

    train_loader, val_loader, test_loader, class_weights, dataset_stats = \
        get_data_loaders(data_path, batch_size=batch_size)

    # Print class distribution
    print("\n  Class Distribution:")
    for name in CLASS_NAMES:
        count = dataset_stats['class_counts'].get(name, 0)
        crop = CROP_MAP[name]
        healthy = "Healthy" if IS_HEALTHY[name] else "Disease"
        print(f"    {DISPLAY_NAMES[name]:45s} | {count:5d} | {crop:7s} | {healthy}")

    # ==================================================================
    # [2/7] Build Model
    # ==================================================================
    print("\n" + "=" * 60)
    print("[2/7] Building EfficientNetB0 model...")
    print("=" * 60)

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=True)
    model = model.to(device)

    params = count_parameters(model)
    print(f"  Total parameters:     {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Frozen parameters:    {params['frozen']:,}")

    # ==================================================================
    # [3/7] Phase 1: Train Classifier Head
    # ==================================================================
    print("\n" + "=" * 60)
    print("[3/7] Phase 1: Training classifier head (backbone frozen)...")
    print("=" * 60)

    # Training handles both phases internally
    model, history = train_model(
        model, train_loader, val_loader, class_weights, device,
        phase1_epochs=phase1_epochs,
        phase2_epochs=0,  # Phase 1 only
        lr_head=1e-3,
    )

    # ==================================================================
    # [4/7] Phase 2: Fine-tune Last 3 Backbone Blocks
    # ==================================================================
    print("\n" + "=" * 60)
    print("[4/7] Phase 2: Fine-tuning last 3 backbone blocks...")
    print("=" * 60)

    model, history_phase2 = train_model(
        model, train_loader, val_loader, class_weights, device,
        phase1_epochs=0,  # Skip phase 1
        phase2_epochs=phase2_epochs,
        lr_finetune=1e-4,
        patience=patience,
    )

    params2 = count_parameters(model)
    print(f"  Phase 2 trainable parameters: {params2['trainable']:,} "
          f"(of {params2['total']:,} total)")

    # Merge histories
    for key in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']:
        history[key].extend(history_phase2[key])
    history['total_epochs'] = len(history['train_loss'])
    history['best_val_loss'] = history_phase2['best_val_loss']

    # ==================================================================
    # [5/7] Evaluate on Test Set
    # ==================================================================
    print("\n" + "=" * 60)
    print("[5/7] Evaluating on test set...")
    print("=" * 60)

    test_results = evaluate_model(model, test_loader, CLASS_NAMES, device)

    print(f"\n  Test Results:")
    print(f"    Accuracy:        {test_results['accuracy']:.4f}")
    print(f"    Macro Precision: {test_results['macro_precision']:.4f}")
    print(f"    Macro Recall:    {test_results['macro_recall']:.4f}")
    print(f"    Macro F1:        {test_results['macro_f1']:.4f}")
    print(f"    Weighted F1:     {test_results['weighted_f1']:.4f}")

    print(f"\n  Per-Class Performance:")
    print(f"    {'Class':<45s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'N':>5s}")
    print(f"    {'-'*70}")
    for class_name in CLASS_NAMES:
        m = test_results['per_class'].get(class_name, {})
        print(f"    {DISPLAY_NAMES[class_name]:<45s} "
              f"{m.get('precision', 0):.3f}  "
              f"{m.get('recall', 0):.3f}  "
              f"{m.get('f1', 0):.3f}  "
              f"{m.get('support', 0):4d}")

    # ==================================================================
    # [6/7] Generate Grad-CAM Explanations
    # ==================================================================
    print("\n" + "=" * 60)
    print("[6/7] Generating Grad-CAM explanations...")
    print("=" * 60)

    gradcam_paths = generate_gradcam_samples(
        model,
        dataset_stats['test_files'],
        dataset_stats['test_labels'],
        device,
        n_per_class=2,
        save_dir=assets_dir,
    )

    # ==================================================================
    # [7/7] Save All Artifacts
    # ==================================================================
    print("\n" + "=" * 60)
    print("[7/7] Saving all artifacts...")
    print("=" * 60)

    # Save model and artifacts
    artifacts = {
        'class_names': CLASS_NAMES,
        'display_names': DISPLAY_NAMES,
        'crop_map': CROP_MAP,
        'is_healthy': IS_HEALTHY,
        'img_size': 224,
        'num_classes': len(CLASS_NAMES),
    }
    save_model(model, artifacts, path=models_dir)

    # Save pipeline results (main metrics file)
    pipeline_results = {
        'dataset': {
            'total_images': dataset_stats['total_images'],
            'num_classes': dataset_stats['num_classes'],
            'class_counts': dataset_stats['class_counts'],
            'train_size': dataset_stats['train_size'],
            'val_size': dataset_stats['val_size'],
            'test_size': dataset_stats['test_size'],
        },
        'model': {
            'architecture': 'EfficientNetB0',
            'pretrained': 'ImageNet',
            'num_params': params['total'],
            'input_size': 224,
        },
        'training': {
            'phase1_epochs': phase1_epochs,
            'phase2_epochs': phase2_epochs,
            'total_epochs': history['total_epochs'],
            'phase_boundary': phase1_epochs,
            'batch_size': batch_size,
            'lr_head': 1e-3,
            'lr_finetune': 1e-4,
            'patience': patience,
            'best_val_loss': float(history['best_val_loss']),
            'device': str(device),
            'finetune_blocks': 3,
        },
        'test_metrics': {
            'accuracy': test_results['accuracy'],
            'macro_precision': test_results['macro_precision'],
            'macro_recall': test_results['macro_recall'],
            'macro_f1': test_results['macro_f1'],
            'weighted_f1': test_results['weighted_f1'],
        },
        'per_class_metrics': test_results['per_class'],
    }

    with open(os.path.join(models_dir, 'pipeline_results.json'), 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    print(f"  Pipeline results saved to models/pipeline_results.json")

    # Save confusion matrix
    with open(os.path.join(models_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump({
            'matrix': test_results['confusion_matrix'],
            'class_names': CLASS_NAMES,
        }, f, indent=2)
    print(f"  Confusion matrix saved to models/confusion_matrix.json")

    # Save per-class metrics as CSV
    per_class_rows = []
    for class_name in CLASS_NAMES:
        m = test_results['per_class'].get(class_name, {})
        per_class_rows.append({
            'class': class_name,
            'display_name': DISPLAY_NAMES[class_name],
            'crop': CROP_MAP[class_name],
            'is_healthy': IS_HEALTHY[class_name],
            'precision': m.get('precision', 0),
            'recall': m.get('recall', 0),
            'f1': m.get('f1', 0),
            'support': m.get('support', 0),
        })
    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(models_dir, 'per_class_metrics.csv'), index=False
    )
    print(f"  Per-class metrics saved to models/per_class_metrics.csv")

    # Save training history
    training_history = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'lr': history['lr'],
        'phase_boundary': phase1_epochs,
        'total_epochs': history['total_epochs'],
    }
    with open(os.path.join(models_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"  Training history saved to models/training_history.json")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print(f"Pipeline complete! Total time: {minutes}m {seconds}s")
    print(f"All artifacts saved to {models_dir}/")
    print("=" * 60)

    return pipeline_results


if __name__ == '__main__':
    run_full_pipeline()
