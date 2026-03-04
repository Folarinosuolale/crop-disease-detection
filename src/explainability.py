import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_loader import get_transforms, CLASS_NAMES, DISPLAY_NAMES, IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Hook-based Grad-CAM for EfficientNetB0.

    Registers forward and backward hooks on the target layer to capture
    activations and gradients, then computes the class activation map.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Default to last convolutional layer in EfficientNetB0
        if target_layer is None:
            target_layer = model.features[-1]

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            heatmap: Numpy array (H, W) with values in [0, 1]
            predicted_class: The model's predicted class index
            confidence: Confidence score for the predicted class
            top5: List of (class_idx, probability) for top 5 predictions
        """
        self.model.eval()

        # Ensure gradient computation
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

        # Top 5 predictions
        top5_probs, top5_indices = probs[0].topk(5)
        top5 = [(idx.item(), prob.item()) for idx, prob in zip(top5_indices, top5_probs)]

        # Use predicted class if no target specified
        if target_class is None:
            target_class = predicted_class

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]                    # (C, H, W)
        activations = self.activations[0]                # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))             # (C,)

        # Weighted combination of activation maps
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)

        # ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cam.cpu().numpy()

        return heatmap, predicted_class, confidence, top5

    def remove_hooks(self):
        """Remove registered hooks to free resources."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def overlay_gradcam(original_image, heatmap, alpha=0.5):
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image: PIL Image or numpy array (H, W, 3) in RGB
        heatmap: Numpy array (h, w) with values in [0, 1]
        alpha: Blending factor (0 = only original, 1 = only heatmap)

    Returns:
        blended: Numpy array (H, W, 3) in RGB, values in [0, 255]
    """
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)

    h, w = original_image.shape[:2]

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap (jet: blue=cold, red=hot)
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    original_float = original_image.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    blended = (1 - alpha) * original_float + alpha * heatmap_float
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def denormalize_tensor(tensor):
    """Convert a normalized tensor back to a displayable image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu().clone()
    img = img * std + mean
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def explain_prediction(model, image_path, device, img_size=224):
    """End-to-end prediction + Grad-CAM explanation for a single image.

    Returns:
        dict with keys: predicted_class, predicted_name, confidence,
                        top5, original_image, heatmap, overlay
    """
    # Load and preprocess
    original_image = Image.open(image_path).convert('RGB')
    transform = get_transforms('eval', img_size)
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap, pred_class, confidence, top5 = gradcam.generate(input_tensor)
    gradcam.remove_hooks()

    # Create overlay on original (not preprocessed) image
    overlay = overlay_gradcam(original_image, heatmap, alpha=0.5)

    return {
        'predicted_class': pred_class,
        'predicted_name': CLASS_NAMES[pred_class],
        'predicted_display': DISPLAY_NAMES[CLASS_NAMES[pred_class]],
        'confidence': confidence,
        'top5': [(CLASS_NAMES[idx], prob) for idx, prob in top5],
        'original_image': np.array(original_image),
        'heatmap': heatmap,
        'overlay': overlay,
    }


def generate_gradcam_samples(model, test_files, test_labels, device,
                             n_per_class=2, save_dir='assets/gradcam_samples/',
                             img_size=224):
    """Generate and save Grad-CAM sample images for each class.

    Selects n_per_class correctly predicted images per class,
    generates Grad-CAM overlays, and saves them as PNG files.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    gradcam = GradCAM(model)
    transform = get_transforms('eval', img_size)

    # Group files by class
    class_files = {}
    for fpath, label in zip(test_files, test_labels):
        if label not in class_files:
            class_files[label] = []
        class_files[label].append(fpath)

    for class_idx in range(len(CLASS_NAMES)):
        class_name = CLASS_NAMES[class_idx]
        files = class_files.get(class_idx, [])
        saved_count = 0

        for fpath in files:
            if saved_count >= n_per_class:
                break

            try:
                original = Image.open(fpath).convert('RGB')
                input_tensor = transform(original).unsqueeze(0).to(device)

                heatmap, pred_class, confidence, _ = gradcam.generate(input_tensor)

                # Only save correctly predicted examples
                if pred_class != class_idx:
                    continue

                overlay = overlay_gradcam(original, heatmap, alpha=0.5)

                # Save as side-by-side: original | overlay
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(np.array(original))
                axes[0].set_title('Original')
                axes[0].axis('off')
                axes[1].imshow(overlay)
                axes[1].set_title(f'Grad-CAM ({confidence:.1%})')
                axes[1].axis('off')

                fig.suptitle(DISPLAY_NAMES[class_name], fontsize=12, fontweight='bold')
                fig.tight_layout()

                save_path = os.path.join(save_dir, f'{class_name}_{saved_count}.png')
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)

                saved_paths.append(save_path)
                saved_count += 1

            except Exception as e:
                print(f"  Warning: Could not process {fpath}: {e}")
                continue

        if saved_count < n_per_class:
            print(f"  Warning: Only saved {saved_count}/{n_per_class} samples for {class_name}")

    gradcam.remove_hooks()
    print(f"  Saved {len(saved_paths)} Grad-CAM samples to {save_dir}")
    return saved_paths
