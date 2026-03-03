# Crop Disease Detection - Technical Report

## 1. Executive Summary

This report documents the development of a deep learning system for automated crop disease detection from leaf images. The project delivers an end-to-end pipeline that takes raw leaf photographs as input and produces disease classifications with visual explanations showing which regions of the leaf the model examined.

**Key deliverables:**
- EfficientNetB0 model trained via two-phase transfer learning achieving **99.61% test accuracy** and **0.9963 macro F1** across 15 classes
- Grad-CAM visual explainability showing spatial attention maps for every prediction
- Weighted CrossEntropyLoss handling a 21:1 class imbalance ratio (152 to 3,209 images per class)
- Interactive Streamlit dashboard with live prediction, Grad-CAM overlay, and treatment recommendations
- Complete artifact pipeline: model weights, training curves, confusion matrices, per-class metrics, and Grad-CAM samples

---

## 2. Problem Statement

### Agricultural Context
Plant diseases are responsible for up to 40% of global crop losses annually, translating to roughly $220 billion in economic damage (FAO estimates). For the three crops in this study:
- **Tomato:** Late blight alone caused the Irish Potato Famine and still devastates tomato crops worldwide. Bacterial spot, septoria leaf spot, and leaf curl virus are persistent threats
- **Potato:** Early blight and late blight are the two most economically significant potato diseases globally
- **Pepper:** Bacterial spot can reduce yields by 50% or more in affected fields

The fundamental challenge is **timely identification**. A trained agronomist can diagnose most leaf diseases by sight, but there are not enough agronomists to cover every field, especially in developing regions where smallholder farmers produce the majority of food.

### Objective
Build an image classification model that:
1. Accurately classifies leaf images into 15 disease/healthy categories across 3 crops
2. Provides visual explanations (Grad-CAM) showing which leaf regions drove the prediction
3. Handles significant class imbalance without sacrificing performance on minority classes

---

## 3. Data Description

### Source
**PlantVillage Dataset** (Penn State University)
- 20,639 leaf photographs across 15 classes
- 3 crops: Pepper (2 classes), Potato (3 classes), Tomato (10 classes)
- Lab-controlled conditions with consistent backgrounds
- RGB images at varying resolutions

### Class Distribution

| Class | Crop | Type | Images | % of Total |
|---|---|---|---|---|
| Tomato - Yellow Leaf Curl Virus | Tomato | Disease | 3,209 | 15.6% |
| Tomato - Bacterial Spot | Tomato | Disease | 2,127 | 10.3% |
| Tomato - Late Blight | Tomato | Disease | 1,909 | 9.3% |
| Tomato - Septoria Leaf Spot | Tomato | Disease | 1,771 | 8.6% |
| Tomato - Spider Mites | Tomato | Disease | 1,676 | 8.1% |
| Tomato - Healthy | Tomato | Healthy | 1,591 | 7.7% |
| Pepper Bell - Healthy | Pepper | Healthy | 1,478 | 7.2% |
| Tomato - Target Spot | Tomato | Disease | 1,404 | 6.8% |
| Potato - Early Blight | Potato | Disease | 1,000 | 4.8% |
| Potato - Late Blight | Potato | Disease | 1,000 | 4.8% |
| Tomato - Early Blight | Tomato | Disease | 1,000 | 4.8% |
| Pepper Bell - Bacterial Spot | Pepper | Disease | 997 | 4.8% |
| Tomato - Leaf Mold | Tomato | Disease | 952 | 4.6% |
| Tomato - Mosaic Virus | Tomato | Disease | 373 | 1.8% |
| Potato - Healthy | Potato | Healthy | 152 | 0.7% |

### Key Data Characteristics
- **Class imbalance:** 21.1:1 ratio between largest and smallest classes
- **Crop imbalance:** Tomato dominates with 77.6% of images (10 classes), Pepper has 12.0% (2 classes), Potato has 10.4% (3 classes)
- **Healthy vs Disease:** 3 healthy classes (4,699 images, 22.8%) vs 12 disease classes (15,940 images, 77.2%)
- **No missing data:** All images are loadable RGB photographs

---

## 4. Data Preparation

### 4.1 Train/Validation/Test Split
**Stratified 70/15/15 split** using sklearn's train_test_split (two-stage):
- Train: ~14,447 images (70%)
- Validation: ~3,096 images (15%)
- Test: ~3,096 images (15%)

Stratification ensures each split preserves the original class distribution, which is critical when the smallest class has only 152 samples (roughly 106 train, 23 val, 23 test).

### 4.2 Data Augmentation
Training images undergo random augmentation to increase effective dataset size and reduce overfitting:

| Augmentation | Parameters | Rationale |
|---|---|---|
| RandomHorizontalFlip | p=0.5 | Leaf orientation is arbitrary |
| RandomVerticalFlip | p=0.3 | Same rationale, lower probability to avoid excessive variation |
| RandomRotation | 15 degrees | Leaves can be photographed at any angle |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05 | Simulates varying lighting conditions in the field |
| RandomAffine | translate=(0.1, 0.1) | Simulates off-center leaf placement |

Validation and test images receive only resize (224x224) and ImageNet normalization.

### 4.3 Class Imbalance Handling
**Inverse-frequency class weights** applied to CrossEntropyLoss:

`weight_i = total_samples / (num_classes * count_i)`

This gives the loss function higher weight for minority classes (Potato Healthy gets ~9x the weight of Tomato Yellow Leaf Curl Virus). This approach was chosen over oversampling because:
- No duplicate images in training (oversampling would repeat the same 152 Potato Healthy images many times, risking overfitting to those specific images)
- More memory-efficient than generating synthetic augmented copies
- Works naturally with PyTorch's CrossEntropyLoss

### 4.4 Preprocessing Pipeline
All images are:
1. Resized to 224x224 pixels (EfficientNetB0's expected input size)
2. Converted to PyTorch tensors (float32, range [0, 1])
3. Normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

ImageNet normalization is used because the backbone was pretrained on ImageNet. Using different normalization would create a distribution mismatch between what the model learned and what it receives.

---

## 5. Model Architecture

### 5.1 Why EfficientNetB0
EfficientNetB0 was selected over alternatives for several reasons:

| Consideration | EfficientNetB0 | ResNet50 | VGG16 |
|---|---|---|---|
| Parameters | 5.3M | 25.6M | 138M |
| ImageNet Top-1 | 77.1% | 76.1% | 71.6% |
| CPU trainable | Yes | Yes | Slow |
| Architecture | Compound scaling | Residual blocks | Plain stacking |

EfficientNetB0 achieves competitive ImageNet accuracy with 5x fewer parameters than ResNet50 and 26x fewer than VGG16, making it practical for training on consumer hardware (including Apple Silicon via MPS).

### 5.2 Architecture Modifications
The pretrained EfficientNetB0 classifier head (Linear(1280, 1000) for ImageNet's 1000 classes) is replaced with:

```
Sequential(
    Dropout(p=0.3),
    Linear(1280, 15)
)
```

- **Dropout(0.3):** Regularization to prevent overfitting, especially important given the small Potato Healthy class
- **Linear(1280, 15):** Maps from EfficientNetB0's 1,280-dimensional feature space to our 15 disease classes

### 5.3 Device Auto-Detection
The pipeline automatically selects the best available compute device:
1. **MPS** (Apple Silicon GPU) -- tested with a small tensor operation before use
2. **CUDA** (NVIDIA GPU) -- standard GPU acceleration
3. **CPU** -- fallback, slower but always available

---

## 6. Training Procedure

### 6.1 Two-Phase Transfer Learning

#### Phase 1: Classifier Head Training (Backbone Frozen)
- **What trains:** Only the new classifier head (Linear layer + Dropout)
- **What is frozen:** All 5.3M backbone parameters
- **Learning rate:** 1e-3
- **Epochs:** 5
- **Optimizer:** AdamW (weight_decay=1e-4)
- **Purpose:** Teach the classifier to map EfficientNetB0's general visual features to our 15 disease classes without disturbing the pretrained representations

This phase is fast because only ~19,215 parameters are being updated (1280 x 15 = 19,200 weights + 15 biases).

#### Phase 2: Partial Fine-Tuning (Last 3 Backbone Blocks Unfrozen)
- **What trains:** Last 3 backbone blocks + classifier head (3,174,955 trainable of 4,026,763 total parameters)
- **Learning rate:** 1e-4 (10x lower than Phase 1)
- **Epochs:** Up to 10 (early stopping with patience=5)
- **Optimizer:** AdamW (weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5 -- halves LR after 3 epochs without improvement)
- **Early stopping:** Training stops if validation loss does not improve for 5 consecutive epochs
- **Best validation loss:** 0.0170 at epoch 9
- **Purpose:** Adapt the backbone's higher-level visual features to the specific domain of leaf pathology while preserving lower-level feature extractors

The lower learning rate in Phase 2 is critical. The backbone already contains useful visual features (edges, textures, shapes) from ImageNet. A high learning rate would destroy these features before they could be adapted, a phenomenon known as catastrophic forgetting.

### 6.2 Loss Function
**Weighted CrossEntropyLoss** with class weights computed from training set inverse frequencies. This ensures the model pays proportionally more attention to minority classes during training.

### 6.3 Regularization
Three mechanisms prevent overfitting:
1. **Dropout (0.3)** in the classifier head
2. **Weight decay (1e-4)** in AdamW optimizer (L2 regularization)
3. **Gradient clipping (max_norm=1.0)** to prevent gradient explosion during fine-tuning
4. **Data augmentation** during training (Section 4.2)
5. **Early stopping** based on validation loss

### 6.4 Training Results

Training completed in approximately **65 minutes on MPS (Apple Silicon)** over a total of **15 epochs** (5 Phase 1 + 10 Phase 2).

**Phase 1 (5 epochs):** The classifier head converged quickly, learning to map EfficientNetB0's frozen feature representations to the 15 disease classes. Validation loss decreased steadily throughout all 5 epochs, confirming that the pretrained features contain sufficient information for disease discrimination even before fine-tuning.

**Phase 2 (10 epochs):** With the last 3 backbone blocks unfrozen (3,174,955 of 4,026,763 parameters trainable), the model refined its feature representations for the plant pathology domain. The best validation loss of **0.0170** was achieved at epoch 9. The learning rate scheduler reduced the LR when plateaus were detected, enabling fine-grained convergence in later epochs.

The two-phase approach proved effective: Phase 1 established a strong baseline by training only the classifier head, and Phase 2 adapted the higher-level backbone features to capture disease-specific visual patterns (lesion textures, discoloration gradients, spot morphology) that ImageNet pretraining alone does not encode.

---

## 7. Evaluation

### 7.1 Overall Test Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.9961 |
| Macro Precision | 0.9958 |
| Macro Recall | 0.9967 |
| Macro F1 | 0.9963 |
| Weighted F1 | 0.9961 |

### 7.2 Per-Class Performance

| Class | Precision | Recall | F1-Score | Test Samples |
|---|---|---|---|---|
| Pepper Bell - Bacterial Spot | 1.000 | 1.000 | 1.000 | 149 |
| Pepper Bell - Healthy | 1.000 | 1.000 | 1.000 | 221 |
| Potato - Early Blight | 1.000 | 1.000 | 1.000 | 150 |
| Potato - Late Blight | 0.993 | 1.000 | 0.997 | 150 |
| Potato - Healthy | 1.000 | 1.000 | 1.000 | 23 |
| Tomato - Bacterial Spot | 0.991 | 1.000 | 0.995 | 319 |
| Tomato - Early Blight | 0.987 | 0.993 | 0.990 | 150 |
| Tomato - Late Blight | 1.000 | 0.993 | 0.997 | 287 |
| Tomato - Leaf Mold | 0.979 | 1.000 | 0.990 | 143 |
| Tomato - Septoria Leaf Spot | 1.000 | 0.981 | 0.991 | 266 |
| Tomato - Spider Mites | 0.992 | 1.000 | 0.996 | 252 |
| Tomato - Target Spot | 0.995 | 0.986 | 0.990 | 210 |
| Tomato - Yellow Leaf Curl Virus | 1.000 | 0.998 | 0.999 | 482 |
| Tomato - Mosaic Virus | 1.000 | 1.000 | 1.000 | 56 |
| Tomato - Healthy | 1.000 | 1.000 | 1.000 | 239 |

### 7.3 Confusion Matrix Analysis

The confusion matrix reveals near-diagonal performance with extremely few misclassifications. Key observations:

- **Cross-crop confusion is essentially zero.** No pepper images were misclassified as potato or tomato diseases, and vice versa. This confirms that the model learned crop-level morphological features (leaf shape, texture, venation) in addition to disease-specific patterns.
- **The rare misclassifications occur within the same crop.** The small number of errors are concentrated among tomato disease classes, which is biologically reasonable given that some tomato diseases produce visually similar lesion patterns, particularly in early stages.
- **Tomato Early Blight, Tomato Leaf Mold, and Tomato Target Spot** are the three classes with the lowest F1 scores (all 0.990), showing minor confusion among themselves. This is expected: early blight and target spot both produce concentric ring-like lesions on tomato leaves, and leaf mold can present with similar discoloration patterns on the upper leaf surface.
- **Potato Late Blight** (P=0.993) shows a marginal precision reduction, suggesting a very small number of other potato or tomato images were incorrectly predicted as Potato Late Blight. This is consistent with the biological similarity between potato and tomato late blight (both caused by *Phytophthora infestans*).
- **All healthy classes achieved perfect 1.000 across all metrics**, confirming that healthy leaf appearance is visually distinct from diseased tissue regardless of crop type.

### 7.4 Analysis
Key findings from the test set evaluation:

- **Potato Healthy** (only 152 total images, 23 test samples) achieved perfect precision, recall, and F1 of 1.000. This demonstrates that inverse-frequency class weighting in the loss function effectively compensated for the extreme minority status (21:1 imbalance ratio). The model learned robust representations of healthy potato leaves despite having far fewer training examples than any other class.
- **The three lowest-performing classes** are Tomato Early Blight (F1=0.990), Tomato Leaf Mold (F1=0.990), and Tomato Target Spot (F1=0.990) -- all still excellent and all within the tomato crop. The minor confusion among these classes is biologically expected: early blight and target spot produce similar concentric lesion patterns, and leaf mold can overlap visually with early-stage blight symptoms.
- **All healthy classes achieved perfect scores** (F1=1.000 for all four healthy categories), confirming that the model clearly distinguishes healthy tissue from disease regardless of crop type.
- **Cross-crop confusion is essentially zero.** Pepper, potato, and tomato diseases are never confused with each other, validating that the model captures both crop-level morphological features and disease-specific patterns.
- **Macro vs weighted F1 convergence** (0.9963 vs 0.9961) indicates consistent performance across both large and small classes, further confirming that class weighting successfully prevented the model from ignoring minority classes.

---

## 8. Explainability (Grad-CAM)

### 8.1 Method
**Gradient-weighted Class Activation Mapping (Grad-CAM)** computes the importance of each spatial region in the input image for the model's prediction.

The process:
1. Forward pass through the model, capturing activations at the last convolutional layer
2. Backward pass for the predicted class, capturing gradients at the same layer
3. Global average pooling of gradients to get channel-wise importance weights
4. Weighted sum of activation maps, followed by ReLU (only positive contributions)
5. Resize heatmap to input image dimensions and overlay as a jet colormap

### 8.2 Target Layer
The last block of EfficientNetB0's feature extractor (`model.features[-1]`) was chosen because:
- It has the richest semantic information (high-level features like disease patterns)
- It maintains sufficient spatial resolution to localize disease regions
- It is the standard choice for Grad-CAM in EfficientNet architectures

### 8.3 Validation Approach
For each of the 15 classes, 2 correctly-classified test images were selected and their Grad-CAM overlays generated. This produces a gallery of 30 images (side-by-side original and heatmap) that allows visual verification that the model attends to disease-relevant regions.

### 8.4 Key Grad-CAM Insights

Grad-CAM heatmaps across the 30 sample images (2 per class) reveal consistent, biologically meaningful attention patterns:

- **Disease classes:** Heatmaps consistently focus on **lesion areas, spots, and regions of discoloration** on the leaf surface. For bacterial spot classes (Pepper and Tomato), attention concentrates on the dark, water-soaked lesion boundaries. For blight classes, the model highlights the characteristic necrotic tissue and concentric ring patterns. Leaf mold attention centers on the yellowed upper-surface regions corresponding to fungal colonization beneath.
- **Viral disease classes:** For Tomato Yellow Leaf Curl Virus and Tomato Mosaic Virus, heatmaps highlight the characteristic leaf curling/distortion patterns and mosaic color variations respectively, rather than discrete lesions. This is appropriate since viral symptoms are often distributed across the leaf rather than localized.
- **Healthy classes:** Heatmaps for healthy leaves show diffuse, low-intensity activation spread across the leaf surface, with no strong focal points. This indicates the model is confirming the absence of disease indicators rather than fixating on any single region.
- **Background avoidance:** Critically, heatmaps show minimal activation on background regions, pot edges, or non-leaf areas. This confirms the model learned pathological features rather than background artifacts or spurious correlations with image context.

These Grad-CAM results validate that the model's decision-making aligns with domain knowledge from plant pathology -- the model is "looking at the right things" when making predictions.

---

## 9. Model Card

| Field | Details |
|---|---|
| **Model Name** | Crop Disease Detector v1.0 |
| **Model Type** | EfficientNetB0 (transfer learning from ImageNet) |
| **Framework** | PyTorch 2.2.2 |
| **Training Data** | PlantVillage, ~14,447 images (70% split) |
| **Validation Data** | PlantVillage, ~3,096 images (15% split) |
| **Test Data** | PlantVillage, ~3,096 images (15% split) |
| **Input** | 224x224 RGB leaf image, ImageNet normalized |
| **Output** | 15-class probability distribution |
| **Primary Metrics** | Accuracy: 0.9961, Macro F1: 0.9963 |
| **Explainability** | Grad-CAM heatmaps on last convolutional layer |
| **Intended Use** | Leaf disease identification for Pepper, Potato, Tomato |
| **Limitations** | (1) Lab-controlled images only (not field conditions); (2) Limited to 15 classes across 3 crops; (3) Potato Healthy class has only 152 training samples |
| **Ethical Considerations** | Should be used as a decision support tool, not a replacement for professional agronomist diagnosis. False negatives (missed diseases) carry higher risk than false positives. |

---

## 10. Limitations & Real-World Considerations

### 10.1 Lab Dataset vs Field Conditions
The PlantVillage dataset consists of leaf images captured under **controlled laboratory conditions**: consistent lighting, clean backgrounds, centered leaves, and high-contrast disease symptoms. This is fundamentally different from field photography, where images contain variable lighting, overlapping foliage, soil backgrounds, partial occlusion, and early-stage symptoms that are subtler than mature lesions.

The 99.6% test accuracy reflects the model's performance on lab-quality images. Published transfer learning benchmarks on PlantVillage consistently report **95–99.5%+ accuracy** across multiple architectures (ResNet, VGG, Inception, EfficientNet), confirming that this performance level is expected for this dataset rather than an anomaly. In contrast, models trained on lab data and evaluated on field-collected images typically see accuracy drops of **10–25 percentage points** due to domain shift.

### 10.2 Dataset Scope
- **3 crops, 15 classes.** Commercial agriculture involves hundreds of crop species and thousands of disease/pest conditions. This model cannot generalize beyond its training distribution.
- **Single data source.** All images come from PlantVillage. Geographic, cultivar, and environmental diversity is limited.
- **No severity staging.** The model classifies disease presence/absence but does not quantify severity (mild vs. advanced infection), which is critical for treatment decisions.

### 10.3 Small Class Reliability
Potato Healthy achieved perfect F1 (1.000) with only 23 test samples. While this is a strong result, the small test set means that a single misclassification would change F1 from 1.000 to approximately 0.957. Per-class metrics on classes with fewer than 50 test samples should be interpreted with appropriate statistical caution.

### 10.4 Deployment Considerations
Before production deployment, the model would require:
1. **Field validation** on photographs taken in actual growing conditions
2. **Robustness testing** across different camera hardware, lighting, and angles
3. **Calibration analysis** to ensure confidence scores are reliable (a model that says 95% confidence should be correct 95% of the time)
4. **Failure mode analysis** on out-of-distribution inputs (weeds, non-target crops, non-disease damage like hail or mechanical injury)

### 10.5 Honest Framing
The high accuracy on PlantVillage demonstrates that the **technical pipeline works**: two-phase transfer learning, class imbalance handling, and Grad-CAM explainability are all correctly implemented and effective. However, the controlled lab images make this a best-case scenario. The real engineering challenge — and the necessary next step — is bridging the gap from lab to field.

---

## 11. Conclusions

1. **Transfer learning from ImageNet to plant pathology** is effective. EfficientNetB0 achieves 99.61% accuracy on 15 classes despite the significant domain shift from general objects to leaf disease patterns.
2. **Two-phase training** is the correct approach. Phase 1 establishes a good classifier head without disturbing pretrained features, and Phase 2 fine-tunes the last 3 backbone blocks to adapt higher-level features to the plant pathology domain while preserving lower-level feature extractors.
3. **Class imbalance handling is critical.** With a 21:1 ratio between largest and smallest classes, weighted CrossEntropyLoss prevents the model from ignoring minority classes like Potato Healthy and Tomato Mosaic Virus.
4. **Grad-CAM validates model reasoning.** Visual explanations show the model attends to disease-relevant leaf regions rather than background artifacts, building confidence that the model learned meaningful pathological features.
5. **The interactive dashboard** enables non-technical agricultural stakeholders to use the model, verify predictions through Grad-CAM overlays, and access treatment recommendations.

### Recommendations for Production Deployment
- **Field validation required:** The model was trained on lab-controlled images. Performance on field photographs (varying lighting, backgrounds, leaf orientations) must be validated before deployment
- **Expand crop coverage:** The current 3-crop, 15-class model should be expanded to cover additional crops and diseases relevant to the target agricultural region
- **Integrate with mobile platforms:** Package the model for smartphone inference to enable field-level disease identification
- **Continuous monitoring:** Track prediction distribution over time to detect data drift (e.g., new disease variants, different camera hardware)
