# Crop Disease Detection with Deep Learning

**Deep Learning | Computer Vision | Transfer Learning | Visual Explainability**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNetB0-green)
![Grad-CAM](https://img.shields.io/badge/Explainability-GradCAM-purple)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

---

## Overview

A deep learning system that identifies crop diseases from leaf images using transfer learning on EfficientNetB0. The model classifies leaf images into 15 categories across 3 crops (Pepper, Potato, Tomato), distinguishing between healthy plants and specific diseases. The project includes **Grad-CAM visual explanations** that highlight which regions of a leaf image the model focuses on when making its diagnosis, making the model's reasoning transparent and interpretable for agricultural stakeholders.

Plant diseases cause up to 40% of global crop losses annually, and early detection is the single most effective intervention. A trained agronomist can identify most diseases by sight, but there are not enough agronomists to cover every field. This project demonstrates that a convolutional neural network trained on leaf images can replicate that visual diagnosis with high accuracy and, critically, can show exactly what it is looking at when it makes its call.

### Why This Matters

Agricultural disease detection directly impacts food security and farmer livelihoods:
- **Late detection** allows diseases to spread across entire fields, turning a treatable outbreak into a total loss
- **Misidentification** leads to wrong treatments, wasted pesticide, and continued crop damage
- **Access gap** -- smallholder farmers in developing regions rarely have access to plant pathologists

This project demonstrates building a model that is not just accurate, but **visually explainable** through Grad-CAM heatmaps.

---

## Key Results

| Metric | Value |
|---|---|
| **Test Accuracy** | 0.996 |
| **Macro F1** | 0.996 |
| **Weighted F1** | 0.996 |
| **Macro Precision** | 0.996 |
| **Macro Recall** | 0.997 |

EfficientNetB0 was selected for its balance of accuracy and efficiency -- 5.3M parameters, fast enough to train on CPU or Apple Silicon, and strong transfer learning performance from ImageNet pretraining.

> **Note on accuracy:** These results are on the PlantVillage lab dataset, where controlled lighting and clean backgrounds make disease patterns highly distinguishable. Published benchmarks consistently report 95–99%+ accuracy on this dataset across multiple architectures. Real-world field deployment would require validation on in-situ photographs, where accuracy is expected to be lower due to variable lighting, overlapping foliage, and early-stage symptoms. See the [Technical Report](docs/PROJECT_REPORT.md#10-limitations--real-world-considerations) for a full limitations analysis.

### Key Findings
1. **Transfer learning works** -- ImageNet features transfer effectively to leaf disease classification despite the significant domain shift from general objects to plant pathology
2. **Class imbalance matters** -- the dataset ranges from 152 images (Potato Healthy) to 3,209 images (Tomato Yellow Leaf Curl Virus), a 21:1 ratio. Weighted loss was critical for balanced performance
3. **Grad-CAM confirms the model looks at lesions** -- heatmaps consistently highlight diseased tissue regions rather than background or healthy leaf areas, validating that the model learned meaningful visual features

---

## Dataset

**PlantVillage** (Penn State University)

- **Images:** 20,639 leaf photographs
- **Classes:** 15 (3 healthy + 12 disease categories)
- **Crops:** 3 (Pepper, Potato, Tomato)
- **Format:** RGB images, varying resolutions, resized to 224x224
- **Source:** PlantVillage open-access repository

### Class Distribution

| Crop | Class | Images |
|---|---|---|
| Pepper | Bacterial Spot | 997 |
| Pepper | Healthy | 1,478 |
| Potato | Early Blight | 1,000 |
| Potato | Late Blight | 1,000 |
| Potato | Healthy | 152 |
| Tomato | Bacterial Spot | 2,127 |
| Tomato | Early Blight | 1,000 |
| Tomato | Late Blight | 1,909 |
| Tomato | Leaf Mold | 952 |
| Tomato | Septoria Leaf Spot | 1,771 |
| Tomato | Spider Mites | 1,676 |
| Tomato | Target Spot | 1,404 |
| Tomato | Yellow Leaf Curl Virus | 3,209 |
| Tomato | Mosaic Virus | 373 |
| Tomato | Healthy | 1,591 |

### Why This Dataset

PlantVillage is the standard benchmark for plant disease classification:
- Widely used in agricultural AI research with hundreds of published papers
- Controlled lab conditions provide clean ground truth labels
- Multi-crop coverage enables cross-crop generalization analysis
- Significant class imbalance reflects real-world disease prevalence patterns
- Large enough to train deep learning models, small enough for rapid iteration

---

## Technical Approach

### 1. Data Preparation
- Stratified 70/15/15 split (train/validation/test) preserving class proportions
- Train augmentation: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15), ColorJitter, RandomAffine
- Validation/test: Resize to 224x224 + ImageNet normalization only
- Inverse-frequency class weights for weighted CrossEntropyLoss

### 2. Model Architecture
- **Base:** EfficientNetB0 pretrained on ImageNet (5.3M parameters)
- **Classifier head:** Dropout(0.3) + Linear(1280, 15)
- **Input:** 224x224 RGB images normalized to ImageNet statistics

### 3. Two-Phase Transfer Learning

| Phase | What Trains | Learning Rate | Epochs | Purpose |
|---|---|---|---|---|
| **Phase 1** | Classifier head only | 1e-3 | 5 | Learn disease-specific features without disturbing pretrained backbone |
| **Phase 2** | Last 3 backbone blocks + head | 1e-4 | Up to 10 | Fine-tune higher-level features for leaf pathology domain |

- **Optimizer:** AdamW (weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Early stopping:** Patience of 5 epochs on validation loss
- **Gradient clipping:** max_norm=1.0

### 4. Evaluation
- Overall: Accuracy, Macro/Weighted Precision, Recall, F1
- Per-class: Precision, Recall, F1 for all 15 classes
- Confusion matrix to identify systematic misclassifications

### 5. Explainability (Grad-CAM)
- **Method:** Gradient-weighted Class Activation Mapping on the last convolutional layer
- **Output:** Heatmap overlays showing which spatial regions of the leaf image contributed most to the prediction
- **Validation:** 2 correctly-classified samples per class saved as side-by-side original/heatmap images
- **Use case:** Builds trust by showing the model focuses on diseased tissue rather than background artifacts

---

## Interactive Dashboard

The Streamlit app includes 5 pages:

1. **Overview** -- Key metrics, pipeline summary, insights
2. **Data Explorer** -- Class distribution charts, sample images, crop breakdown
3. **Model Performance** -- Confusion matrix heatmap, training curves, per-class metrics
4. **Grad-CAM Explanations** -- Gallery of Grad-CAM samples by class
5. **Live Prediction** -- Upload a leaf image, get disease prediction + confidence + Grad-CAM overlay + treatment recommendation

### Run the Dashboard

```bash
cd crop-disease-detection
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Real-World Relevance

### Industry Application
This project mirrors workflows used in precision agriculture and agri-tech:
- **Field scouting:** Automated disease detection from drone or smartphone imagery
- **Early warning systems:** Identifying outbreaks before they spread across fields
- **Extension services:** Providing disease identification to farmers without access to agronomists
- **Crop insurance:** Documenting and verifying disease claims with visual evidence

### Agricultural Impact
- Plant diseases cause estimated **$220 billion in annual losses** globally (FAO)
- Early detection can reduce crop losses by **50-80%** through timely intervention
- Smallholder farmers (who produce 80% of food in developing countries) benefit most from accessible diagnostic tools
- Visual explainability (Grad-CAM) enables agronomists to **verify** model predictions rather than blindly trust them

### Technical Transferability
- The two-phase transfer learning approach applies to any domain-specific image classification task
- Grad-CAM explainability is applicable to any CNN-based model
- The pipeline architecture (data loading, training, evaluation, explainability, dashboard) is a reusable template

---

## How to Reproduce

```bash
# 1. Clone and install
git clone <repository-url>
cd crop-disease-detection
pip install -r requirements.txt

# 2. Run the full pipeline (trains model, generates all artifacts)
python src/run_pipeline.py

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch, torchvision |
| **Architecture** | EfficientNetB0 (transfer learning from ImageNet) |
| **Explainability** | Grad-CAM (custom hook-based implementation) |
| **Image Processing** | Pillow, OpenCV |
| **Metrics** | scikit-learn |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **App Framework** | Streamlit |
| **Data** | Pandas, NumPy |

---
