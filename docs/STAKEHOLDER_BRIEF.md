# Crop Disease Detection - Stakeholder Brief

## What Is This?

We built an intelligent system that can identify crop diseases from photographs of leaves. A farmer or agricultural officer takes a picture of a suspicious-looking leaf, uploads it, and within seconds receives a diagnosis: which disease it is (or whether the plant is healthy), how confident the system is, and a visual explanation showing exactly which part of the leaf triggered the diagnosis. The system also provides a recommended treatment action for each identified disease.

---

## The Problem We Solved

When a crop starts showing signs of disease, farmers face a critical question: **"What is wrong with my plants, and what should I do about it?"**

Getting this wrong is costly in both directions:

- **Misidentify the disease:** The farmer applies the wrong treatment, wastes money on pesticide, and the disease continues to spread. A bacterial spot treated with a fungicide will not improve.
- **Miss the disease entirely:** The farmer assumes the crop is healthy. By the time symptoms become obvious, the outbreak has spread to neighboring plants and the window for effective treatment has closed.

Traditionally, disease identification requires either a trained agronomist visiting the field in person, or the farmer sending leaf samples to a laboratory. Both are slow and expensive. In many agricultural regions, especially in the developing world, neither option is available at all.

Our system provides instant, visual, explainable diagnosis from a single leaf photograph.

---

## What We Built

### The Model

We trained a deep learning model on 20,639 photographs of crop leaves spanning 3 crops and 15 categories (12 diseases and 3 healthy plant types). The model learned to recognize the visual patterns that distinguish each disease: the spots, discoloration, curling, mold, and lesion patterns that characterize each condition.

The 15 categories it can identify:

| Crop | Conditions |
|---|---|
| **Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

### How Good Is It?

The model achieves **99.6% accuracy** on a test set of images it was never trained on. This means that 99.6% of the time, it correctly identifies the exact disease (or correctly identifies a healthy plant).

All 15 classes achieved an F1 score of 0.990 or higher, with a macro-averaged F1 of 0.9963. The lowest-performing classes -- Tomato Early Blight, Tomato Leaf Mold, and Tomato Target Spot -- still achieved F1 scores of 0.990. Notably, Potato Healthy, the smallest class with only 152 images, achieved a perfect F1 of 1.000.

For context:
- A random guess among 15 categories would be correct about 7% of the time
- Even trained agricultural students typically achieve 50-70% accuracy on leaf disease identification in controlled studies
- The model performs exceptionally well across all disease categories, including diseases with similar early-stage symptoms (like Early Blight and Target Spot, both at 0.990 F1)

**Important caveat:** These results are on laboratory-quality images with controlled lighting and clean backgrounds. Published research consistently reports 95-99%+ accuracy on this same dataset, so the high performance is expected. In real-world field conditions -- variable lighting, overlapping plants, early-stage symptoms -- accuracy would be lower. Field validation is a critical next step before deployment (see Recommendations below).

### The Visual Explanation

What makes this system different from a simple "black box" classifier is that it shows its reasoning. For every prediction, the system generates a **heatmap overlay** on the original leaf image:
- **Red/warm areas** highlight where the model is focusing most of its attention
- **Blue/cool areas** are regions the model considers less important for this diagnosis

This allows an agronomist or experienced farmer to verify whether the model is looking at the right part of the leaf. If the heatmap highlights the disease lesions, the prediction is trustworthy. If it highlights the background or healthy tissue, the prediction should be treated with more skepticism.

### The Dashboard

We built an interactive web application where anyone on the team can:

- **Explore the data** -- See how many images exist per disease, view sample leaf photographs, and understand the distribution across crops
- **Review model performance** -- See the confusion matrix showing where the model excels and where it struggles, along with training curves showing how the model improved over time
- **Browse Grad-CAM explanations** -- View a gallery of heatmap overlays for every disease category
- **Test live predictions** -- Upload any leaf photograph and get an instant diagnosis with visual explanation and treatment recommendation

---

## What We Discovered

### Key Findings

1. **The model learns the right visual features.** Grad-CAM heatmaps consistently highlight disease-relevant regions (spots, lesions, discoloration) rather than leaf edges, stems, or background, confirming the model is learning real pathological patterns rather than shortcuts.

2. **Class imbalance reflects real-world prevalence.** The dataset has 3,209 images of Tomato Yellow Leaf Curl Virus but only 152 of Potato Healthy. This mirrors reality: common diseases are better documented. We handled this by making the model pay more attention to rare classes during training.

3. **Some diseases are visually similar.** The model may occasionally confuse diseases that look alike in early stages. This is biologically expected and occurs even among human experts. The confidence score helps flag uncertain predictions.

4. **Transfer learning is powerful.** Rather than training from scratch, we started with a model that already understood general visual concepts (edges, textures, colors) from millions of natural images, then taught it to apply those concepts to leaf pathology. This is like teaching a photography expert to identify plant diseases -- they already understand what to look for visually, they just need to learn the specific disease patterns.

---

## Why This Matters for Agricultural Operations

### For Extension Services
Agricultural extension officers cover large areas with many farmers. This system allows them to:
- Quickly triage disease reports by uploading leaf photos
- Provide consistent diagnoses regardless of which officer responds
- Build a record of disease occurrences with visual evidence

### For Farmers
- **Immediate diagnosis** instead of waiting days for an expert visit
- **Treatment recommendations** alongside each diagnosis
- **Visual proof** of the disease for insurance or assistance claims

### For Research Organizations
- **Rapid screening** of field samples during disease surveillance
- **Standardized classification** that eliminates inter-observer variability
- **Training tool** for agricultural students learning disease identification

### Economic Impact
- Early detection of Late Blight in potatoes can prevent 50-70% yield losses
- Targeted treatment (applying the right pesticide for the specific disease) reduces chemical costs and environmental impact
- Automated screening reduces the cost per diagnosis from expert consultation rates to essentially zero marginal cost

---

## What We Recommend Next

1. **Field validation** -- Test the model on leaf photographs taken in actual field conditions (varying lighting, angles, backgrounds) to assess real-world accuracy
2. **Pilot deployment** -- Package the model into a mobile app for a small group of extension officers to test over one growing season
3. **Expand coverage** -- Add more crops and diseases relevant to the target region
4. **Feedback loop** -- Collect correction data when the model is wrong to continuously improve accuracy
5. **Integration** -- Connect the disease identification system with existing agricultural advisory platforms

---

## Technical Details (For Those Who Want Them)

| Item | Detail |
|---|---|
| Architecture | EfficientNetB0 (5.3M parameters, pretrained on ImageNet) |
| Training approach | Two-phase transfer learning (head only, then fine-tuning last 3 backbone blocks) |
| Training data | 20,639 leaf photographs across 15 classes |
| Test accuracy | 0.9961 (99.6%) |
| Test macro F1 | 0.9963 |
| Explainability | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| Class imbalance | Inverse-frequency weighted CrossEntropyLoss (21:1 ratio) |
| Dashboard | Streamlit web application with 5 interactive pages |
| Framework | PyTorch 2.2.2, torchvision, OpenCV, Plotly, Streamlit |
| Compute | Supports Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU |

---

## Questions?

This document provides a high-level overview. For full technical details including model architecture, training procedure, per-class performance, and Grad-CAM analysis, refer to the **Technical Report** (PROJECT_REPORT.md).
