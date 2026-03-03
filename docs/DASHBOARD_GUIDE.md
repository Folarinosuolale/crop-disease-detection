# Dashboard Insights and Values Guide

A complete reference for every metric, chart, and value displayed across the five pages of the Crop Disease Detection Dashboard. Written so that anyone, regardless of technical background, can understand what they are seeing and what it means.

---

## Page 1: Overview

### The Five Headline Metrics

These five numbers sit at the top of the dashboard and summarize the model's overall performance at classifying leaf diseases.

**Test Accuracy: 0.9961**
The percentage of test images the model classified correctly out of all test images. An accuracy of 0.9961 means 99.6% of leaf images were diagnosed correctly -- only about 4 out of every 1,000 leaves were misclassified. This is measured on images the model never saw during training, so it reflects real-world performance on new, unseen leaves.

**Macro F1: 0.9963**
The average F1 score across all 15 classes, treating each class equally regardless of how many images it has. A Macro F1 of 0.9963 means the model achieves near-perfect balance between precision and recall across every class, including rare ones. This is important because it prevents the model from "hiding" poor performance on rare classes (like Potato Healthy, which has only 152 images) behind strong performance on common classes (like Tomato Yellow Leaf Curl Virus, which has 3,209). A perfect score is 1.0.

**Weighted F1: 0.9961**
Similar to Macro F1, but weights each class by the number of test images it has. A Weighted F1 of 0.9961 means the model correctly handles 99.6% of cases when accounting for how frequently each disease appears in practice. The Weighted F1 (0.9961) is very close to the Macro F1 (0.9963), indicating the model performs consistently well across both common and rare classes rather than relying on strong performance on common classes alone.

**Macro Precision: 0.9958**
When the model predicts a specific disease, this is how often it is correct, averaged across all 15 classes. A Macro Precision of 0.9958 means that 99.6% of the time, when the model says a leaf has a specific disease, it is right. This translates to very few false alarms -- healthy plants are rarely flagged as diseased, and one disease is almost never misidentified as another.

**Macro Recall: 0.9967**
Of all the images that actually belong to a given disease, this is what percentage the model successfully detected, averaged across all 15 classes. A Macro Recall of 0.9967 means the model catches 99.7% of all disease instances, missing fewer than 3 in every 1,000 affected leaves. In agriculture, recall is critical because a missed disease can spread and cause widespread crop damage -- this near-perfect recall means the model is highly reliable at flagging problems before they go unnoticed.

### Key Insights Panel

**Hardest Class** -- The class with the lowest F1 score, indicating where the model struggles most. This is typically a class with few training images or visual similarity to other classes.

**Easiest Class** -- The class with the highest F1 score, where the model rarely makes mistakes. Diseases with distinctive visual patterns (like Yellow Leaf Curl Virus, which causes dramatic leaf curling) tend to be easiest.

### Pipeline Summary Table

This table shows the seven stages of the machine learning pipeline:

- **Data**: 20,639 leaf images across 15 classes and 3 crops
- **Split**: 70% training, 15% validation, 15% test (stratified to preserve class proportions)
- **Architecture**: EfficientNetB0 pretrained on ImageNet (5.3M parameters)
- **Phase 1**: Classifier head trained with backbone frozen (5 epochs)
- **Phase 2**: Last 3 backbone blocks fine-tuned (up to 10 epochs with early stopping)
- **Evaluation**: Test set metrics including per-class performance
- **Explainability**: Grad-CAM heatmaps for visual prediction explanations

---

## Page 2: Data Explorer

### Tab 1: Class Distribution

**Horizontal Bar Chart**
Shows the number of images per class, grouped by crop (color-coded). The chart makes class imbalance immediately visible:
- The longest bar (Tomato - Yellow Leaf Curl Virus: 3,209 images) is 21x larger than the shortest (Potato - Healthy: 152 images)
- Tomato classes dominate the dataset (10 out of 15 classes)
- Understanding this imbalance is critical because it directly affects model performance on each class

### Tab 2: Sample Images

**Image Grid**
Select a class from the dropdown to view sample leaf images from that class. This helps you visually understand what each disease looks like:
- **Bacterial Spot**: Dark, raised spots on the leaf surface
- **Early Blight**: Concentric ring patterns (target-like lesions)
- **Late Blight**: Large, irregular dark patches, often with water-soaked appearance
- **Leaf Mold**: Yellow patches on upper leaf surface with fuzzy growth underneath
- **Septoria Leaf Spot**: Small circular spots with dark borders and light centers
- **Spider Mites**: Fine yellow stippling across the leaf surface
- **Target Spot**: Brown spots with concentric rings, similar to early blight
- **Yellow Leaf Curl Virus**: Dramatic upward leaf curling and yellowing
- **Mosaic Virus**: Mottled light and dark green patches across the leaf
- **Healthy**: Uniform green coloring without spots or discoloration

### Tab 3: Crop Breakdown

**Pie Chart (left)**
Shows the proportion of images belonging to each crop:
- Tomato: ~77.6% (16,012 images across 10 classes)
- Pepper: ~12.0% (2,475 images across 2 classes)
- Potato: ~10.4% (2,152 images across 3 classes)

**Stacked Bar Chart (right)**
Shows the healthy vs diseased split within each crop:
- Pepper: ~60% Healthy, ~40% Bacterial Spot
- Potato: ~7% Healthy (only 152 images), ~93% Disease
- Tomato: ~10% Healthy, ~90% across 9 disease classes

---

## Page 3: Model Performance

### Tab 1: Overall Metrics

**Metric Cards**
Display four headline metrics (Accuracy, Macro F1, Precision, Recall) in styled metric cards with green accent borders.

### Tab 2: Confusion Matrix

**Heatmap**
A 15x15 grid where each row represents the true class and each column represents what the model predicted. The diagonal cells (top-left to bottom-right) show correct predictions -- darker diagonal cells mean better performance.

**How to read it:**
- Each cell shows the number of test images from the row's true class that were classified as the column's predicted class
- Strong diagonal: The model correctly classifies most images
- Off-diagonal cells: These are misclassifications. Look for patterns -- for example, if many Tomato Early Blight images are classified as Tomato Target Spot, it suggests these diseases look similar to the model

**Top Confusions**
Below the heatmap, the most common misclassification pairs are highlighted. These represent the cases where the model struggles most, and often correspond to diseases with similar visual symptoms.

### Tab 3: Training Curves

**Loss Curves (top)**
Two lines showing how training loss (blue) and validation loss (orange) changed over each epoch:
- Both lines should decrease over time, indicating the model is learning
- If training loss decreases but validation loss increases, the model is overfitting (memorizing training data rather than learning general patterns)
- The **vertical dashed line** marks the boundary between Phase 1 (head only) and Phase 2 (full fine-tuning). You will typically see a jump in learning when Phase 2 begins as the backbone starts adapting

**Accuracy Curves (bottom)**
Two lines showing training accuracy (blue) and validation accuracy (orange):
- Both should increase over time
- The gap between training and validation accuracy indicates how much the model overfits
- A small gap (<5%) means good generalization

### Tab 4: Per-Class Performance

**Grouped Bar Chart**
Three bars per class (Precision, Recall, F1) showing how the model performs on each disease individually:
- Classes where all three bars are tall: The model handles these well
- Classes where Recall is low: The model misses some instances of this disease (false negatives)
- Classes where Precision is low: The model sometimes falsely predicts this disease for other conditions (false positives)

**Styled Table**
A detailed table with exact metric values for each class, color-coded from red (poor) to green (strong). Includes the support count (number of test images per class), which is important context -- a low F1 on a class with only 23 test images is less reliable than a low F1 on a class with 480.

---

## Page 4: Grad-CAM Explanations

### Gallery

Select a class from the dropdown (or "All Classes") to view Grad-CAM samples. Each image shows a side-by-side comparison:

**Left: Original Image**
The unmodified leaf photograph as the model receives it.

**Right: Grad-CAM Overlay**
The same image with a heatmap overlay:
- **Red/warm areas**: Regions the model pays most attention to when making its prediction. These should correspond to disease symptoms (spots, lesions, discoloration)
- **Blue/cool areas**: Regions the model largely ignores. Background and healthy tissue typically appear blue
- **Confidence percentage**: How certain the model is about its prediction

### What to Look For

**Good Grad-CAM behavior:**
- Heatmap highlights the diseased region (spots, lesions, curling)
- Healthy tissue and background remain blue/cool
- Consistent focus across multiple images of the same disease

**Concerning Grad-CAM behavior:**
- Heatmap highlights the image border or background instead of the leaf
- Heatmap spreads evenly across the entire image without focusing on any region
- Different images of the same disease show inconsistent focus areas

---

## Page 5: Live Prediction

### The Upload Area
Drag and drop or click to upload a leaf photograph (JPG, JPEG, or PNG). The image will be resized to 224x224 pixels and preprocessed before inference.

### The Prediction Result

**Disease Name and Confidence**
The model's top prediction with its confidence score (0-100%). A higher confidence means the model is more certain. Predictions below 50% confidence should be treated with caution and verified by an expert.

**Grad-CAM Overlay**
A heatmap overlay on your uploaded image showing which leaf regions the model focused on. Use this to verify that the model is looking at the right areas:
- For a bacterial spot diagnosis, the heatmap should highlight the spots
- For a healthy prediction, the heatmap may highlight the overall leaf structure
- If the heatmap focuses on the background instead of the leaf, the prediction is less trustworthy

### Top-5 Predictions Bar Chart
Shows the model's top 5 predicted classes with their confidence scores. Useful for understanding alternative diagnoses:
- If the top prediction has 90% confidence and the second has 5%, the model is very certain
- If the top two predictions are close (e.g., 40% and 35%), the model is uncertain and both diagnoses should be considered

### Disease Information Panel
For disease predictions, this panel provides:
- **Severity level** (Low, Moderate, High)
- **Description** of the disease and its symptoms
- **Recommended action** including treatment options and preventive measures

For healthy predictions, the panel confirms the plant appears disease-free and recommends continued monitoring.
