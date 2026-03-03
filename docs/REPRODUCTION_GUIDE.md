# Reproduction Guide - Crop Disease Detection

A step-by-step walkthrough to reproduce this project from scratch on your own machine.

---

## Prerequisites

Before you begin, make sure you have:

- **Python 3.10 or later** installed on your system
- **pip** (Python package manager, comes with Python)
- **Git** (optional, for cloning the repository)
- A terminal or command prompt
- About **4 GB of free disk space** (for packages, dataset, and model artifacts)

To check your Python version, open a terminal and run:
```bash
python3 --version
```

If you see something like `Python 3.10.x` or higher, you are good to go.

---

## Step 1: Get the Project Files

**Option A: Clone from Git (if hosted in a repository)**
```bash
git clone <repository-url>
cd crop-disease-detection
```

**Option B: Copy the folder manually**
If you received the project as a zip file or folder, extract it and navigate to the project root:
```bash
cd /path/to/crop-disease-detection
```

You should see the following structure:
```
crop-disease-detection/
    app/
    assets/
    data/
    docs/
    models/
    src/
    requirements.txt
    README.md
```

---

## Step 2: Create a Virtual Environment (Recommended)

It is best practice to isolate project dependencies so they do not conflict with other Python projects on your machine.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the beginning of your terminal prompt, confirming the virtual environment is active.

---

## Step 3: Install Dependencies

All required Python packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

This will install the following (and their sub-dependencies):
- **torch, torchvision** -- Deep learning framework and pretrained models
- **numpy, pandas** -- Data manipulation
- **scikit-learn** -- Metrics and data splitting
- **matplotlib, seaborn, plotly** -- Visualization
- **streamlit** -- Web dashboard framework
- **Pillow** -- Image loading and processing
- **opencv-python** -- Image processing (Grad-CAM overlays)
- **tqdm** -- Progress bars during training
- **joblib** -- Model artifact serialization

Installation typically takes 3-10 minutes depending on your internet speed. PyTorch is the largest package (~700 MB).

**Note for Apple Silicon (M1/M2/M3) users:** PyTorch 2.2+ includes native MPS (Metal Performance Shaders) support. The pipeline will automatically detect and use your GPU for training. No additional setup needed.

---

## Step 4: Verify the Dataset

The PlantVillage dataset should already be in the `data/PlantVillage/` folder with 15 class subfolders:

```bash
ls data/PlantVillage/
```

You should see 15 folders:
```
Pepper__bell___Bacterial_spot/
Pepper__bell___healthy/
Potato___Early_blight/
Potato___Late_blight/
Potato___healthy/
Tomato_Bacterial_spot/
Tomato_Early_blight/
Tomato_Late_blight/
Tomato_Leaf_Mold/
Tomato_Septoria_leaf_spot/
Tomato_Spider_mites_Two_spotted_spider_mite/
Tomato__Target_Spot/
Tomato__Tomato_YellowLeaf__Curl_Virus/
Tomato__Tomato_mosaic_virus/
Tomato_healthy/
```

The total should be 20,639 images across all folders. Each folder contains JPG images of leaves for that disease class.

If the dataset is missing, you will need to download it from the PlantVillage repository and place the 15 class folders inside `data/PlantVillage/`.

---

## Step 5: Run the Full Pipeline

This is the main step. It executes the entire deep learning workflow:

```bash
python src/run_pipeline.py
```

**What happens during this step:**

1. **[1/7] Data Loading** (a few seconds) -- Scans the dataset directory, collects all image paths and labels, creates a stratified 70/15/15 train/val/test split, sets up DataLoaders with augmentation for training.

2. **[2/7] Model Building** (a few seconds) -- Downloads EfficientNetB0 pretrained weights from ImageNet (first run only, ~20 MB), replaces the classifier head with a 15-class output layer, freezes the backbone.

3. **[3/7] Phase 1: Head Training** (2-5 minutes) -- Trains only the classifier head for 5 epochs with the backbone frozen. This is fast because only ~19,000 parameters are being updated.

4. **[4/7] Phase 2: Partial Fine-Tuning** (15-50 minutes) -- Unfreezes the last 3 backbone blocks and fine-tunes with a 10x smaller learning rate for up to 10 epochs. Early stopping may terminate this phase sooner if validation loss plateaus. **This is the longest step.** Training time depends heavily on your hardware:
   - Apple Silicon (MPS): ~15-25 minutes
   - NVIDIA GPU (CUDA): ~10-15 minutes
   - CPU only: ~30-50 minutes

5. **[5/7] Evaluation** (about 30 seconds) -- Evaluates the best model on the held-out test set. Computes accuracy, macro/weighted precision, recall, F1, per-class metrics, and confusion matrix.

6. **[6/7] Grad-CAM Generation** (1-2 minutes) -- Generates Grad-CAM heatmap overlays for 2 correctly-classified samples per class (30 images total). These are saved as side-by-side PNG images.

7. **[7/7] Saving Artifacts** (a few seconds) -- Saves all model weights, metadata, metrics, and training history to the `models/` directory.

**Expected total runtime:** 20-65 minutes depending on hardware.

**Expected output:** The terminal will print progress for each step including per-epoch training loss and accuracy, followed by test set metrics and artifact save confirmations. At the end you should see `Pipeline complete!`

**Files generated in `models/`:**
- `best_model.pt` -- Trained model weights (PyTorch state_dict)
- `artifacts.pkl` -- Class names, display names, crop mappings, model config
- `pipeline_results.json` -- All metrics, dataset stats, training config
- `confusion_matrix.json` -- Full 15x15 confusion matrix
- `per_class_metrics.csv` -- Per-class precision, recall, F1, support
- `training_history.json` -- Per-epoch loss, accuracy, learning rate curves

**Files generated in `assets/gradcam_samples/`:**
- 30 PNG images (2 per class) showing side-by-side original and Grad-CAM overlay

---

## Step 6: Launch the Dashboard

Start the Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

Your terminal will display:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open that URL in your web browser. You will see the interactive dashboard with five pages accessible from the sidebar:

1. **Overview** -- Summary metrics and key insights
2. **Data Explorer** -- Class distribution charts, sample images, crop breakdown
3. **Model Performance** -- Confusion matrix, training curves, per-class metrics
4. **Grad-CAM Explanations** -- Gallery of Grad-CAM samples for each class
5. **Live Prediction** -- Upload a leaf image for instant diagnosis with Grad-CAM

To stop the dashboard, press `Ctrl+C` in the terminal.

---

## Step 7: Explore and Modify (Optional)

### Changing Training Duration

To speed up or extend training, edit `src/run_pipeline.py` and change the epoch parameters:

```python
# Faster (less accurate)
run_full_pipeline(phase1_epochs=3, phase2_epochs=5, patience=3)

# More thorough (takes longer)
run_full_pipeline(phase1_epochs=10, phase2_epochs=20, patience=7)
```

### Changing Batch Size

If you run out of memory during training, reduce the batch size:

```python
run_full_pipeline(batch_size=16)  # Default is 32
```

### Using a Different Dataset

To use your own leaf image dataset:
1. Organize images into subfolders by class (one folder per class)
2. Update the `CLASS_NAMES`, `DISPLAY_NAMES`, `CROP_MAP`, and `IS_HEALTHY` constants in `src/data_loader.py`
3. Update the `DISEASE_INFO` dictionary in `app/streamlit_app.py`
4. Re-run the pipeline

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"
You are likely not in the virtual environment. Run `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows) and try again.

### "FileNotFoundError" or "class folder not found"
Make sure you are running commands from the project root directory (the folder containing `requirements.txt`). The pipeline expects `data/PlantVillage/` to contain the 15 class folders.

### Out of memory during training
Reduce the batch size: edit `src/run_pipeline.py` and change `batch_size=32` to `batch_size=16` or `batch_size=8`.

### Streamlit shows a blank page or error
Make sure the pipeline completed successfully first (Step 5). The dashboard depends on artifacts that the pipeline generates in `models/` and `assets/`.

### MPS (Apple Silicon) errors
If you see MPS-related errors, the pipeline will automatically fall back to CPU. You can also force CPU by modifying `src/run_pipeline.py` to set `device = torch.device('cpu')`.

### Training seems stuck
Phase 2 fine-tuning can take 15-60 minutes depending on hardware. Check that per-epoch progress is still printing. If validation loss has not improved for several epochs, early stopping will terminate training automatically.

---

## File Reference

| File | Purpose |
|---|---|
| `src/data_loader.py` | Scans dataset, creates splits, defines augmentation, builds DataLoaders |
| `src/model_training.py` | EfficientNetB0 architecture, two-phase training, evaluation metrics |
| `src/explainability.py` | Grad-CAM implementation, heatmap overlay, sample generation |
| `src/run_pipeline.py` | Orchestrates the full 7-step end-to-end pipeline |
| `app/streamlit_app.py` | Interactive web dashboard (5 pages) |
| `requirements.txt` | Python package dependencies |
