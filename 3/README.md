# Assignment 3 — Image Classification (Bag of Visual Words)

## Overview

This assignment implements image classification using the Bag of Visual Words (BoVW) model with SIFT descriptors, evaluated on two datasets: Caltech-Transportation and GTSRB (German Traffic Sign Recognition Benchmark).

## Topics Covered

- **SIFT Feature Extraction** — Local feature descriptors for visual vocabulary
- **K-Means Clustering** — Building visual vocabularies of varying sizes (50–10000)
- **Bag of Visual Words (BoVW)** — Histogram-based image representation
- **SVM Classification** — Support Vector Machines for multi-class classification
- **KNN Classification** — K-Nearest Neighbors with various K values
- **Dataset Comparison** — Performance on transportation vs. traffic sign datasets

## Files

| File | Description |
|------|-------------|
| `A.py` | Main pipeline — feature extraction, vocabulary building, SVM/KNN classification |
| `B_caltech.ipynb` | Jupyter notebook with analysis on Caltech-Transportation dataset |
| `B_gtsrb.ipynb` | Jupyter notebook with analysis on GTSRB dataset |
| `Dataset 1/` | Caltech-Transportation train/test images |
| `Dataset 2/` | GTSRB train/test images |
| `Data/` | Pre-computed features and vocabulary files (.npy) |

## Usage

```bash
# Run the full classification pipeline
python A.py

# Or open the Jupyter notebooks for interactive analysis
jupyter notebook B_caltech.ipynb
jupyter notebook B_gtsrb.ipynb
```

## Configuration

In `A.py`, modify these variables to control behavior:

```python
DATASET_CHOICE = 1          # 1 = Caltech-Transportation, 2 = GTSRB
VOCABULARY_SIZES = [50, 100, 200, 400, 800]
K_VALUES = [1, 3, 5, 7, 9]  # For KNN
```

## Results

See [Report.pdf](Report.pdf) for accuracy tables, confusion matrices, and analysis.
