# Assignment 1 — Image Processing & Noise Removal

## Overview

This assignment implements fundamental image processing techniques from scratch, including custom filtering, noise removal, integral images, and text segmentation via projection profiling.

## Topics Covered

- **Custom Median Filter** — Sliding window median computation (no OpenCV)
- **Gaussian Filtering** — Noise smoothing with various kernel sizes
- **Salt & Pepper Noise** — Addition and removal using median filters
- **Integral Images** — Efficient area-sum computation
- **Projection Profiling** — Horizontal and vertical projection for text line detection
- **Text Segmentation** — Connected components + morphological operations to isolate text regions

## Files

| File | Description |
|------|-------------|
| `generate.py` | Generates Gaussian and Salt & Pepper noisy versions of the input image |
| `noise_free.py` | Text segmentation on a clean image using integral images, morphology, and projection profiling |
| `noise_gaussian.py` | Gaussian noise analysis — filtering with different kernel sizes |
| `noise_salt_and_pepper.py` | Salt & Pepper noise analysis — median filtering comparisons |
| `Src/filters.py` | Custom median and Gaussian filter implementations |
| `Src/integral.py` | Integral image computation |
| `Src/noise.py` | Noise generation functions |
| `Src/projection_profiling.py` | Projection profiling for text segmentation |
| `Src/helper_functions.py` | Display and visualization utilities |

## Usage

```bash
python generate.py               # Step 1: Generate noisy images
python noise_free.py              # Text segmentation on clean image
python noise_gaussian.py          # Gaussian noise removal analysis
python noise_salt_and_pepper.py   # Salt & Pepper noise removal analysis
```

## Results

See [Report.pdf](Report.pdf) for detailed analysis and figures.

### Sample Figures

| Gaussian Filtering Comparison | Salt & Pepper Removal |
|---|---|
| ![Gaussian](Report/gaussian/compare_kernel_0.png) | ![S&P](Report/salt_n_pepper/multiple_kernels.png) |

| Projection Profiles | Text Segmentation |
|---|---|
| ![Projection](Report/projection/Horizontal_Vertical_Profiles.png) | ![Segmentation](Report/text_segmentation/regions.png) |
