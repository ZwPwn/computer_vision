# Assignment 2 — Panoramic Image Stitching

## Overview

This assignment implements a full panoramic image stitching pipeline using feature detection (SIFT, SURF, ORB), homography estimation, and multiple stitching strategies (planar, cylindrical, hybrid).

## Topics Covered

- **Feature Detection & Matching** — SIFT, SURF, and ORB keypoint detection with BFMatcher
- **Homography Estimation** — RANSAC-based homography computation
- **Planar Stitching** — Direct homography-based warping
- **Cylindrical Stitching** — Cylindrical projection for wide-angle panoramas
- **Hybrid Stitching** — Combined cylindrical projection + planar refinement
- **Camera Calibration** — Using calibration parameters for projection

## Files

| File | Description |
|------|-------------|
| `main.py` | Main entry point — runs stitching pipeline on multiple datasets |
| `camera_params.yaml` | Camera intrinsic parameters |
| `src/answers/bfmatcher.py` | Feature matching and homography estimation |
| `src/answers/image_operations.py` | Image warping and transformation utilities |
| `src/answers/stitch/stitch_planar.py` | Planar stitching implementation |
| `src/answers/stitch/stitch_cylindrical.py` | Cylindrical projection stitching |
| `src/answers/stitch/stitch_hybrid.py` | Hybrid stitching (cylindrical + planar) |
| `src/helper/io.py` | Image I/O utilities |
| `src/helper/calibration_main.py` | Camera calibration routines |

## Usage

```bash
python main.py
```

Configure dataset paths and stitching method in `main.py`.

## Results

See [Report.pdf](Report.pdf) for detailed comparisons across algorithms and datasets.

Output panoramas are saved to [`out/`](out/) organized by dataset and algorithm.
