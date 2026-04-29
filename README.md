# Optic Disc & Cup Segmentation using Connected Component Labeling

**Course:** EC 312 – Digital Image Processing  
**Assignment:** #01 — CCL-based Segmentation  
**Author:** Ibrahim Abdullah (454188) | CE 45 A, NUST

---

## Overview

This project implements a classical image processing pipeline to segment the **Optic Disc (OD)** and **Optic Cup (OC)** from retinal fundus images in the [Drishti-GS dataset](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php). Segmentation is performed without any deep learning — using thresholding, morphological operations, and a manually implemented 8-connectivity Connected Component Labeling (CCL) algorithm.

Performance is evaluated using the **Dice Coefficient** for both the Disc and Cup classes.

---

## Results

| Experiment  | Disc Percentile | Cup Percentile | Avg Disc Dice | Avg Cup Dice |
|-------------|-----------------|----------------|---------------|--------------|
| Final Model | 92              | 75             | 0.5923        | 0.2801       |

---

## Pipeline

```
Green Channel Extraction
        ↓
CLAHE Contrast Enhancement
        ↓
Median Blur (kernel=21)
        ↓
Circular ROI Mask
        ↓
92nd Percentile Threshold → Binary Mask
        ↓
Morphological Opening
        ↓
Manual 8-Conn CCL → Largest Component = Optic Disc
        ↓
75th Percentile (within Disc) → Binary Mask
        ↓
Morphological Opening
        ↓
Manual 8-Conn CCL → Largest Component = Optic Cup
        ↓
Dice Coefficient vs. Ground Truth
```

---

## Project Structure

```
project/
├── main.py                   # Full segmentation pipeline
├── Results/
│   ├── Masks/                # Saved predicted masks (.png)
│   ├── Comparisons/          # Side-by-side comparison plots
│   └── final_scores.csv      # Per-image Dice scores
└── README.md
```

---

## Requirements

```
python >= 3.8
opencv-python
numpy
pandas
matplotlib
```

Install dependencies:

```bash
pip install opencv-python numpy pandas matplotlib
```

---

## Dataset Setup

This project uses the **Drishti-GS** dataset. Expected directory structure:

```
Drishti-GS/
└── Test/
    ├── Images/
    │   └── drishtiGS_001.png ...
    └── Test_GT/
        └── drishtiGS_001/
            └── SoftMap/
                ├── drishtiGS_001_ODsegSoftmap.png
                └── drishtiGS_001_cupsegSoftmap.png
```

Update the `dataset_root` path in `main.py` to point to your local copy:

```python
dataset_root = r"path/to/Drishti-GS/Test"
```

---

## Usage

```bash
python main.py
```

Output is printed to the console and saved under `Results/`:

```
Processing 51 images...
Processed drishtiGS_003 | Disc: 0.8836 | Cup: 0.2181
...
==============================
Average Disc Dice: 0.5923
Average Cup Dice:  0.2801
==============================
```

---

## Key Design Choices

**Green channel** — provides the highest contrast for retinal structures compared to red or blue channels.

**CLAHE** — adaptive histogram equalization improves local contrast without over-amplifying noise, making thresholds more reliable across varying image qualities.

**Circular ROI mask** — removes bright border artefacts common in fundus photography, preventing them from skewing percentile thresholds.

**Percentile thresholding (V-set)** — adaptive to each image's intensity distribution, making it more robust than a fixed global threshold.

**Manual 8-connectivity CCL** — implemented from scratch using a two-pass union-find approach, satisfying the assignment requirement and providing full control over the labeling logic.

---

## Limitations

- Cup Dice (0.28) is considerably lower than Disc Dice (0.59) due to the cup's small size and low contrast with the surrounding disc tissue.
- Performance is sensitive to image quality and anatomical variation across patients.
- The pipeline does not generalize well to images with strong artefacts or unusual disc positions.
