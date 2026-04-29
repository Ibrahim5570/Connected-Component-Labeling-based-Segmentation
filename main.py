# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import cv2  
# OpenCV library. Used for reading images from disk, applying filters, thresholding,
# morphological operations, finding contours, and general computer vision tasks.

import numpy as np  
# NumPy library. Handles multidimensional arrays (images are arrays of pixel values),
# allows efficient mathematical operations on images.

import os  
# Provides functions to interact with the operating system.
# We use it to check if files/folders exist, and to create directories if they don't.

import pandas as pd  
# Pandas library. Useful for storing tabular data.
# Here it stores Dice scores for each image and allows saving as CSV.

import matplotlib.pyplot as plt  
# Matplotlib library. Used for displaying images, plotting, and saving figures.

from pathlib import Path  
# Provides an object-oriented approach for handling file paths.
# Works across operating systems, better than raw strings.


# ==========================================================
# CONNECTED COMPONENT LABELING (8-CONNECTIVITY)
# ==========================================================

# Function to find connected regions (blobs) in a binary image.
# 8-connectivity means a pixel is considered connected to its horizontal, vertical, and diagonal neighbors.
# This implementation uses a union-find (parent) structure to merge equivalent labels.

def manual_ccl_8_conn(binary_img):

    height, width = binary_img.shape  
    # Get the dimensions of the input binary image.

    label_map = np.zeros((height, width), dtype=np.int32)  
    # Initialize an empty label map (all zeros). Will store labels for each object.

    parent = {}  
    # Dictionary to keep track of equivalent labels (union-find).
    # parent[label] = root label

    next_label = 1  
    # Start labeling foreground objects from 1. 0 is reserved for background.

    # --------------------------
    # PASS 1: Assign Temporary Labels
    # --------------------------
    # We loop through every pixel and assign temporary labels.
    # If a pixel has neighboring labeled pixels, we merge them later using union-find.

    for i in range(1, height - 1):  # Loop over rows (skip first and last to avoid boundary errors)
        for j in range(1, width - 1):  # Loop over columns

            if binary_img[i, j] > 0:  
                # Only process foreground pixels (white pixels)

                neighbors = [
                    label_map[i - 1, j - 1],  # Top-left
                    label_map[i - 1, j],      # Top
                    label_map[i - 1, j + 1],  # Top-right
                    label_map[i, j - 1]       # Left
                ]
                # Only check already-visited neighbors.
                # Since scan is left-to-right, top-to-bottom, these are the only neighbors that may already be labeled.

                non_zero = [n for n in neighbors if n > 0]  
                # Keep only neighbors that already have a label.

                if not non_zero:  
                    # Case 1: No labeled neighbors → this is a new object
                    label_map[i, j] = next_label  # Assign a new label
                    parent[next_label] = next_label  # Initialize its parent to itself
                    next_label += 1  # Increment label counter

                else:  
                    # Case 2: One or more labeled neighbors → assign the smallest label
                    smallest = min(non_zero)  # Choose smallest label to maintain consistency
                    label_map[i, j] = smallest

                    # Merge equivalent labels using union-find
                    for lbl in non_zero:  
                        root_a = smallest
                        root_b = lbl

                        while parent[root_a] != root_a:  # Find root of smallest label
                            root_a = parent[root_a]

                        while parent[root_b] != root_b:  # Find root of neighbor label
                            root_b = parent[root_b]

                        if root_a != root_b:  # Merge trees if roots are different
                            parent[root_b] = root_a

    # --------------------------
    # PASS 2: Resolve Label Equivalences
    # --------------------------
    # Replace all temporary labels with their root labels to finalize labeling.

    for i in range(height):
        for j in range(width):

            if label_map[i, j] > 0:  # Only for foreground pixels
                root = label_map[i, j]
                while parent[root] != root:  # Find root of current label
                    root = parent[root]
                label_map[i, j] = root  # Assign root label

    return label_map  # Return fully labeled image


# ==========================================================
# EXTRACT LARGEST CONNECTED COMPONENT
# ==========================================================
# Often, the largest bright object in a fundus image is the optic disc or cup.
# This function keeps only the largest connected component and removes all others.

def extract_largest_component(label_map):

    labels, counts = np.unique(label_map[label_map > 0], return_counts=True)  
    # Find all unique labels (excluding background) and count pixels in each label.

    if len(counts) == 0:  
        # No objects found
        return np.zeros_like(label_map, dtype=np.uint8)  # Return blank image

    dominant_label = labels[np.argmax(counts)]  
    # Choose the label with the largest number of pixels

    return (label_map == dominant_label).astype(np.uint8) * 255  
    # Return binary mask of largest object (0 = background, 255 = object)


# ==========================================================
# DICE SCORE CALCULATION
# ==========================================================
# Dice coefficient: Measures overlap between predicted mask and ground truth.
# Formula: Dice = 2 * |A ∩ B| / (|A| + |B|)

def calculate_dice(pred_mask, gt_path):

    if not os.path.exists(gt_path):
        return 0.0  # If GT does not exist, return 0

    gt = cv2.imread(gt_path, 0)  # Load GT mask as grayscale

    if gt is None:
        return 0.0  # Failed to load

    pred_bin = (pred_mask > 0).astype(np.float32)  # Convert predicted mask to binary float
    gt_bin = (gt > 0).astype(np.float32)  # Convert GT mask to binary float

    intersection = np.sum(pred_bin * gt_bin)  # Count pixels where both masks are 1
    total_area = np.sum(pred_bin) + np.sum(gt_bin)  # Total number of foreground pixels

    return (2.0 * intersection) / total_area if total_area > 0 else 0.0  # Dice coefficient


# ==========================================================
# GET DATASET PATHS
# ==========================================================
# Build a list of dictionaries containing image paths and GT paths.

def get_drishti_paths(base_dir, mode="Testing"):

    data_pairs = []  # List to store information about each sample

    subset_path = Path(base_dir) / mode / "Images"  # Folder containing images
    gt_root = Path(base_dir) / mode / "Test_GT"  # Folder containing ground truth

    for img_path in subset_path.rglob("*.png"):  # Recursive search for PNG images
        img_id = img_path.stem  # Get filename without extension
        gt_folder = gt_root / img_id / "SoftMap"  # Ground truth folder for this image

        data_pairs.append({
            "image": str(img_path),  # Image path
            "cup_gt": str(gt_folder / f"{img_id}_cupsegSoftmap.png"),  # Cup GT
            "disc_gt": str(gt_folder / f"{img_id}_ODsegSoftmap.png"),  # Disc GT
            "id": img_id  # Image ID
        })

    return data_pairs  # Return list of samples


# ==========================================================
# MAIN SEGMENTATION PIPELINE
# ==========================================================

def process_segmentation(sample, seg_dir, samp_dir):

    img = cv2.imread(sample['image'])  # Load image

    if img is None:
        return None, None  # Skip if image failed to load

    # --------------------------
    # STEP 1: GREEN CHANNEL EXTRACTION
    # --------------------------
    # Green channel gives best contrast for fundus structures
    gray = img[:, :, 1]
    h, w = gray.shape

    # --------------------------
    # STEP 2: CONTRAST ENHANCEMENT (CLAHE)
    # --------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
    gray_enhanced = clahe.apply(gray)  # Apply CLAHE

    # --------------------------
    # STEP 3: NOISE REDUCTION
    # --------------------------
    gray_blurred = cv2.medianBlur(gray_enhanced, 21)  # Median blur to reduce noise

    # --------------------------
    # STEP 4: CIRCULAR ROI MASK
    # --------------------------
    center_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(center_mask, (w // 2, h // 2), int(min(w, h) * 0.55), 255, -1)  
    # Circle mask centered in the image

    masked_roi = cv2.bitwise_and(gray_blurred, gray_blurred, mask=center_mask)  
    # Apply mask to remove black corners

    valid_pixels = masked_roi[masked_roi > 0]  # Only non-zero pixels

    if len(valid_pixels) == 0:
        return None, None  # Skip if empty

    # ======================================================
    # OPTIC DISC SEGMENTATION
    # ======================================================
    thresh_od = np.percentile(valid_pixels, 92)  # V-set: top 8% brightest pixels

    binary_od = (masked_roi > thresh_od).astype(np.uint8)  # Threshold image

    kernel = np.ones((5, 5), np.uint8)  # Morphology kernel

    binary_od = cv2.morphologyEx(binary_od, cv2.MORPH_OPEN, kernel)  # Remove small noise

    labels_od = manual_ccl_8_conn(binary_od)  # Label connected components
    p_disc = extract_largest_component(labels_od)  # Keep largest component

    # ======================================================
    # OPTIC CUP SEGMENTATION
    # ======================================================
    if np.sum(p_disc) > 0:  # Proceed if disc detected
        disc_pixels = gray_blurred[p_disc > 0]  # Pixels inside disc
        if len(disc_pixels) > 0:
            thresh_cup = np.percentile(disc_pixels, 75)  # Brightest 25% for cup
            binary_cup = (gray_blurred > thresh_cup).astype(np.uint8)
            binary_cup = cv2.bitwise_and(binary_cup, binary_cup, mask=(p_disc > 0).astype(np.uint8))
            binary_cup = cv2.morphologyEx(binary_cup, cv2.MORPH_OPEN, kernel)
            labels_cup = manual_ccl_8_conn(binary_cup)
            p_cup = extract_largest_component(labels_cup)
        else:
            p_cup = np.zeros_like(gray)
    else:
        p_cup = np.zeros_like(gray)

    return p_disc, p_cup


# ==========================================================
# MAIN EXECUTION BLOCK
# ==========================================================
if __name__ == "__main__":

    dataset_root = r"D:\Nust\Semester 6\DIP\DIP Assignment1\Drishti-GS\Test"
    segmented_results_dir = "Results/Masks"
    output_samples_dir = "Results/Comparisons"

    for folder in [segmented_results_dir, output_samples_dir]:
        os.makedirs(folder, exist_ok=True)  # Create output directories if they don't exist

    test_samples = get_drishti_paths(dataset_root, mode="Test")  # Get list of test images

    results_list = []  # Store Dice results

    print(f"Processing {len(test_samples)} images...")

    for sample in test_samples:

        p_disc, p_cup = process_segmentation(sample, segmented_results_dir, output_samples_dir)

        if p_disc is None:
            continue  # Skip failed images

        d_od = calculate_dice(p_disc, sample['disc_gt'])
        d_cup = calculate_dice(p_cup, sample['cup_gt'])

        results_list.append({"ID": sample['id'], "Disc Dice": d_od, "Cup Dice": d_cup})

        print(f"Processed {sample['id']} | Disc: {d_od:.4f} | Cup: {d_cup:.4f}")

    if results_list:
        df = pd.DataFrame(results_list)
        print("\n" + "=" * 30)
        print(f"Average Disc Dice: {df['Disc Dice'].mean():.4f}")
        print(f"Average Cup Dice:  {df['Cup Dice'].mean():.4f}")
        print("=" * 30)
        df.to_csv("Results/final_scores.csv", index=False)  # Save CSV