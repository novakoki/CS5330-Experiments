# Project Requirements: nuScenes Car Detection (Custom Split)

## 1. Project Overview
**Goal:** Conduct a 4-step ablation study to train a LiDAR-based 3D object detector (PointPillars) for the "Car" class.
**Constraint:** Google Colab T4 GPU (Limit: ~6 hours training time).
**Method:** Use a **Custom Data Split** derived from nuScenes Blobs 01 & 02 to create a reasonable training and validation size.

## 2. Data Requirements (The Custom Split)
We will download `v1.0-trainval_meta` and `v1.0-trainval01_blobs` + `v1.0-trainval02_blobs`.
From these ~180 scenes, we will generate:

* **Custom-Full-Train:** ~150 scenes (Scenes from Blobs 01/02 that belong to official Train).
* **Custom-Small-Train:** ~30 scenes (A subset of Custom-Full-Train for the baseline).
* **Custom-Val (The Golden Set):** ~30 scenes (Scenes from Blobs 01/02 that belong to official Val).
    * **CRITICAL:** This 30-scene set must be used for ALL 4 experiments to ensure valid comparison.

## 3. Experimental Scope

### Experiment 1: The Baseline
- **Goal:** Establish lower-bound performance.
- **Data:** `Custom-Small-Train` (~30 scenes).
- **Augmentation:** **DISABLED** (No rotation, scaling, flip, or copy-paste).
- **Validation:** `Custom-Val`.

### Experiment 2: Geometric Augmentation
- **Goal:** Measure improvement from standard geometric transformations.
- **Data:** `Custom-Small-Train` (~30 scenes).
- **Augmentation:** **ENABLED** (Global Rotation, Scaling, Random Flip).
- **Validation:** `Custom-Val`.

### Experiment 3: Data Scaling
- **Goal:** Measure impact of 5x data increase.
- **Data:** `Custom-Full-Train` (~150 scenes).
- **Augmentation:** **ENABLED** (Include Copy-Paste/GT Sampling).
- **Validation:** `Custom-Val`.

### Experiment 4: Transfer Learning
- **Goal:** Measure improvement from domain adaptation.
- **Data:** `Custom-Full-Train`.
- **Initialization:** Load weights pre-trained on KITTI.
- **Validation:** `Custom-Val`.

## 4. Output Requirements
- **Metrics:** mAP (Car) on the 30-scene Validation Set.
- **Visuals:** Inference images on `Custom-Val` scenes.