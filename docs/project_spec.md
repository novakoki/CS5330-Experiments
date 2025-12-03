# Project Specification: Custom PointPillars for nuScenes Car Detection

## 1. Project Overview

**Goal:** Build and train a lightweight, custom PyTorch implementation of the PointPillars 3D object detector, focused exclusively on the "Car" class.
**Platform:** Google Colab (Single T4 GPU).
**Constraints:**

- **No MMDetection3D:** To avoid dependency issues, we use a pure PyTorch approach.
- **Limited Compute:** Training must finish within ~6 hours.
- **Limited Storage:** We use a custom partial download of the nuScenes dataset.

## 2. Data Strategy: The "Custom Split"

Since the full dataset is too large, we will engineer a custom dataset from specific nuScenes "Blobs".

### 2.1 Source Data

We require the following files from the nuScenes website:

1. `v1.0-trainval_meta.tgz` (Metadata)
2. `v1.0-trainval01_blobs.tgz` (LiDAR for scenes ~0001 to ~0088)
3. `v1.0-trainval02_blobs.tgz` (LiDAR for scenes ~0089 to ~0170)

### 2.2 The Split Logic

We will parse the downloaded scenes and sort them into two lists based on the official nuScenes split:

* **Custom-Full-Train (~150 scenes):** All scenes present in Blobs 01/02 that belong to the official *Train* split.
* **Custom-Small-Train (~30 scenes):** A subset of Custom-Full-Train (used for the Baseline experiment).
* **Custom-Val (~30 scenes):** All scenes present in Blobs 01/02 that belong to the official *Val* split.
  * **CRITICAL:** This `Custom-Val` set is the "Golden Set". It must be used as the validation set for **all 4 experiments** to ensure scientific consistency.

## 3. Experimental Scope (Ablation Study)

We will implement 4 distinct configuration modes to demonstrate performance improvements.

| Exp ID      | Name                   | Training Data | Augmentation               | Weights                    |
| :---------- | :--------------------- | :------------ | :------------------------- | :------------------------- |
| **1** | **Baseline**     | 30 Scenes     | None                       | Random Init                |
| **2** | **Augmentation** | 30 Scenes     | Rotate, Scale, Flip        | Random Init                |
| **3** | **Data Scaling** | 150 Scenes    | + Copy-Paste (GT Sampling) | Random Init                |
| **4** | **Transfer**     | 150 Scenes    | All of the above           | **KITTI Pretrained** |

## 4. Technical Design

### 4.1 Directory Structure

```text
.
├── data/
│   └── nuscenes/         # Unpacked metadata and 'samples/' folder
├── model/
│   ├── pointpillars.py   # Main nn.Module
│   ├── pillars.py        # Voxelization (PillarFeatureNet)
│   └── backbone.py       # SECOND/RPN Backbone
├── dataset/
│   ├── nuscenes.py       # Custom torch Dataset using nuscenes-devkit
│   └── transforms.py     # Augmentations (Geom + Copy-Paste)
├── utils/
│   ├── anchors.py        # Anchor generation logic
│   ├── evaluation.py     # mAP calculation
│   └── weight_transfer.py # Script to convert MMDet3D weights
├── scripts/
│   ├── create_splits.py  # Generates train/val JSON lists
│   ├── eval_official.py  # Runs nuScenes-devkit evaluation
│   └── visualize.py      # Generates BEV images
└── train.py              # Main Training Loop
```


### 4.2 Module Specifications

#### A. Dataset Loader (`dataset/nuscenes.py`)

* **Library:** Use `nuscenes-devkit` to query tokens.
* **Logic:**
  * Load point cloud (`.bin`).
  * Filter Ground Truth boxes for `category == 'vehicle.car'`.
  * Transform Global Coordinates **$\to$** Ego **$\to$** Sensor (Lidar).
* **Output:** `(points, gt_boxes_3d, num_voxels, coords)`

#### B. The Model (`model/pointpillars.py`)

* **Architecture:** Simplified PointPillars.
* **Anchors:** Must be tuned for nuScenes Cars.
  * **Size:** **$W=1.97m, L=4.63m, H=1.73m$**
  * **Rotations:** **$0, \pi/2$**
  * **Z-Center:** **$-1.78m$** (Ground level)

#### C. Weight Transfer (`utils/weight_transfer.py`)

* **Logic:**
  1. Load a pretrained `.pth` file (from MMDet3D model zoo).
  2. Create a mapping dictionary to rename keys (e.g., `backbone.blocks.0` **$\to$** `feature_extractor.conv1`).
  3. Save a new `init_weights.pth` compatible with our custom model.

## 5. Evaluation & Visualization

### 5.1 Official Metrics (`scripts/eval_official.py`)

* **Goal:** Generate a `submission.json` file.
* **Process:**
  1. Run inference on the `Custom-Val` set.
  2. Transform predictions: Sensor **$\to$** Global coordinates.
  3. Format to nuScenes JSON spec (translation, size, rotation, score).
  4. Call `NuScenesEval` to compute mAP and NDS.

### 5.2 Reporting Assets (`scripts/visualize.py`)

* **Qualitative:**
  * Generate "Red vs Green" Bird's Eye View (BEV) plots.
  * Green = Ground Truth, Red = Prediction.
* **Quantitative:**
  * `plot_scaling_law.py`: Line chart of mAP vs. Dataset Size.
  * `plot_ablation.py`: Bar chart of the 4 experiments.

## 6. Implementation Roadmap

1. **Environment & Data:** Write `scripts/create_splits.py` to scan the downloaded blobs and generate `train_scenes.json` and `val_scenes.json`.
2. **Dataset Class:** Implement `dataset/nuscenes.py`. Ensure it visualizes one sample correctly before moving on.
3. **Model Architecture:** Implement `model/pointpillars.py` (Voxelization + Backbone + Head).
4. **Training Loop:** Write `train.py` with Focal Loss and OneCycleLR scheduler.
5. **Baseline Run:** Execute Experiment 1.
6. **Evaluation:** Implement `scripts/eval_official.py` to check the Baseline score.
7. **Iterate:** Add Augmentation (Exp 2), then Data Scaling (Exp 3), then Transfer Learning (Exp 4).
