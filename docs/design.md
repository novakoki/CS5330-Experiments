# Technical Design Document

## 1. Directory Structure
Root directory should contain:
- `configs/`: Custom configuration files.
- `data/nuscenes/`:
    - `v1.0-trainval_meta/`: Metadata.
    - `samples/`: LiDAR data from Blobs 01 & 02.
    - `nuscenes_infos_custom_train.pkl`: Generated Info file for 150 scenes.
    - `nuscenes_infos_custom_val.pkl`: Generated Info file for 30 scenes.

## 2. Data Preparation Pipeline
### Script: `scripts/create_custom_splits.py`
**Logic:**
1.  Load the standard `nuscenes_infos_train.pkl` and `val.pkl`.
2.  Scan the `data/nuscenes/samples/LIDAR_TOP/` directory to see which files actually exist (from Blobs 01 & 02).
3.  **Split Logic:**
    - If a scene's LiDAR files exist AND it is in the official Train list -> Add to `nuscenes_infos_custom_train.pkl`.
    - If a scene's LiDAR files exist AND it is in the official Val list -> Add to `nuscenes_infos_custom_val.pkl`.
4.  **Subset Logic:** Create `nuscenes_infos_custom_small.pkl` by taking the first 30 scenes from the custom train file.

## 3. Configuration Management

### Base Config (`configs/base_car.py`)
- Model: PointPillars.
- Classes: `['car']`.
- **Validation:** Always point to `ann_file = 'nuscenes_infos_custom_val.pkl'`.

### Experiment 1 Config (`configs/exp1_baseline.py`)
- Inherit from Base.
- **Dataset:** `ann_file = 'nuscenes_infos_custom_small.pkl'`.
- **Pipeline:** Disable `GlobalRotScaleTrans`, `RandomFlip3D`, `ObjectSample`.

### Experiment 2 Config (`configs/exp2_aug.py`)
- Inherit from Exp 1.
- **Pipeline:** Enable `GlobalRotScaleTrans`, `RandomFlip3D`. Keep `ObjectSample` disabled.

### Experiment 3 Config (`configs/exp3_scale.py`)
- Inherit from Exp 2.
- **Dataset:** Change `ann_file = 'nuscenes_infos_custom_train.pkl'` (The full 150 scenes).
- **Pipeline:** Enable `ObjectSample` (Copy-Paste).

### Experiment 4 Config (`configs/exp4_transfer.py`)
- Inherit from Exp 3.
- **Load:** `load_from = 'checkpoints/kitti_pretrained.pth'`.
- **Head:** Re-initialize classification head.

## 4. Execution Workflow
1.  **Download:** `v1.0-trainval_meta`, `v1.0-trainval01_blobs`, `v1.0-trainval02_blobs`.
2.  **Filter:** Run `scripts/create_custom_splits.py` to generate the 3 pkl files (Small, Train, Val).
3.  **Train Exp 1 & 2:** Use the "Small" pkl.
4.  **Train Exp 3 & 4:** Use the "Train" pkl.
5.  **Evaluate:** All experiments evaluate on "Val" pkl.