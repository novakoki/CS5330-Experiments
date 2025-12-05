# MMDetection3D configs for custom nuScenes PointPillars (car only)

This folder mirrors the experiments in `custom_implementation/docs` but uses the
official MMDetection3D training pipeline.

## Preparation
1. Generate scene-token splits (matches the partial blob download):
   ```bash
   python mmdetection3d_implementaion/scripts/create_custom_splits.py
   ```
   This now also writes `custom_val_small.json` (default 30 scenes) to speed up experiments. The script now filters out any scene with missing LIDAR files across the entire scene to avoid converter crashes on partial downloads.
2. Build MMDetection3D info files filtered to the custom splits (uses keyframes only, `max_sweeps=0` by default):
   ```bash
   python mmdetection3d_implementaion/scripts/prepare_mmdet_data.py  # uses custom_val_small.json by default
   ```
   This writes `custom_infos_{small_train,full_train,val}.pkl` under `data/nuscenes/`.
   If you want copy-paste/GT sampling for Exp3/Exp4, also create a database at `data/nuscenes/custom_dbinfos_train.pkl`
   (e.g., adapt `tools/create_data.py` to your no-sweeps setup) before training.

## Training commands (from `mmdetection3d/`)
- Experiment 1 (small split, no aug):
  ```bash
  python tools/train.py ../mmdetection3d_implementaion/configs/exp1_baseline.py
  ```
- Experiment 2 (small split + geo aug):
  ```bash
  python tools/train.py ../mmdetection3d_implementaion/configs/exp2_augmentation.py
  ```
- Experiment 3 (full split + geo aug, optional GT sampling):
  ```bash
  python tools/train.py ../mmdetection3d_implementaion/configs/exp3_scaling.py
  ```
- Experiment 4 (full split + aug, KITTI init):
  ```bash
  python tools/train.py ../mmdetection3d_implementaion/configs/exp4_transfer.py
  ```

All configs point to `data/nuscenes/` and assume training is launched from the `mmdetection3d/` directory.
