# Custom PointPillars for nuScenes Cars

End-to-end lightweight PointPillars pipeline (PyTorch only) targeting the nuScenes *vehicle.car* class. Designed for Colab (single T4) with the partial blob download (trainval_meta + blobs 01/02).

## Setup
- Install deps: `pip install torch torchvision torchaudio nuscenes-devkit matplotlib`
- Download `v1.0-trainval_meta.tgz`, `v1.0-trainval01_keyframes.tgz`, `v1.0-trainval02_keyframes.tgz` from nuScenes; extract to `data/nuscenes/` (keyframes-only is sufficient for this pipeline).
- Generate splits: `python scripts/create_splits.py --data_root data/nuscenes --output_dir data/splits`
- Optional Colab bootstrap: `bash scripts/colab_setup.sh`

## Training
Choose a config under `configs/`:
- `exp1_baseline.py` – 30 scenes, no aug
- `exp2_augmentation.py` – 30 scenes + rotate/scale/flip
- `exp3_scaling.py` – 150 scenes + copy-paste
- `exp4_transfer.py` – 150 scenes + aug + KITTI init

Run: `python train.py --config configs/exp1_baseline.py --device cuda`
Checkpoints land in `outputs/<exp>/` (best, per-epoch, last).

## Evaluation & Visuals
- Official nuScenes eval + submission: `python scripts/eval_official.py --config configs/exp1_baseline.py --checkpoint outputs/exp1_baseline/best.pth --output outputs/exp1_baseline/submission.json`
- Quick BEV overlays (green GT, red pred): `python scripts/visualize.py --config configs/exp1_baseline.py --checkpoint outputs/exp1_baseline/best.pth --output_dir viz/`
- Plotting helpers: `scripts/plot_scaling_law.py`, `scripts/plot_ablation.py` (expect JSON inputs).

## Transfer Weights
Convert MMDetection3D PointPillars weights: `python utils/weight_transfer.py --src mmdet_ckpt.pth --dst weights/kitti_pointpillars_car_init.pth`

## Key Files
- `dataset/` – nuScenes loader + augmentations
- `model/` – voxelizer, PFN, BEV backbone, detection head
- `utils/anchors.py` – anchor gen + target assignment
- `train.py` – training loop (OneCycleLR, focal loss, AMP)
- `scripts/` – splits, eval, visuals, plotting, Colab setup
