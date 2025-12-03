#!/usr/bin/env bash
# Lightweight setup for Google Colab runtime.
set -e

pip install --quiet nuscenes-devkit matplotlib

echo "Download nuScenes blobs (v1.0-trainval_meta.tgz, v1.0-trainval01_blobs.tgz, v1.0-trainval02_blobs.tgz) manually from the nuScenes website."
echo "Upload and extract them into data/nuscenes/ before running training."

python scripts/create_splits.py --data_root data/nuscenes --output_dir data/splits
echo "Setup complete. Ready to launch training via: python train.py --config configs/exp1_baseline.py"

