#!/usr/bin/env bash
# Lightweight setup for Google Colab runtime with Drive-hosted nuScenes blobs.
set -euo pipefail

# Paths
SRC_DIR="${SRC_DIR:-/content/drive/MyDrive}"
DATA_DIR="${DATA_DIR:-data/nuscenes}"

pip install --quiet nuscenes-devkit matplotlib

echo "Using source directory: ${SRC_DIR}"
echo "Extracting nuScenes archives into: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"

for archive in v1.0-trainval_meta.tgz v1.0-trainval01_blobs.tgz v1.0-trainval02_blobs.tgz; do
  src_path="${SRC_DIR}/${archive}"
  if [ ! -f "${src_path}" ]; then
    echo "Missing ${src_path}. Please place the archive in ${SRC_DIR}."
    exit 1
  fi
  echo "Extracting ${archive}..."
  tar -xzf "${src_path}" -C "${DATA_DIR}"
done

python scripts/create_splits.py --data_root "${DATA_DIR}" --output_dir data/splits
echo "Setup complete. Ready to launch training via: python train.py --config configs/exp1_baseline.py"
