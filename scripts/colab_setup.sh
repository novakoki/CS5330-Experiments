#!/usr/bin/env bash
# Lightweight setup for Google Colab runtime with Drive-hosted nuScenes blobs.
set -euo pipefail

# Paths (archives will be downloaded to data_dir)
DATA_DIR="${DATA_DIR:-data/nuscenes}"
URL_META="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz"
URL_KF01="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval01_keyframes.tgz"
URL_KF02="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval02_keyframes.tgz"

pip install --quiet nuscenes-devkit matplotlib

echo "Downloading nuScenes keyframe archives into: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"

download_and_extract() {
  local url="$1"
  local dest_dir="$2"
  local fname
  fname="$(basename "${url}")"
  if [ ! -f "${fname}" ]; then
    echo "Downloading ${fname} ..."
    curl -L -o "${fname}" "${url}"
  else
    echo "Found existing ${fname}, skipping download."
  fi
  echo "Extracting ${fname} ..."
  tar -xzf "${fname}" -C "${dest_dir}"
}

download_and_extract "${URL_META}" "${DATA_DIR}"
download_and_extract "${URL_KF01}" "${DATA_DIR}"
download_and_extract "${URL_KF02}" "${DATA_DIR}"

python scripts/create_splits.py --data_root "${DATA_DIR}" --output_dir data/splits
echo "Setup complete. Ready to launch training via: python train.py --config configs/exp1_baseline.py"
