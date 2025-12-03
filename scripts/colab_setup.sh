#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a Colab runtime for the custom nuScenes experiments.
# - Installs pinned dependencies
# - Clones mmdetection3d and installs it in editable mode
# - Generates custom split PKLs if the downloaded nuScenes blobs are present

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMD3D_DIR="${MMD3D_DIR:-/content/mmdetection3d}"
DATA_ROOT="${DATA_ROOT:-$EXP_ROOT/data/nuscenes}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[setup] Using EXP_ROOT=$EXP_ROOT"
echo "[setup] Installing Python dependencies..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$EXP_ROOT/requirements.txt"

if [ ! -d "$MMD3D_DIR" ]; then
  echo "[setup] Cloning mmdetection3d into $MMD3D_DIR"
  git clone --depth 1 https://github.com/open-mmlab/mmdetection3d.git "$MMD3D_DIR"
else
  echo "[setup] Found existing mmdetection3d at $MMD3D_DIR"
fi

pushd "$MMD3D_DIR" >/dev/null
echo "[setup] Installing mmdetection3d in editable mode..."
"$PYTHON_BIN" -m pip install -e .
popd >/dev/null

export PYTHONPATH="$MMD3D_DIR:$PYTHONPATH"
echo "[setup] PYTHONPATH updated to include $MMD3D_DIR"

if [ -d "$DATA_ROOT/samples/LIDAR_TOP" ]; then
  echo "[setup] Creating custom split PKLs under $DATA_ROOT"
  "$PYTHON_BIN" "$EXP_ROOT/scripts/create_custom_splits.py" --data-root "$DATA_ROOT"
else
  echo "[setup] Skipping split creation (LiDAR files not found at $DATA_ROOT/samples/LIDAR_TOP)."
  echo "        Download v1.0-trainval_meta and blobs 01/02 into $DATA_ROOT then re-run this script."
fi

echo "[setup] Done. To train, export MMD3D_DIR=$MMD3D_DIR and run scripts/run_experiments.sh"
