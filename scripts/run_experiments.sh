#!/usr/bin/env bash
set -euo pipefail

# Run the four experiments sequentially using mmdetection3d's train script.

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMD3D_DIR="${MMD3D_DIR:-/content/mmdetection3d}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="${WORK_DIR:-$EXP_ROOT/work_dirs}"

CONFIGS=(
  "$EXP_ROOT/configs/exp1_baseline.py"
  "$EXP_ROOT/configs/exp2_aug.py"
  "$EXP_ROOT/configs/exp3_scale.py"
  "$EXP_ROOT/configs/exp4_transfer.py"
)

mkdir -p "$WORK_DIR"

if [ ! -d "$MMD3D_DIR" ]; then
  echo "Missing mmdetection3d directory at $MMD3D_DIR. Run scripts/colab_setup.sh first." >&2
  exit 1
fi

export PYTHONPATH="$MMD3D_DIR:$PYTHONPATH"

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg%.*}")"
  run_dir="$WORK_DIR/$name"
  mkdir -p "$run_dir"
  echo "[run] Training $name -> work_dir=$run_dir"
  (cd "$MMD3D_DIR" && "$PYTHON_BIN" tools/train.py "$cfg" --work-dir "$run_dir")
done

echo "[run] All experiments finished. Logs and checkpoints are under $WORK_DIR"
