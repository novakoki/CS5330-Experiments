#!/usr/bin/env bash

# Run all experiments sequentially, letting each config choose its own timestamped
# work_dir via RUN_SUFFIX.
set -euo pipefail

EXPS=(
  exp1_baseline
  exp2_augmentation
  exp3_scaling
  exp4_transfer
)

for EXP in "${EXPS[@]}"; do
  RUN_SUFFIX="${EXP}_$(date +%Y%m%d-%H%M%S)"
  echo ">>> Running ${EXP} with RUN_SUFFIX=${RUN_SUFFIX}"
  RUN_SUFFIX="${RUN_SUFFIX}" python mmdetection3d_implementaion/mmdetection3d/tools/train.py \
    "mmdetection3d_implementaion/configs/${EXP}.py"
  echo ">>> Finished ${EXP}"
done

echo "All experiments completed."
