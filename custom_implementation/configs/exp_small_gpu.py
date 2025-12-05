"""Configuration tuned for ~6GB GPU without FP16."""
from copy import deepcopy

from configs.base_car import config as base

config = deepcopy(base)

# Paths
config["work_dir"] = "outputs/exp_small_gpu"

# GPU/memory constraints
config["amp"] = False  # disable FP16
config["train"]["batch_size"] = 2  # nudge up to improve GPU utilization; drop back to 1 if OOM
# config["dataset"]["max_voxels"] = {"train": 10000, "val": 10000}  # moderate cap for more pillars than ultra-conservative setting

# Optional: keep num_workers/pin_memory/prefetch from base; tweak locally if CPU becomes the bottleneck.
config["train"]["epochs"] = 1