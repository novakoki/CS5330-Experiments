from copy import deepcopy
from datetime import datetime
import os

from mmengine.config import read_base

with read_base():
    from .pointpillars_car_base import *  # noqa: F401,F403,F405

base_train_dataloader = deepcopy(train_dataloader)
train_dataloader = deepcopy(base_train_dataloader)
train_dataloader["dataset"]["ann_file"] = "custom_infos_small_train.pkl"

run_suffix = os.environ.get("RUN_SUFFIX", datetime.now().strftime("%Y%m%d-%H%M%S"))
work_dir = f"outputs/mmdet/exp1_baseline/{run_suffix}"
load_from = None

# Clean up modules to avoid deepcopy issues in MMEngine.
del os
del datetime
