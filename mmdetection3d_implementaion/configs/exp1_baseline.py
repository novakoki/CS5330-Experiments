from copy import deepcopy

from mmengine.config import read_base

with read_base():
    from .pointpillars_car_base import *  # noqa: F401,F403,F405

base_train_dataloader = deepcopy(train_dataloader)
train_dataloader = deepcopy(base_train_dataloader)
train_dataloader["dataset"]["ann_file"] = "custom_infos_small_train.pkl"

work_dir = "../outputs/mmdet/exp1_baseline"
load_from = None
