"""Experiment 3: Full custom train split with copy-paste augmentation."""
from copy import deepcopy

from configs.base_car import config as base

config = deepcopy(base)
config["work_dir"] = "outputs/exp3_scaling"
config["dataset"]["train_scenes"] = "data/splits/custom_full_train.json"
config["dataset"]["augmentations"] = {
    "rotation": True,
    "scaling": True,
    "flip": True,
    "copy_paste": True,
}
config["pretrained"] = None
