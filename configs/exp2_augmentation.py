"""Experiment 2: Small split with geometric augmentation."""
from copy import deepcopy

from configs.base_car import config as base

config = deepcopy(base)
config["work_dir"] = "outputs/exp2_augmentation"
config["dataset"]["train_scenes"] = "data/splits/custom_small_train.json"
config["dataset"]["augmentations"] = {
    "rotation": True,
    "scaling": True,
    "flip": True,
    "copy_paste": False,
}
config["pretrained"] = None
