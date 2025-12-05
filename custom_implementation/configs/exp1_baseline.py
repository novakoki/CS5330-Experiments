"""Experiment 1: Baseline on small custom split without augmentation."""
from copy import deepcopy

from configs.base_car import config as base

config = deepcopy(base)
config["work_dir"] = "outputs/exp1_baseline"
config["dataset"]["train_scenes"] = "data/splits/custom_small_train.json"
config["dataset"]["augmentations"] = {
    "rotation": False,
    "scaling": False,
    "flip": False,
    "copy_paste": False,
}
config["pretrained"] = None
