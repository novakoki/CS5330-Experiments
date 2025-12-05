"""Experiment 4: Full split with augmentation + KITTI pretrained init."""
from copy import deepcopy

from configs.base_car import config as base

config = deepcopy(base)
config["work_dir"] = "outputs/exp4_transfer"
config["dataset"]["train_scenes"] = "data/splits/custom_full_train.json"
config["dataset"]["augmentations"] = {
    "rotation": True,
    "scaling": True,
    "flip": True,
    "copy_paste": True,
}
# Point to converted PointPillars KITTI weights produced by utils/weight_transfer.py
config["pretrained"] = "weights/kitti_pointpillars_car_init.pth"
