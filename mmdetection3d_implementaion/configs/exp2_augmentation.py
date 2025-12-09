from copy import deepcopy
from datetime import datetime
import os

from mmengine.config import read_base

with read_base():
    from .pointpillars_car_base import *  # noqa: F401,F403,F405

aug_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

base_train_dataloader = deepcopy(train_dataloader)
train_dataloader = deepcopy(base_train_dataloader)
train_dataloader["dataset"]["ann_file"] = "custom_infos_small_train.pkl"
train_dataloader["dataset"]["pipeline"] = aug_pipeline

run_suffix = os.environ.get("RUN_SUFFIX", datetime.now().strftime("%Y%m%d-%H%M%S"))
work_dir = f"../outputs/mmdet/exp2_augmentation/{run_suffix}"
load_from = None

del os
del datetime
