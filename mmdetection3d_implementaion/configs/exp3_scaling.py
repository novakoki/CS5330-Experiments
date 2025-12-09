from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path

from mmengine.config import read_base

with read_base():
    from .pointpillars_car_base import *  # noqa: F401,F403,F405

db_sampler = None
if Path(db_info_path).exists():
    db_sampler = dict(
        data_root=data_root,
        info_path=str(db_info_path),
        rate=1.0,
        prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(car=5)),
        classes=class_names,
        sample_groups=dict(car=15),
        points_loader=dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=5,
            use_dim=5,
            backend_args=backend_args,
        ),
    )

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
]

if db_sampler is not None:
    train_pipeline.append(dict(type="ObjectSample", db_sampler=db_sampler))

train_pipeline += [
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.9, 1.1],
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
train_dataloader["dataset"]["ann_file"] = "custom_infos_full_train.pkl"
train_dataloader["dataset"]["pipeline"] = train_pipeline

run_suffix = os.environ.get("RUN_SUFFIX", datetime.now().strftime("%Y%m%d-%H%M%S"))
work_dir = f"../outputs/mmdet/exp3_scaling/{run_suffix}"
load_from = None

del os
del datetime
