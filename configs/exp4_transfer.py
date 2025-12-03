"""Experiment 4: Transfer learning with KITTI pretraining."""

_base_ = ["./exp3_scale.py"]

load_from = "checkpoints/kitti_pretrained.pth"

model = dict(
    bbox_head=dict(
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform", bias=0.0)
    )
)
