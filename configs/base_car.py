"""Base config for PointPillars on the custom nuScenes car-only split."""

import copy

# Dataset settings
dataset_type = "NuScenesDataset"
data_root = "data/nuscenes/"
class_names = ["car"]
metainfo = dict(classes=class_names)

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)

point_cloud_range = [-50, -50, -5, 50, 50, 3]
voxel_size = [0.2, 0.2, 8]
norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + "nuscenes_infos_custom_train.pkl",
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(car=5)),
    classes=class_names,
    sample_groups=dict(car=2),
)

def build_train_pipeline(transforms_cfg):
    pipeline = [
        dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend="disk"),
        ),
        dict(
            type="LoadPointsFromMultiSweeps",
            sweeps_num=10,
            use_dim=[0, 1, 2, 3, 4],
            pad_empty_sweeps=True,
            remove_close=True,
        ),
        dict(
            type="LoadAnnotations3D",
            with_bbox_3d=True,
            with_label_3d=True,
            with_attr_label=False,
        ),
    ]
    if transforms_cfg.get("use_object_sample", True):
        pipeline.append(dict(type="ObjectSample", db_sampler=db_sampler))
    if transforms_cfg.get("use_global_rot_scale", True):
        pipeline.append(
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.9, 1.1],
                translation_std=[0, 0, 0],
            )
        )
    if transforms_cfg.get("use_random_flip", True):
        pipeline.append(
            dict(
                type="RandomFlip3D",
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.0,
            )
        )
    pipeline.extend(
        [
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="PointShuffle"),
            dict(type="DefaultFormatBundle3D", class_names=class_names),
            dict(
                type="Collect3D",
                keys=["points", "gt_bboxes_3d", "gt_labels_3d"],
                meta_keys=("token",),
            ),
        ]
    )
    return pipeline

train_transforms = dict(
    use_global_rot_scale=True,
    use_random_flip=True,
    use_object_sample=True,
)

train_pipeline = build_train_pipeline(train_transforms)

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend="disk"),
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=point_cloud_range,
    ),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["points"]),
]

train_ann_file = "nuscenes_infos_custom_train.pkl"
val_ann_file = "nuscenes_infos_custom_val.pkl"

train_dataloader = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    shuffle=True,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
    ),
)

val_dataloader = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    shuffle=False,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
    ),
)

test_dataloader = copy.deepcopy(val_dataloader)

optimizer = dict(type="AdamW", lr=2e-3, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="cyclic",
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy="cyclic", target_ratio=(0.85, 1), cyclic_times=1, step_ratio_up=0.4
)

runner = dict(type="EpochBasedRunner", max_epochs=24)
evaluation = dict(interval=2, pipeline=test_pipeline)
checkpoint_config = dict(interval=2, max_keep_ckpts=3)
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# Model definition (single-class PointPillars)
model = dict(
    type="PointPillars",
    voxel_layer=dict(
        max_num_points=32,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range,
    ),
    voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        norm_cfg=norm_cfg,
    ),
    middle_encoder=dict(type="PointPillarsScatter", in_channels=64, output_shape=[512, 512]),
    backbone=dict(
        type="SECOND",
        in_channels=64,
        norm_cfg=norm_cfg,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
    ),
    neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=norm_cfg,
    ),
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="Anchor3DRangeGenerator",
            ranges=[point_cloud_range],
            sizes=[[4.73, 2.08, 1.77]],
            rotations=[0, 1.57],
            reshape_out=False,
        ),
        diff_rad_by_sin=True,
        assigner_per_size=False,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=0.1111111111111111, loss_weight=1.0),
        loss_dir=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    train_cfg=dict(
        assigner=[
            dict(
                type="MaxIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1,
            )
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.2,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500,
    ),
)

# Reusable hooks/runtime defaults
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
