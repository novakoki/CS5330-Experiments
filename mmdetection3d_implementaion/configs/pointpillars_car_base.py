"""Base MMDetection3D config for car-only PointPillars on custom nuScenes splits."""
from pathlib import Path

# Runtime defaults (mirrors mmdet3d/configs/_base_/default_runtime.py to avoid
# lazy/non-lazy config chain issues).
default_scope = "mmdet3d"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1,
        max_keep_ckpts=5,
        save_best="auto",
        rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="Det3DVisualizationHook"),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
data_root = "data/nuscenes/"
class_names = ("car",)
metainfo = dict(classes=class_names)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.16, 0.16, 0.2]

input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts="samples/LIDAR_TOP", img="", sweeps="")
backend_args = None

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="NuScenesDataset",
        data_root=data_root,
        ann_file="custom_infos_small_train.pkl",
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="NuScenesDataset",
        data_root=data_root,
        ann_file="custom_infos_val.pkl",
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="NuScenesMetric",
    data_root=data_root,
    ann_file=data_root + "custom_infos_val.pkl",
    metric="bbox",
    classes=class_names,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2),
)

max_epochs = 10
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0 / 10,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        eta_min=1e-5,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

model = dict(
    type="MVXFasterRCNN",
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(12000, 12000),
        ),
    ),
    pts_voxel_encoder=dict(
        type="HardVFE",
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
    ),
    pts_middle_encoder=dict(
        type="PointPillarsScatter",
        in_channels=64,
        output_shape=[int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
                      int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])],
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=64,
        norm_cfg=dict(type="naiveSyncBN2d", eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
    ),
    pts_neck=dict(
        type="mmdet.FPN",
        norm_cfg=dict(type="naiveSyncBN2d", eps=1e-3, momentum=0.01),
        act_cfg=dict(type="ReLU"),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3,
    ),
    pts_bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="AlignedAnchor3DRangeGenerator",
            ranges=[[point_cloud_range[0], point_cloud_range[1], -1.78, point_cloud_range[3], point_cloud_range[4], -1.78]],
            scales=[1, 2, 4],
            sizes=[[4.63, 1.97, 1.73]],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True,
        ),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder", code_size=9),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
            ),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False,
        )
    ),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500,
        )
    ),
)

work_dir = "outputs/mmdet/base_pointpillars_car"

# Convenience flag for later conditional logic in derived configs.
db_info_path = Path(data_root) / "custom_dbinfos_train.pkl"
