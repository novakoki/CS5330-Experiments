backend_args = None
base_train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='custom_infos_small_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(img='', pts='samples/LIDAR_TOP', sweeps=''),
        data_root='data/nuscenes/',
        metainfo=dict(classes=('car', )),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                point_cloud_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                type='ObjectRangeFilter'),
            dict(classes=('car', ), type='ObjectNameFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        type='NuScenesDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
class_names = ('car', )
data_prefix = dict(img='', pts='samples/LIDAR_TOP', sweeps='')
data_root = 'data/nuscenes/'
db_info_path = data / nuscenes / custom_dbinfos_train.pkl
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 50
metainfo = dict(classes=('car', ))
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            max_voxels=(
                12000,
                12000,
            ),
            point_cloud_range=[
                -51.2,
                -51.2,
                -5.0,
                51.2,
                51.2,
                3.0,
            ],
            voxel_size=[
                0.16,
                0.16,
                0.2,
            ])),
    pts_backbone=dict(
        in_channels=64,
        layer_nums=[
            3,
            5,
            5,
        ],
        layer_strides=[
            2,
            2,
            2,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='naiveSyncBN2d'),
        out_channels=[
            64,
            128,
            256,
        ],
        type='SECOND'),
    pts_bbox_head=dict(
        anchor_generator=dict(
            custom_values=[
                0,
                0,
            ],
            ranges=[
                [
                    -51.2,
                    -51.2,
                    -1.78,
                    51.2,
                    51.2,
                    -1.78,
                ],
            ],
            reshape_out=True,
            rotations=[
                0,
                1.57,
            ],
            scales=[
                1,
                2,
                4,
            ],
            sizes=[
                [
                    4.63,
                    1.97,
                    1.73,
                ],
            ],
            type='AlignedAnchor3DRangeGenerator'),
        assigner_per_size=False,
        bbox_coder=dict(code_size=9, type='DeltaXYZWLHRBBoxCoder'),
        diff_rad_by_sin=True,
        dir_offset=-0.7854,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=1.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_dir=dict(
            loss_weight=0.2, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        num_classes=1,
        type='Anchor3DHead',
        use_direction_classifier=True),
    pts_middle_encoder=dict(
        in_channels=64, output_shape=[
            640,
            640,
        ], type='PointPillarsScatter'),
    pts_neck=dict(
        act_cfg=dict(type='ReLU'),
        in_channels=[
            64,
            128,
            256,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='naiveSyncBN2d'),
        num_outs=3,
        out_channels=256,
        start_level=0,
        type='mmdet.FPN'),
    pts_voxel_encoder=dict(
        feat_channels=[
            64,
            64,
        ],
        in_channels=5,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='naiveSyncBN1d'),
        point_cloud_range=[
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ],
        type='HardVFE',
        voxel_size=[
            0.16,
            0.16,
            0.2,
        ],
        with_cluster_center=True,
        with_distance=False,
        with_voxel_center=True),
    test_cfg=dict(
        pts=dict(
            max_num=500,
            min_bbox_size=0,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            use_rotate_nms=True)),
    train_cfg=dict(
        pts=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.6,
                type='Max3DIoUAssigner'),
            code_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                0.2,
            ],
            debug=False,
            pos_weight=-1)),
    type='MVXFasterRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.1, type='LinearLR'),
    dict(
        T_max=50,
        begin=0,
        by_epoch=True,
        end=50,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
point_cloud_range = [
    -51.2,
    -51.2,
    -5.0,
    51.2,
    51.2,
    3.0,
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='custom_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(img='', pts='samples/LIDAR_TOP', sweeps=''),
        data_root='data/nuscenes/',
        metainfo=dict(classes=('car', )),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                point_cloud_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/nuscenes/custom_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        point_cloud_range=[
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ],
        type='PointsRangeFilter'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='custom_infos_small_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(img='', pts='samples/LIDAR_TOP', sweeps=''),
        data_root='data/nuscenes/',
        metainfo=dict(classes=('car', )),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                point_cloud_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                type='ObjectRangeFilter'),
            dict(classes=('car', ), type='ObjectNameFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        type='NuScenesDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        point_cloud_range=[
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ],
        type='ObjectRangeFilter'),
    dict(classes=('car', ), type='ObjectNameFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='custom_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(img='', pts='samples/LIDAR_TOP', sweeps=''),
        data_root='data/nuscenes/',
        metainfo=dict(classes=('car', )),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                point_cloud_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/nuscenes/custom_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
voxel_size = [
    0.16,
    0.16,
    0.2,
]
work_dir = 'outputs/mmdet/exp1_baseline'
