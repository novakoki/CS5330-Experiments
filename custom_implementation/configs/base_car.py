"""Base configuration for nuScenes car-only PointPillars experiments."""

# Common configuration dictionary consumed by train.py via runpy.
config = {
    # Reproducibility and logging
    "seed": 42,
    "deterministic": False,
    "log_interval": 10,
    "val_interval": 1,
    "save_every": 1,
    # Moderate dataloader workers for faster host-side preprocessing
    "num_workers": 2,
    "pin_memory": True,
    "prefetch_factor": 2,
    "amp": True,
    "grad_clip": 10.0,
    "work_dir": "outputs/base_car",
    # Dataset settings
    "dataset": {
        "data_root": "data/nuscenes",
        # Scene token lists created by scripts/create_splits.py
        "train_scenes": "data/splits/custom_small_train.json",
        "val_scenes": "data/splits/custom_val.json",
        # Optionally skip loading val split during training to save host RAM
        "load_val": True,
        "class_name": "vehicle.car",
        "max_points_per_voxel": 32,
        "max_voxels": {"train": 12000, "val": 12000},
        # Point cloud range and voxelization grid
        "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        "voxel_size": [0.16, 0.16, 0.2],
        "augmentations": {
            "rotation": False,
            "scaling": False,
            "flip": False,
            "copy_paste": False,
        },
    },
    # Model hyperparameters
    "model": {
        "num_classes": 1,
        "class_names": ["car"],
        # Anchor settings aligned with mmdetection3d nuScenes PointPillars
        "anchor_size": [4.63, 1.97, 1.73],
        "anchor_rotations": [0, 1.5707963267948966],
        "anchor_bottom_heights": [-1.78],
        "pillar_features": 64,
        "backbone_layer_nums": [3, 5, 5],
        "backbone_strides": [2, 2, 2],
        "backbone_out_channels": [64, 128, 256],
        "neck_upsample_strides": [1, 2, 4],
        "neck_out_channels": [128, 128, 128],
        "head_channels": 256,
        "use_batchnorm": True,
        "loss": {
            "alpha": 0.25,
            "gamma": 2.0,
            "bbox_weight": 1.0,
            "dir_weight": 0.2,
        },
    },
    # Optimizer and schedule
    "train": {
        "epochs": 50,
        "batch_size": 2,
        "base_lr": 0.001,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "onecycle_max_lr": 0.003,
        "pct_start": 0.4,
        "div_factor": 10.0,
        "final_div_factor": 100.0,
    },
    # Pretrained weights path for transfer setting (optional)
    "pretrained": None,
}
