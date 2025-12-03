"""Experiment 1: Baseline on the small split with no augmentation."""

_base_ = ["./base_car.py"]

train_ann_file = "nuscenes_infos_custom_small.pkl"
train_transforms = dict(
    use_global_rot_scale=False,
    use_random_flip=False,
    use_object_sample=False,
)
train_pipeline = build_train_pipeline(train_transforms)

train_dataloader["dataset"]["ann_file"] = train_ann_file
train_dataloader["dataset"]["pipeline"] = train_pipeline
