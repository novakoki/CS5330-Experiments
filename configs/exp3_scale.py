"""Experiment 3: Full custom train split with copy-paste enabled."""

_base_ = ["./exp2_aug.py"]

train_ann_file = "nuscenes_infos_custom_train.pkl"
train_transforms = dict(
    use_global_rot_scale=True,
    use_random_flip=True,
    use_object_sample=True,
)
train_pipeline = build_train_pipeline(train_transforms)

train_dataloader["dataset"]["ann_file"] = train_ann_file
train_dataloader["dataset"]["pipeline"] = train_pipeline
