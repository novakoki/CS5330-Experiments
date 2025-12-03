"""Experiment 2: Small split + geometric augmentations."""

_base_ = ["./exp1_baseline.py"]

train_transforms = dict(
    use_global_rot_scale=True,
    use_random_flip=True,
    use_object_sample=False,
)
train_pipeline = build_train_pipeline(train_transforms)

train_dataloader["dataset"]["pipeline"] = train_pipeline
