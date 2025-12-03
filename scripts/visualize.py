"""Visualize BEV predictions vs ground truth."""
import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader

from dataset.nuscenes import NuScenesCarDataset, collate_batch
from model.pointpillars import PointPillars


def load_config(path: str) -> Dict:
    cfg = {}
    exec(open(path, "r").read(), cfg)
    return cfg["config"]


def plot_bev(points, gt_boxes, pred_boxes, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], s=0.3, c="gray", alpha=0.5)
    for box in gt_boxes:
        rect = patches.Rectangle(
            (box[0] - box[3] / 2, box[1] - box[4] / 2),
            box[3],
            box[4],
            linewidth=1,
            edgecolor="green",
            facecolor="none",
        )
        ax.add_patch(rect)
    for box in pred_boxes:
        rect = patches.Rectangle(
            (box[0] - box[3] / 2, box[1] - box[4] / 2),
            box[3],
            box[4],
            linewidth=1,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Green: GT, Red: Pred")
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_aspect("equal")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file")
    parser.add_argument("--output_dir", default="visualizations", help="Directory to save plots")
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    dataset = NuScenesCarDataset(
        data_root=dataset_cfg["data_root"],
        scene_list_path=dataset_cfg["val_scenes"],
        split="val",
        point_cloud_range=dataset_cfg["point_cloud_range"],
        voxel_size=dataset_cfg["voxel_size"],
        max_points_per_voxel=dataset_cfg["max_points_per_voxel"],
        max_voxels=dataset_cfg["max_voxels"]["val"],
        class_name=dataset_cfg["class_name"],
        augmentations={"rotation": False, "scaling": False, "flip": False, "copy_paste": False},
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_batch)
    model_cfg = cfg["model"]
    model_cfg["max_points_per_voxel"] = dataset_cfg["max_points_per_voxel"]
    model_cfg["max_voxels"] = dataset_cfg["max_voxels"]["val"]
    model = PointPillars(
        grid_size=dataset.grid_size,
        voxel_size=dataset_cfg["voxel_size"],
        point_cloud_range=dataset_cfg["point_cloud_range"],
        model_cfg=model_cfg,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
    model.eval()

    for idx, batch in enumerate(loader):
        if idx >= args.num_samples:
            break
        outputs = model(batch)
        dets = model.predict(outputs, score_thresh=0.3, nms_thresh=0.2)[0]
        plot_bev(
            batch["points"][0].numpy(),
            batch["gt_boxes"][0].numpy(),
            dets["boxes"].cpu().numpy(),
            Path(args.output_dir) / f"sample_{idx}.png",
        )
        print(f"Saved visualization for sample {idx}")


if __name__ == "__main__":
    main()

