"""nuScenes car-only Dataset wrapper for PointPillars."""
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.transforms import DatabaseSampler, apply_augmentations

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud, Box
    from nuscenes.utils.geometry_utils import BoxVisibility
except ImportError as exc:  # pragma: no cover - dependency handled in runtime env
    raise ImportError("nuscenes-devkit is required: pip install nuscenes-devkit") from exc


def filter_by_range(points: np.ndarray, pc_range: List[float]) -> np.ndarray:
    """Crop points to the configured point cloud range."""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[mask]


def boxes_to_numpy(boxes: List[Box]) -> np.ndarray:
    """Convert nuScenes Box objects to ndarray [x, y, z, dx, dy, dz, yaw]."""
    arr = []
    for box in boxes:
        w, l, h = box.wlh
        yaw = box.orientation.yaw_pitch_roll[0]
        arr.append([box.center[0], box.center[1], box.center[2], w, l, h, yaw])
    if len(arr) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


class NuScenesCarDataset(Dataset):
    """Minimal nuScenes Dataset that emits points and car boxes in LiDAR frame."""

    def __init__(
        self,
        data_root: str,
        scene_list_path: str,
        split: str,
        point_cloud_range: List[float],
        voxel_size: List[float],
        max_points_per_voxel: int,
        max_voxels: int,
        class_name: str = "vehicle.car",
        augmentations: Dict = None,
        version: str = "v1.0-trainval",
        copy_paste_db: str = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.class_name = class_name
        self.augmentations = augmentations or {}
        self.version = version
        # lazy=True keeps meta on disk until accessed, saving host RAM
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=False, lazy=True)
        with open(scene_list_path, "r") as f:
            self.scene_tokens = json.load(f)
        self.sample_tokens = self._collect_sample_tokens()
        self.grid_size = self._grid_size()
        copy_paste_path = copy_paste_db or os.path.join(data_root, "dbinfos_car.pkl")
        self.db_sampler = DatabaseSampler(copy_paste_path) if self.augmentations.get("copy_paste") else None
        self.split = split

    def _grid_size(self) -> Tuple[int, int, int]:
        pc_range = self.point_cloud_range
        vs = self.voxel_size
        return (
            int((pc_range[3] - pc_range[0]) / vs[0]),
            int((pc_range[4] - pc_range[1]) / vs[1]),
            int((pc_range[5] - pc_range[2]) / vs[2]),
        )

    def _collect_sample_tokens(self) -> List[str]:
        tokens: List[str] = []
        for scene_token in self.scene_tokens:
            scene = self.nusc.get("scene", scene_token)
            sample_token = scene["first_sample_token"]
            while sample_token:
                tokens.append(sample_token)
                sample = self.nusc.get("sample", sample_token)
                sample_token = sample["next"]
        return tokens

    def _load_points(self, lidar_token: str) -> np.ndarray:
        pointsensor = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.data_root, pointsensor["filename"])
        pc = LidarPointCloud.from_file(lidar_path)
        points = pc.points.transpose(1, 0)  # (N, 4) [x, y, z, intensity]
        return points.astype(np.float32)

    def _load_boxes(self, lidar_token: str) -> List[Box]:
        _, boxes, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE)
        car_boxes = [box for box in boxes if box.name.startswith(self.class_name)]
        return car_boxes

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> Dict:
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        points = self._load_points(lidar_token)
        points = filter_by_range(points, self.point_cloud_range)
        boxes = self._load_boxes(lidar_token)
        gt_boxes = boxes_to_numpy(boxes)
        if self.augmentations:
            points, gt_boxes = apply_augmentations(
                points,
                gt_boxes,
                self.augmentations,
                sampler=self.db_sampler,
                point_cloud_range=self.point_cloud_range,
            )
        # Filter boxes outside range
        if len(gt_boxes) > 0:
            box_mask = (
                (gt_boxes[:, 0] >= self.point_cloud_range[0])
                & (gt_boxes[:, 0] <= self.point_cloud_range[3])
                & (gt_boxes[:, 1] >= self.point_cloud_range[1])
                & (gt_boxes[:, 1] <= self.point_cloud_range[4])
            )
            gt_boxes = gt_boxes[box_mask]
        targets = {
            "gt_boxes": gt_boxes.astype(np.float32),
            "gt_labels": np.ones((len(gt_boxes),), dtype=np.int64),
        }
        meta = {"sample_token": sample_token, "lidar_token": lidar_token, "scene_token": sample["scene_token"]}
        return {"points": points.astype(np.float32), "targets": targets, "meta": meta}


def collate_batch(batch: List[Dict]) -> Dict[str, List[torch.Tensor]]:
    """Custom collate that keeps variable number of points/boxes."""
    collated = {
        "points": [torch.from_numpy(sample["points"]) for sample in batch],
        "gt_boxes": [torch.from_numpy(sample["targets"]["gt_boxes"]) for sample in batch],
        "gt_labels": [torch.from_numpy(sample["targets"]["gt_labels"]) for sample in batch],
        "meta": [sample["meta"] for sample in batch],
    }
    return collated
