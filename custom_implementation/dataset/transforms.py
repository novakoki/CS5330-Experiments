"""Lightweight geometric and copy-paste augmentations for nuScenes LiDAR."""
import math
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np


def limit_period(val: np.ndarray, offset: float = 0.0, period: float = 2 * math.pi) -> np.ndarray:
    """Wrap angles into a period with the given offset."""
    return val - np.floor((val - offset) / period) * period


def random_rotation(points: np.ndarray, boxes: np.ndarray, rot_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random yaw rotation around Z to points and boxes."""
    angle = np.random.uniform(*rot_range)
    c, s = np.cos(angle), np.sin(angle)
    rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    rotated_points = points.copy()
    rotated_points[:, :3] = rotated_points[:, :3] @ rot_mat.T
    rotated_boxes = boxes.copy()
    rotated_boxes[:, :3] = rotated_boxes[:, :3] @ rot_mat.T
    rotated_boxes[:, -1] = limit_period(rotated_boxes[:, -1] + angle, offset=0.5 * math.pi)
    return rotated_points, rotated_boxes


def random_scaling(points: np.ndarray, boxes: np.ndarray, scale_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly scale points and box centers/sizes."""
    scale = np.random.uniform(*scale_range)
    scaled_points = points.copy()
    scaled_points[:, :3] *= scale
    scaled_boxes = boxes.copy()
    scaled_boxes[:, :6] *= scale
    return scaled_points, scaled_boxes


def random_flip(points: np.ndarray, boxes: np.ndarray, axis: str = "x") -> Tuple[np.ndarray, np.ndarray]:
    """Flip points and boxes along an axis (x or y)."""
    flipped_points = points.copy()
    flipped_boxes = boxes.copy()
    if axis == "x":
        flipped_points[:, 0] = -flipped_points[:, 0]
        flipped_boxes[:, 0] = -flipped_boxes[:, 0]
        flipped_boxes[:, -1] = limit_period(np.pi - flipped_boxes[:, -1], offset=0.5 * np.pi)
    elif axis == "y":
        flipped_points[:, 1] = -flipped_points[:, 1]
        flipped_boxes[:, 1] = -flipped_boxes[:, 1]
        flipped_boxes[:, -1] = limit_period(-flipped_boxes[:, -1], offset=0.5 * np.pi)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")
    return flipped_points, flipped_boxes


def boxes_bev_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute BEV IoU for axis-aligned boxes (assumes yaw already aligned by sampling)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    boxes_a_corners = np.stack(
        [
            boxes_a[:, 0] - boxes_a[:, 3] / 2,
            boxes_a[:, 1] - boxes_a[:, 4] / 2,
            boxes_a[:, 0] + boxes_a[:, 3] / 2,
            boxes_a[:, 1] + boxes_a[:, 4] / 2,
        ],
        axis=1,
    )
    boxes_b_corners = np.stack(
        [
            boxes_b[:, 0] - boxes_b[:, 3] / 2,
            boxes_b[:, 1] - boxes_b[:, 4] / 2,
            boxes_b[:, 0] + boxes_b[:, 3] / 2,
            boxes_b[:, 1] + boxes_b[:, 4] / 2,
        ],
        axis=1,
    )
    max_xy = np.minimum(boxes_a_corners[:, None, 2:], boxes_b_corners[:, 2:])
    min_xy = np.maximum(boxes_a_corners[:, None, :2], boxes_b_corners[:, :2])
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=None)
    inter_area = inter[..., 0] * inter[..., 1]
    area_a = (boxes_a_corners[:, 2] - boxes_a_corners[:, 0]) * (boxes_a_corners[:, 3] - boxes_a_corners[:, 1])
    area_b = (boxes_b_corners[:, 2] - boxes_b_corners[:, 0]) * (boxes_b_corners[:, 3] - boxes_b_corners[:, 1])
    union = area_a[:, None] + area_b - inter_area
    return inter_area / np.clip(union, a_min=1e-6, a_max=None)


class DatabaseSampler:
    """Simple GT sampling (copy-paste) helper.

    Expects a pickle file with a list of entries, each containing:
        - box: ndarray (7,) [x, y, z, dx, dy, dz, yaw]
        - points: ndarray (N, 4) local point cloud cropped to the box
    """

    def __init__(self, info_path: str, max_paste: int = 15, iou_threshold: float = 0.01):
        self.info_path = info_path
        self.max_paste = max_paste
        self.iou_threshold = iou_threshold
        self.enabled = os.path.exists(info_path)
        self.db_infos: List[Dict] = []
        if self.enabled:
            with open(info_path, "rb") as f:
                self.db_infos = pickle.load(f)

    def sample(self, num: int) -> List[Dict]:
        if not self.enabled or len(self.db_infos) == 0:
            return []
        num = min(num, len(self.db_infos))
        return random.sample(self.db_infos, num)

    def __call__(
        self,
        points: np.ndarray,
        boxes: np.ndarray,
        point_cloud_range: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return points, boxes
        pasted = 0
        candidate_boxes: List[np.ndarray] = []
        candidate_points: List[np.ndarray] = []
        for entry in self.sample(self.max_paste):
            new_box = entry["box"].copy()
            # Randomly place box within the horizontal range while keeping yaw
            x_min, y_min, _, x_max, y_max, _ = point_cloud_range
            new_box[0] = np.random.uniform(x_min + new_box[3] / 2, x_max - new_box[3] / 2)
            new_box[1] = np.random.uniform(y_min + new_box[4] / 2, y_max - new_box[4] / 2)
            new_box = new_box.reshape(1, -1)
            if len(boxes) > 0:
                iou = boxes_bev_iou(new_box, boxes)
                if float(iou.max()) > self.iou_threshold:
                    continue
            candidate_boxes.append(new_box)
            pts = entry["points"].copy()
            pts[:, :3] += new_box[0, :3]
            candidate_points.append(pts)
            pasted += 1
        if pasted == 0:
            return points, boxes
        all_boxes = np.concatenate([boxes] + candidate_boxes, axis=0)
        all_points = np.concatenate([points] + candidate_points, axis=0)
        return all_points, all_boxes


def apply_augmentations(
    points: np.ndarray,
    boxes: np.ndarray,
    aug_cfg: Dict,
    sampler: DatabaseSampler = None,
    point_cloud_range: List[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply configured augmentations to a single frame."""
    if aug_cfg.get("rotation", False):
        points, boxes = random_rotation(points, boxes, rot_range=(-math.pi / 4, math.pi / 4))
    if aug_cfg.get("scaling", False):
        points, boxes = random_scaling(points, boxes, scale_range=(0.95, 1.05))
    if aug_cfg.get("flip", False):
        if random.random() < 0.5:
            points, boxes = random_flip(points, boxes, axis="x")
        if random.random() < 0.5:
            points, boxes = random_flip(points, boxes, axis="y")
    if aug_cfg.get("copy_paste", False) and sampler is not None and point_cloud_range is not None:
        points, boxes = sampler(points, boxes, point_cloud_range)
    return points, boxes
