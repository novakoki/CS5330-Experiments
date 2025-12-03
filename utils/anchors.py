"""Anchor generation and target assignment for PointPillars."""
import math
from typing import List, Tuple

import numpy as np
import torch


def generate_anchors(
    point_cloud_range: List[float],
    voxel_size: List[float],
    grid_size: Tuple[int, int, int],
    sizes: List[float],
    rotations: List[float],
    bottom_heights: List[float],
    feature_map_stride: int = 2,
) -> torch.Tensor:
    """Generate anchors over the BEV grid."""
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    grid_x, grid_y, _ = grid_size
    out_x = int(math.ceil(grid_x / feature_map_stride))
    out_y = int(math.ceil(grid_y / feature_map_stride))
    stride_x = voxel_size[0] * feature_map_stride
    stride_y = voxel_size[1] * feature_map_stride
    x_centers = np.arange(out_x, dtype=np.float32) * stride_x + x_min + stride_x / 2
    y_centers = np.arange(out_y, dtype=np.float32) * stride_y + y_min + stride_y / 2
    anchors = []
    for z in bottom_heights:
        for rot in rotations:
            for y in y_centers:
                for x in x_centers:
                    anchors.append([x, y, z, sizes[0], sizes[1], sizes[2], rot])
    anchors = np.asarray(anchors, dtype=np.float32)
    return torch.from_numpy(anchors)


def bev_iou(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Compute axis-aligned BEV IoU between anchors and gt boxes (ignores yaw)."""
    if gt_boxes.numel() == 0 or anchors.numel() == 0:
        return torch.zeros((anchors.shape[0], gt_boxes.shape[0]), device=anchors.device)
    a_min = anchors[:, :2] - anchors[:, 3:5] / 2
    a_max = anchors[:, :2] + anchors[:, 3:5] / 2
    b_min = gt_boxes[:, :2] - gt_boxes[:, 3:5] / 2
    b_max = gt_boxes[:, :2] + gt_boxes[:, 3:5] / 2
    max_xy = torch.min(a_max[:, None, :], b_max[None, :, :])
    min_xy = torch.max(a_min[:, None, :], b_min[None, :, :])
    inter = (max_xy - min_xy).clamp(min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    area_a = (a_max[:, 0] - a_min[:, 0]) * (a_max[:, 1] - a_min[:, 1])
    area_b = (b_max[:, 0] - b_min[:, 0]) * (b_max[:, 1] - b_min[:, 1])
    union = area_a[:, None] + area_b - inter_area
    return inter_area / union.clamp(min=1e-6)


def encode_boxes(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Encode gt boxes with respect to anchors."""
    eps = 1e-6
    targets = torch.zeros_like(anchors)
    targets[:, 0:3] = (gt_boxes[:, 0:3] - anchors[:, 0:3]) / anchors[:, 3:6].clamp(min=eps)
    targets[:, 3:6] = torch.log(gt_boxes[:, 3:6] / anchors[:, 3:6].clamp(min=eps))
    targets[:, 6] = torch.sin(gt_boxes[:, 6] - anchors[:, 6])
    return targets


def assign_targets(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    pos_iou_thr: float = 0.6,
    neg_iou_thr: float = 0.45,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign ground truth boxes to anchors.

    Returns:
        labels: (A,) where -1=ignore, 0=negative, 1=positive
        bbox_targets: (A, 7) encoded regression targets
        dir_targets: (A,) direction class labels (0 or 1)
    """
    device = anchors.device
    num_anchors = anchors.shape[0]
    labels = torch.full((num_anchors,), -1, device=device, dtype=torch.int64)
    bbox_targets = torch.zeros((num_anchors, 7), device=device, dtype=torch.float32)
    dir_targets = torch.zeros((num_anchors,), device=device, dtype=torch.int64)
    if gt_boxes.numel() == 0:
        labels[:] = 0
        return labels, bbox_targets, dir_targets
    ious = bev_iou(anchors, gt_boxes)
    max_ious, argmax_ious = ious.max(dim=1)
    labels[max_ious < neg_iou_thr] = 0
    pos_mask = max_ious >= pos_iou_thr
    if pos_mask.any():
        labels[pos_mask] = 1
        matched_gt = gt_boxes[argmax_ious[pos_mask]]
        matched_anchors = anchors[pos_mask]
        bbox_targets[pos_mask] = encode_boxes(matched_gt, matched_anchors)
        dir_targets[pos_mask] = ((matched_gt[:, 6] - matched_anchors[:, 6]) > 0).long()
    # Ensure each gt has at least one positive anchor
    gt_max_ious, gt_arg = ious.max(dim=0)
    labels[gt_arg] = 1
    matched_gt = gt_boxes
    matched_anchors = anchors[gt_arg]
    bbox_targets[gt_arg] = encode_boxes(matched_gt, matched_anchors)
    dir_targets[gt_arg] = ((matched_gt[:, 6] - matched_anchors[:, 6]) > 0).long()
    return labels, bbox_targets, dir_targets
