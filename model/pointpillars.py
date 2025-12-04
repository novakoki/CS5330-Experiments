"""Lightweight PointPillars model for nuScenes car detection."""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import BEVBackbone, FPNNeck
from model.pillars import PillarFeatureNet, PointPillarsScatter, Voxelizer
from utils.anchors import assign_targets, generate_anchors
from utils.evaluation import nms_bev


class PointPillars(nn.Module):
    def __init__(self, grid_size, voxel_size, point_cloud_range, model_cfg):
        super().__init__()
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.model_cfg = model_cfg
        self.num_classes = model_cfg.get("num_classes", 1)
        self.num_anchors_per_loc = len(model_cfg["anchor_rotations"]) * len(model_cfg["anchor_bottom_heights"])
        self.feature_map_stride = model_cfg.get("feature_map_stride", 2)

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=model_cfg.get("max_points_per_voxel", 32),
            max_num_voxels=model_cfg.get("max_voxels", 12000),
        )
        self.pillar_feature_net = PillarFeatureNet(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            num_filters=model_cfg.get("pillar_features", 64),
            use_norm=model_cfg.get("use_batchnorm", True),
        )
        self.scatter = PointPillarsScatter(
            output_channels=model_cfg.get("pillar_features", 64),
            grid_size=grid_size,
        )
        self.backbone = BEVBackbone(
            in_channels=model_cfg.get("pillar_features", 64),
            layer_nums=model_cfg.get("backbone_layer_nums", [3, 5, 5]),
            layer_strides=model_cfg.get("backbone_strides", [2, 2, 2]),
            out_channels=model_cfg.get("backbone_out_channels", [64, 128, 256]),
            use_batchnorm=model_cfg.get("use_batchnorm", True),
        )
        self.neck = FPNNeck(
            in_channels=model_cfg.get("backbone_out_channels", [64, 128, 256]),
            upsample_strides=model_cfg.get("neck_upsample_strides", [1, 2, 4]),
            out_channels=model_cfg.get("neck_out_channels", [128, 128, 128]),
            fuse_channels=model_cfg.get("head_channels", 256),
            use_batchnorm=model_cfg.get("use_batchnorm", True),
        )
        head_in_channels = model_cfg.get("head_channels", 256)
        self.conv_cls = nn.Conv2d(head_in_channels, self.num_anchors_per_loc * self.num_classes, kernel_size=1)
        self.conv_box = nn.Conv2d(head_in_channels, self.num_anchors_per_loc * 7, kernel_size=1)
        self.conv_dir_cls = nn.Conv2d(head_in_channels, self.num_anchors_per_loc * 2, kernel_size=1)

        self.register_buffer(
            "anchors",
            generate_anchors(
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                grid_size=grid_size,
                sizes=model_cfg["anchor_size"],
                rotations=model_cfg["anchor_rotations"],
                bottom_heights=model_cfg["anchor_bottom_heights"],
                feature_map_stride=self.feature_map_stride,
            ),
            persistent=False,
        )

    def voxelize(self, points_list: List[torch.Tensor]):
        voxels_list = []
        coords_list = []
        num_points_list = []
        for batch_idx, points in enumerate(points_list):
            voxels, coords, num_points = self.voxelizer(points)
            if len(voxels) == 0:
                continue
            batch_indices = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int64)
            coords_with_batch = torch.cat([batch_indices, coords], dim=1)
            voxels_list.append(voxels)
            coords_list.append(coords_with_batch)
            num_points_list.append(num_points)
        if len(voxels_list) == 0:
            raise RuntimeError("No voxels generated from input point clouds.")
        return (
            torch.cat(voxels_list, dim=0),
            torch.cat(coords_list, dim=0),
            torch.cat(num_points_list, dim=0),
            len(points_list),
        )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        voxels, coords, num_points, batch_size = self.voxelize(batch["points"])
        device = next(self.parameters()).device
        voxels = voxels.to(device)
        coords = coords.to(device)
        num_points = num_points.to(device)
        pillar_features = self.pillar_feature_net(voxels, num_points, coords[:, 1:])
        spatial_features = self.scatter(pillar_features, coords, batch_size)
        feats_multi = self.backbone(spatial_features)
        feats = self.neck(feats_multi)
        cls_preds = self.conv_cls(feats)
        box_preds = self.conv_box(feats)
        dir_preds = self.conv_dir_cls(feats)
        return {"cls_preds": cls_preds, "box_preds": box_preds, "dir_preds": dir_preds}

    @staticmethod
    def _flatten_preds(cls_preds, box_preds, dir_preds, num_classes):
        batch_size = cls_preds.shape[0]
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, num_classes)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        return cls_preds, box_preds, dir_preds

    @staticmethod
    def decode_boxes(box_preds: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        anchors = anchors.to(box_preds.device)
        boxes = torch.zeros_like(box_preds)
        boxes[:, 3:6] = torch.exp(box_preds[:, 3:6]) * anchors[:, 3:6]
        boxes[:, :3] = box_preds[:, :3] * anchors[:, 3:6] + anchors[:, :3]
        boxes[:, 6] = box_preds[:, 6] + anchors[:, 6]
        return boxes

    def loss(
        self,
        preds: Dict[str, torch.Tensor],
        batch: Dict,
        loss_cfg: Dict,
    ) -> Dict[str, torch.Tensor]:
        cls_preds, box_preds, dir_preds = self._flatten_preds(preds["cls_preds"], preds["box_preds"], preds["dir_preds"], self.num_classes)
        device = cls_preds.device
        anchors = self.anchors.to(device)
        total_cls, total_box, total_dir, total_pos = 0.0, 0.0, 0.0, 0
        for b in range(cls_preds.shape[0]):
            gt_boxes = batch["gt_boxes"][b].to(device)
            labels, bbox_targets, dir_targets = assign_targets(anchors, gt_boxes)
            positive_mask = labels == 1
            neg_mask = labels == 0
            num_pos = positive_mask.sum().clamp(min=1)
            total_pos += int(num_pos.item())
            cls_target = torch.zeros_like(cls_preds[b])
            cls_target[positive_mask, 0] = 1.0
            alpha = loss_cfg.get("alpha", 0.25)
            gamma = loss_cfg.get("gamma", 2.0)
            cls_loss = self.focal_loss(cls_preds[b].squeeze(-1), cls_target.squeeze(-1), positive_mask | neg_mask, alpha, gamma)
            if positive_mask.any():
                reg_loss = F.smooth_l1_loss(box_preds[b][positive_mask], bbox_targets[positive_mask], reduction="sum") / num_pos
            else:
                reg_loss = torch.tensor(0.0, device=device)
            if positive_mask.any():
                dir_pred = dir_preds[b][positive_mask]
                dir_loss = F.cross_entropy(dir_pred, dir_targets[positive_mask], reduction="sum") / num_pos
            else:
                dir_loss = torch.tensor(0.0, device=device)
            total_cls += cls_loss
            total_box += reg_loss * loss_cfg.get("bbox_weight", 2.0)
            total_dir += dir_loss * loss_cfg.get("dir_weight", 0.2)
        loss_dict = {
            "cls_loss": total_cls / cls_preds.shape[0],
            "box_loss": total_box / cls_preds.shape[0],
            "dir_loss": total_dir / cls_preds.shape[0],
        }
        loss_dict["total_loss"] = loss_dict["cls_loss"] + loss_dict["box_loss"] + loss_dict["dir_loss"]
        return loss_dict

    @staticmethod
    def focal_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = ce * ((1 - p_t) ** gamma)
        alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
        focal = focal * alpha_factor
        focal = focal[mask]
        return focal.sum() / mask.sum().clamp(min=1)

    def predict(self, preds: Dict[str, torch.Tensor], score_thresh: float = 0.3, nms_thresh: float = 0.2):
        cls_preds, box_preds, dir_preds = self._flatten_preds(preds["cls_preds"], preds["box_preds"], preds["dir_preds"], self.num_classes)
        anchors = self.anchors.to(cls_preds.device)
        batch_boxes: List[Dict[str, torch.Tensor]] = []
        for b in range(cls_preds.shape[0]):
            scores = torch.sigmoid(cls_preds[b].squeeze(-1))
            boxes = self.decode_boxes(box_preds[b], anchors)
            dir_labels = dir_preds[b].argmax(dim=-1)
            boxes[:, 6] = boxes[:, 6] + (dir_labels * torch.pi)
            mask = scores > score_thresh
            scores = scores[mask]
            boxes = boxes[mask]
            if scores.numel() == 0:
                batch_boxes.append({"boxes": torch.zeros((0, 7), device=cls_preds.device), "scores": torch.zeros((0,), device=cls_preds.device)})
                continue
            keep = nms_bev(boxes, scores, iou_threshold=nms_thresh)
            batch_boxes.append({"boxes": boxes[keep], "scores": scores[keep]})
        return batch_boxes
