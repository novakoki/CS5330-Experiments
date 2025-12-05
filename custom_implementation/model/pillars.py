"""Voxelization and Pillar Feature Net modules."""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Voxelizer:
    """Simple CPU voxelizer for PointPillars."""

    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        max_points_per_voxel: int,
        max_num_voxels: int,
    ):
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.pc_range = np.array(point_cloud_range, dtype=np.float32)
        self.grid_size = np.floor((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size).astype(np.int64)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_num_voxels = max_num_voxels

    def __call__(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = points.device
        voxel_size = torch.as_tensor(self.voxel_size, device=device, dtype=points.dtype)
        pc_range = torch.as_tensor(self.pc_range[:3], device=device, dtype=points.dtype)
        grid_size = torch.as_tensor(self.grid_size, device=device)

        coords = torch.floor((points[:, :3] - pc_range) / voxel_size).long()
        within_bounds = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < grid_size[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < grid_size[1])
            & (coords[:, 2] >= 0)
            & (coords[:, 2] < grid_size[2])
        )
        points = points[within_bounds]
        coords = coords[within_bounds]

        if points.numel() == 0:
            return (
                torch.zeros((0, self.max_points_per_voxel, points.shape[1]), device=device, dtype=points.dtype),
                torch.zeros((0, 3), device=device, dtype=torch.int64),
                torch.zeros((0,), device=device, dtype=torch.int64),
            )

        unique_coors, inverse, counts = torch.unique(coords, dim=0, return_inverse=True, return_counts=True)
        if unique_coors.shape[0] > self.max_num_voxels:
            topk = torch.topk(counts, self.max_num_voxels).indices
            keep_mask = torch.zeros_like(counts, dtype=torch.bool)
            keep_mask[topk] = True
            selected_mask = keep_mask[inverse]
            points = points[selected_mask]
            coords = coords[selected_mask]
            inverse = inverse[selected_mask]
            unique_coors = unique_coors[topk]
            remap = torch.full((counts.shape[0],), -1, device=device, dtype=torch.int64)
            remap[topk] = torch.arange(topk.numel(), device=device, dtype=torch.int64)
            inverse = remap[inverse]

        num_voxels = unique_coors.shape[0]
        voxels = points.new_zeros((num_voxels, self.max_points_per_voxel, points.shape[1]))
        voxel_counts = torch.bincount(inverse, minlength=num_voxels)

        # Sort by voxel index to get stable offsets within each voxel.
        order = torch.argsort(inverse)
        inverse_sorted = inverse[order]
        start = torch.cumsum(voxel_counts, dim=0) - voxel_counts
        offsets_sorted = torch.arange(inverse_sorted.shape[0], device=device) - start[inverse_sorted]
        offsets = torch.zeros_like(inverse_sorted)
        offsets = offsets.scatter(0, torch.arange(offsets_sorted.numel(), device=device), offsets_sorted)
        # Map back to original point order
        offsets_unsorted = torch.zeros_like(inverse)
        offsets_unsorted[order] = offsets_sorted

        mask = offsets_unsorted < self.max_points_per_voxel
        if mask.any():
            voxels[inverse[mask], offsets_unsorted[mask]] = points[mask]
        num_points = torch.minimum(
            voxel_counts,
            torch.full_like(voxel_counts, self.max_points_per_voxel),
        )

        return voxels, unique_coors, num_points


class PFNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=not use_norm)
        self.use_norm = use_norm
        if use_norm:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear(inputs)
        if self.use_norm:
            x = self.bn(x.view(-1, self.out_channels)).view(x.shape)
        x = F.relu(x)
        x_max = torch.max(x, dim=1)[0]
        return x_max


class PillarFeatureNet(nn.Module):
    """Encodes raw pillar voxels into learned features."""

    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        num_filters: int = 64,
        use_norm: bool = True,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = point_cloud_range
        self.pfn = PFNLayer(in_channels=10, out_channels=num_filters, use_norm=use_norm)

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # voxels: (M, P, 4), coords: (M, 3) as (x, y, z) indices
        device = voxels.device
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.unsqueeze(-1).unsqueeze(-1).clamp(min=1)
        f_cluster = voxels[:, :, :3] - points_mean
        # Voxel center offsets
        voxel_size = torch.tensor(self.voxel_size, device=device, dtype=voxels.dtype)
        pc_range = torch.tensor(self.pc_range[:3], device=device, dtype=voxels.dtype)
        coords_f = coords.to(voxels.dtype)
        voxel_centers = (coords_f + 0.5) * voxel_size + pc_range
        f_center = voxels[:, :, :3] - voxel_centers.unsqueeze(1)
        features = torch.cat([voxels, f_cluster, f_center], dim=-1)
        features = self.pfn(features)
        return features


class PointPillarsScatter(nn.Module):
    """Scatter pillar features to a BEV canvas."""

    def __init__(self, output_channels: int, grid_size: Tuple[int, int, int]):
        super().__init__()
        self.nx, self.ny, self.nz = grid_size
        self.output_channels = output_channels

    def forward(self, pillar_features: torch.Tensor, coords: torch.Tensor, batch_size: int) -> torch.Tensor:
        # coords: (M, 4) -> batch_idx, x, y, z
        spatial_feature = pillar_features.new_zeros((batch_size, self.output_channels, self.ny, self.nx))
        coords = coords.long()
        b = coords[:, 0]
        x = coords[:, 1]
        y = coords[:, 2]
        spatial_feature[b, :, y, x] = pillar_features
        return spatial_feature
