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
        points_np = points.cpu().numpy()
        voxel_size = self.voxel_size
        coors = np.floor((points_np[:, :3] - self.pc_range[:3]) / voxel_size).astype(np.int64)
        within_bounds = (
            (coors[:, 0] >= 0)
            & (coors[:, 0] < self.grid_size[0])
            & (coors[:, 1] >= 0)
            & (coors[:, 1] < self.grid_size[1])
            & (coors[:, 2] >= 0)
            & (coors[:, 2] < self.grid_size[2])
        )
        points_np = points_np[within_bounds]
        coors = coors[within_bounds]
        voxel_dict = {}
        voxel_num = 0
        for point, coord in zip(points_np, coors):
            coord_key = (coord[0], coord[1], coord[2])
            if coord_key not in voxel_dict:
                if voxel_num >= self.max_num_voxels:
                    continue
                voxel_dict[coord_key] = []
                voxel_num += 1
            if len(voxel_dict[coord_key]) < self.max_points_per_voxel:
                voxel_dict[coord_key].append(point)
        voxels = np.zeros((len(voxel_dict), self.max_points_per_voxel, points_np.shape[1]), dtype=np.float32)
        coords = np.zeros((len(voxel_dict), 3), dtype=np.int64)
        num_points = np.zeros((len(voxel_dict),), dtype=np.int64)
        for i, (coord, pts) in enumerate(voxel_dict.items()):
            coords[i] = np.array(coord)
            num = len(pts)
            voxels[i, :num] = np.array(pts, dtype=np.float32)
            num_points[i] = num
        return (
            torch.from_numpy(voxels),
            torch.from_numpy(coords),
            torch.from_numpy(num_points),
        )


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
        for i in range(coords.shape[0]):
            b, x, y, _ = coords[i]
            spatial_feature[b, :, y, x] = pillar_features[i]
        return spatial_feature
