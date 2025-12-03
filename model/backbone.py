"""2D BEV backbone for PointPillars."""
import torch.nn as nn
import torch.nn.functional as F


class BaseBEVBackbone(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 64, out_channels: int = 128, use_batchnorm: bool = True):
        super().__init__()
        norm = nn.BatchNorm2d if use_batchnorm else nn.Identity
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=not use_batchnorm),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_batchnorm),
            norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

