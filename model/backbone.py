"""BEV backbone and FPN neck for PointPillars."""
import torch
import torch.nn as nn


class BEVBackbone(nn.Module):
    """Multi-stage 2D CNN that returns intermediate feature maps."""

    def __init__(self, in_channels: int, layer_nums, layer_strides, out_channels, use_batchnorm: bool = True):
        super().__init__()
        assert len(layer_nums) == len(layer_strides) == len(out_channels)
        norm = nn.BatchNorm2d if use_batchnorm else nn.Identity
        blocks = []
        c_in = in_channels
        for i in range(len(layer_nums)):
            cur_layers = [
                nn.Conv2d(c_in, out_channels[i], kernel_size=3, stride=layer_strides[i], padding=1, bias=not use_batchnorm),
                norm(out_channels[i]),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_nums[i]):
                cur_layers.extend(
                    [
                        nn.Conv2d(out_channels[i], out_channels[i], kernel_size=3, padding=1, bias=not use_batchnorm),
                        norm(out_channels[i]),
                        nn.ReLU(inplace=True),
                    ]
                )
            blocks.append(nn.Sequential(*cur_layers))
            c_in = out_channels[i]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs


class FPNNeck(nn.Module):
    """Upsample multi-scale features and fuse into a single BEV map."""

    def __init__(self, in_channels, upsample_strides, out_channels, fuse_channels: int = None, use_batchnorm: bool = True):
        super().__init__()
        assert len(in_channels) == len(upsample_strides) == len(out_channels)
        norm = nn.BatchNorm2d if use_batchnorm else nn.Identity
        self.deblocks = nn.ModuleList()
        for c_in, stride, c_out in zip(in_channels, upsample_strides, out_channels):
            if stride >= 1:
                block = nn.Sequential(
                    nn.ConvTranspose2d(
                        c_in,
                        c_out,
                        kernel_size=stride,
                        stride=stride,
                        bias=not use_batchnorm,
                    ),
                    norm(c_out),
                    nn.ReLU(inplace=True),
                )
            else:
                stride_inv = int(round(1 / stride))
                block = nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=stride_inv, stride=stride_inv, bias=not use_batchnorm),
                    norm(c_out),
                    nn.ReLU(inplace=True),
                )
            self.deblocks.append(block)
        self.fuse_channels = fuse_channels or sum(out_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(sum(out_channels), self.fuse_channels, kernel_size=1, bias=not use_batchnorm),
            norm(self.fuse_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outs = []
        for i in range(len(x)):
            outs.append(self.deblocks[i](x[i]))
        fused = torch.cat(outs, dim=1)
        return self.fuse(fused)
