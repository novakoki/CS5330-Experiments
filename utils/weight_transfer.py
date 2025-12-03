"""Convert MMDetection3D PointPillars weights to this custom model."""
import argparse
from collections import OrderedDict
from typing import Dict

import torch


DEFAULT_MAPPING: Dict[str, str] = {
    # Pillar Feature Net
    "pts_voxel_encoder.pfn_layers.0.linear.weight": "pillar_feature_net.pfn.linear.weight",
    "pts_voxel_encoder.pfn_layers.0.linear.bias": "pillar_feature_net.pfn.linear.bias",
    "pts_voxel_encoder.pfn_layers.0.norm.weight": "pillar_feature_net.pfn.bn.weight",
    "pts_voxel_encoder.pfn_layers.0.norm.bias": "pillar_feature_net.pfn.bn.bias",
    "pts_voxel_encoder.pfn_layers.0.norm.running_mean": "pillar_feature_net.pfn.bn.running_mean",
    "pts_voxel_encoder.pfn_layers.0.norm.running_var": "pillar_feature_net.pfn.bn.running_var",
    # Backbone
    "pts_backbone.conv_input.0.weight": "backbone.block.0.weight",
    "pts_backbone.conv_input.0.bias": "backbone.block.0.bias",
    "pts_backbone.conv_input.1.weight": "backbone.block.1.weight",
    "pts_backbone.conv_input.1.bias": "backbone.block.1.bias",
    "pts_backbone.conv_input.1.running_mean": "backbone.block.1.running_mean",
    "pts_backbone.conv_input.1.running_var": "backbone.block.1.running_var",
    "pts_backbone.conv_input.3.weight": "backbone.block.3.weight",
    "pts_backbone.conv_input.3.bias": "backbone.block.3.bias",
    "pts_backbone.conv_input.4.weight": "backbone.block.4.weight",
    "pts_backbone.conv_input.4.bias": "backbone.block.4.bias",
    "pts_backbone.conv_input.4.running_mean": "backbone.block.4.running_mean",
    "pts_backbone.conv_input.4.running_var": "backbone.block.4.running_var",
    # Detection head (cls/box/dir)
    "pts_bbox_head.conv_cls.weight": "conv_cls.weight",
    "pts_bbox_head.conv_cls.bias": "conv_cls.bias",
    "pts_bbox_head.conv_reg.weight": "conv_box.weight",
    "pts_bbox_head.conv_reg.bias": "conv_box.bias",
    "pts_bbox_head.conv_dir_cls.weight": "conv_dir_cls.weight",
    "pts_bbox_head.conv_dir_cls.bias": "conv_dir_cls.bias",
}


def convert_weights(src: str, dst: str, mapping: Dict[str, str] = None):
    mapping = mapping or DEFAULT_MAPPING
    ckpt = torch.load(src, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    converted = OrderedDict()
    missing = []
    for k, v in state.items():
        if k in mapping:
            converted[mapping[k]] = v
        else:
            missing.append(k)
    torch.save(converted, dst)
    print(f"Saved converted weights to {dst}. Converted {len(converted)} tensors.")
    if missing:
        print(f"Skipped {len(missing)} keys (no mapping provided).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to MMDetection3D checkpoint (.pth)")
    parser.add_argument("--dst", required=True, help="Output path for converted weights")
    args = parser.parse_args()
    convert_weights(args.src, args.dst)

