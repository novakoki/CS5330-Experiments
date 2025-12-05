"""Generate nuScenes scene token splits for car-only PointPillars experiments.

This mirrors the split logic in ``custom_implementation/docs`` but is intended
to be used with MMDetection3D configs. We assume the partial nuScenes download
containing only keyframes (blobs 01/02).
"""
import argparse
import json
from pathlib import Path
from typing import List, Sequence

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def collect_scenes(nusc: NuScenes, target_split: Sequence[str]) -> List[str]:
    """Return scene tokens that belong to the provided official split names."""
    split_names = set(target_split)
    scene_tokens = []
    for scene in nusc.scene:
        if scene["name"] in split_names:
            scene_tokens.append(scene["token"])
    return scene_tokens


def scene_has_all_lidar(nusc: NuScenes, scene_token: str) -> bool:
    """Return True if every LIDAR_TOP file in the scene exists on disk."""
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        sd_token = sample["data"].get("LIDAR_TOP")
        if not sd_token:
            return False
        sd = nusc.get("sample_data", sd_token)
        fname = sd.get("filename")
        if not fname or not (Path(nusc.dataroot) / fname).exists():
            return False
        sample_token = sample.get("next")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/nuscenes", help="nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="nuScenes dataset version")
    parser.add_argument(
        "--output_dir", type=str, default="data/splits", help="Directory to place JSON files"
    )
    parser.add_argument("--small_train_count", type=int, default=30, help="Number of scenes for the small train split")
    parser.add_argument(
        "--limit_val_scenes",
        type=int,
        default=30,
        help="Number of scenes for a reduced val split (also saves the full list)",
    )
    parser.add_argument(
        "--limit_train_scenes",
        type=int,
        default=150,
        help="Maximum scenes from the official train list to keep (controls blob subset size)",
    )
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_scenes = collect_scenes(nusc, splits.train)
    val_scenes = collect_scenes(nusc, splits.val)

    train_scenes_with_lidar = [t for t in train_scenes if scene_has_all_lidar(nusc, t)]
    val_scenes_with_lidar = [t for t in val_scenes if scene_has_all_lidar(nusc, t)]
    dropped_train = len(train_scenes) - len(train_scenes_with_lidar)
    dropped_val = len(val_scenes) - len(val_scenes_with_lidar)

    # Constrain to the partial blob download (01/02). Adjust the limit if more blobs are present.
    train_scenes = sorted(train_scenes_with_lidar)[: args.limit_train_scenes]
    small_train = train_scenes[: args.small_train_count]

    with open(Path(args.output_dir) / "custom_full_train.json", "w") as f:
        json.dump(train_scenes, f)
    with open(Path(args.output_dir) / "custom_small_train.json", "w") as f:
        json.dump(small_train, f)
    with open(Path(args.output_dir) / "custom_val.json", "w") as f:
        json.dump(val_scenes, f)

    val_small = sorted(val_scenes)[: args.limit_val_scenes]
    with open(Path(args.output_dir) / "custom_val_small.json", "w") as f:
        json.dump(val_small, f)

    print(f"Custom-Full-Train scenes: {len(train_scenes)} (dropped {dropped_train} missing LIDAR)")
    print(f"Custom-Small-Train scenes: {len(small_train)}")
    print(f"Custom-Val scenes: {len(val_scenes)} (dropped {dropped_val} missing LIDAR)")
    print(f"Custom-Val-Small scenes: {len(val_small)} (first {args.limit_val_scenes} sorted by name)")


if __name__ == "__main__":
    main()
