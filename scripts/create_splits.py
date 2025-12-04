"""Generate custom nuScenes scene lists for the partial blob download."""
import argparse
import json
from pathlib import Path
from typing import List

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def collect_scenes(nusc: NuScenes, target_split: List[str]) -> List[str]:
    split_names = set(target_split)
    scene_tokens = []
    for scene in nusc.scene:
        if scene["name"] in split_names:
            scene_tokens.append(scene["token"])
    return scene_tokens


def scene_has_lidar(nusc: NuScenes, scene_token: str) -> bool:
    """Return True if the first sample's LIDAR_TOP file exists on disk."""
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]
    if not sample_token:
        return False
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"].get("LIDAR_TOP")
    if not sd_token:
        return False
    sd = nusc.get("sample_data", sd_token)
    fname = sd.get("filename")
    if not fname:
        return False
    return (Path(nusc.dataroot) / fname).exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/nuscenes", help="nuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="nuScenes dataset version")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Directory to place JSON files")
    parser.add_argument("--small_train_count", type=int, default=30, help="Number of scenes for the small train split")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_scenes = collect_scenes(nusc, splits.train)
    val_scenes = collect_scenes(nusc, splits.val)

    # Filter to scenes whose first LIDAR keyframe file actually exists on disk
    train_scenes_with_lidar = [t for t in train_scenes if scene_has_lidar(nusc, t)]
    val_scenes_with_lidar = [t for t in val_scenes if scene_has_lidar(nusc, t)]
    dropped_train = len(train_scenes) - len(train_scenes_with_lidar)
    dropped_val = len(val_scenes) - len(val_scenes_with_lidar)

    # Constrain to 150 scenes (blob 01/02 subset); adjust here if you downloaded more blobs.
    train_scenes = sorted(train_scenes_with_lidar)[:150]
    small_train = train_scenes[: args.small_train_count]

    with open(Path(args.output_dir) / "custom_full_train.json", "w") as f:
        json.dump(train_scenes, f)
    with open(Path(args.output_dir) / "custom_small_train.json", "w") as f:
        json.dump(small_train, f)
    with open(Path(args.output_dir) / "custom_val.json", "w") as f:
        json.dump(val_scenes, f)

    print(f"Custom-Full-Train scenes: {len(train_scenes)} (dropped {dropped_train} missing LIDAR)")
    print(f"Custom-Small-Train scenes: {len(small_train)}")
    print(f"Custom-Val scenes: {len(val_scenes)} (dropped {dropped_val} missing LIDAR)")


if __name__ == "__main__":
    main()
