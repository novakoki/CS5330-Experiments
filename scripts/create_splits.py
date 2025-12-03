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
    small_train = sorted(train_scenes)[: args.small_train_count]

    with open(Path(args.output_dir) / "custom_full_train.json", "w") as f:
        json.dump(train_scenes, f)
    with open(Path(args.output_dir) / "custom_small_train.json", "w") as f:
        json.dump(small_train, f)
    with open(Path(args.output_dir) / "custom_val.json", "w") as f:
        json.dump(val_scenes, f)

    print(f"Custom-Full-Train scenes: {len(train_scenes)}")
    print(f"Custom-Small-Train scenes: {len(small_train)}")
    print(f"Custom-Val scenes: {len(val_scenes)}")


if __name__ == "__main__":
    main()

