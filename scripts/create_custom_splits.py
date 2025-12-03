"""Generate custom nuScenes split files for the CS5330 ablation experiments.

This script:
1) Loads the official nuScenes info PKLs (train/val) produced by the
   mmdetection3d dataset tools.
2) Keeps only samples whose LiDAR files actually exist inside
   ``data_root/samples/LIDAR_TOP`` (only the Blobs you downloaded).
3) Saves three filtered PKLs:
   - ``nuscenes_infos_custom_train.pkl`` (~150 scenes from official train)
   - ``nuscenes_infos_custom_val.pkl``   (~30 scenes from official val)
   - ``nuscenes_infos_custom_small.pkl`` first 30 scenes from custom train

Example:
    python scripts/create_custom_splits.py \\
        --data-root data/nuscenes \\
        --train-info nuscenes_infos_train.pkl \\
        --val-info nuscenes_infos_val.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def load_infos(pkl_path: Path) -> Dict:
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def save_infos(pkl_path: Path, payload: Dict) -> None:
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)


def get_lidar_basename(info: Dict) -> str:
    """Return the filename (not path) of the LIDAR_TOP sample inside an info dict."""
    if "lidar_path" in info:
        return os.path.basename(info["lidar_path"])
    if "pts_path" in info:  # legacy key from some converters
        return os.path.basename(info["pts_path"])
    data = info.get("data", {})
    lidar_entry = data.get("LIDAR_TOP")
    if isinstance(lidar_entry, (list, tuple)) and lidar_entry:
        return os.path.basename(lidar_entry[0])
    if isinstance(lidar_entry, str):
        return os.path.basename(lidar_entry)
    raise KeyError("Could not find LIDAR_TOP path inside info record.")


def filter_infos_by_available_files(
    infos: Sequence[Dict], available_files: set[str]
) -> List[Dict]:
    filtered: List[Dict] = []
    for info in infos:
        try:
            lidar_file = get_lidar_basename(info)
        except KeyError:
            continue
        if lidar_file in available_files:
            filtered.append(info)
    return filtered


def build_small_split(infos: Sequence[Dict], max_scenes: int = 30) -> List[Dict]:
    """Take the first N unique scenes (keep all samples within those scenes)."""
    keep_scenes: set[str] = set()
    small_infos: List[Dict] = []
    for info in infos:
        scene_token = info.get("scene_token")
        if scene_token is None:
            continue
        if scene_token in keep_scenes or len(keep_scenes) < max_scenes:
            keep_scenes.add(scene_token)
            small_infos.append(info)
        if len(keep_scenes) >= max_scenes:
            continue
    return small_infos


def describe_split(name: str, infos: Iterable[Dict]) -> str:
    infos = list(infos)
    scenes = {i.get("scene_token") for i in infos if "scene_token" in i}
    return f"{name}: {len(infos)} samples across {len(scenes)} scenes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create custom nuScenes splits.")
    parser.add_argument("--data-root", default="data/nuscenes", type=str)
    parser.add_argument(
        "--train-info",
        default="nuscenes_infos_train.pkl",
        type=str,
        help="Official train info file name (under data_root).",
    )
    parser.add_argument(
        "--val-info",
        default="nuscenes_infos_val.pkl",
        type=str,
        help="Official val info file name (under data_root).",
    )
    parser.add_argument(
        "--small-scenes",
        default=30,
        type=int,
        help="Number of scenes to keep for the small baseline split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    lidar_dir = data_root / "samples" / "LIDAR_TOP"

    train_pkl = data_root / args.train_info
    val_pkl = data_root / args.val_info

    if not lidar_dir.is_dir():
        raise FileNotFoundError(f"Cannot find LiDAR directory: {lidar_dir}")
    if not train_pkl.is_file() or not val_pkl.is_file():
        raise FileNotFoundError(
            f"Missing nuScenes info files: {train_pkl} or {val_pkl}"
        )

    available_files = {p.name for p in lidar_dir.iterdir() if p.is_file()}
    print(f"Found {len(available_files)} LiDAR files in {lidar_dir}")

    train_info = load_infos(train_pkl)
    val_info = load_infos(val_pkl)

    train_infos = train_info.get("infos", train_info)
    val_infos = val_info.get("infos", val_info)

    custom_train_infos = filter_infos_by_available_files(train_infos, available_files)
    custom_val_infos = filter_infos_by_available_files(val_infos, available_files)
    custom_small_infos = build_small_split(custom_train_infos, args.small_scenes)

    out_train = data_root / "nuscenes_infos_custom_train.pkl"
    out_val = data_root / "nuscenes_infos_custom_val.pkl"
    out_small = data_root / "nuscenes_infos_custom_small.pkl"

    payload_template = dict(metadata=train_info.get("metadata"), version=train_info.get("version"))

    save_infos(out_train, dict(payload_template, infos=custom_train_infos))
    save_infos(out_val, dict(payload_template, infos=custom_val_infos))
    save_infos(out_small, dict(payload_template, infos=custom_small_infos))

    print(describe_split("Custom Train", custom_train_infos))
    print(describe_split("Custom Val", custom_val_infos))
    print(describe_split("Custom Small", custom_small_infos))
    print("Done.")


if __name__ == "__main__":
    main()
