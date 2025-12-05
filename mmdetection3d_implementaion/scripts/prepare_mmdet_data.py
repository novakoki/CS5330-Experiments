"""Prepare MMDetection3D-ready nuScenes artifacts for the custom car-only study.

Steps:
1) (Optional) regenerate scene-token splits that match the partial blob download.
2) Build nuScenes info PKLs with ``max_sweeps=0`` (keyframes only).
3) Filter the info PKLs into custom small/full/val variants that align with the
   experiment definitions in ``custom_implementation/docs``.

Run this from the repository root:
    python mmdetection3d_implementaion/scripts/prepare_mmdet_data.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import mmengine
from nuscenes.nuscenes import NuScenes

# Make sure we can import the local MMDetection3D utilities without installing.
ROOT = Path(__file__).resolve().parents[1]
MMDET3D_DIR = ROOT / "mmdetection3d"
sys.path.insert(0, str(MMDET3D_DIR))

from tools.dataset_converters.nuscenes_converter import (  # noqa: E402
    obtain_sensor2top,
)
from tools.dataset_converters.nuscenes_converter import get_available_scenes  # noqa: E402


def load_scene_tokens(split_path: Path) -> List[str]:
    with open(split_path, "r") as f:
        return json.load(f)


def filter_infos_by_scene(infos: List[dict], allowed_scene_tokens: Iterable[str], nusc: NuScenes) -> List[dict]:
    allowed = set(allowed_scene_tokens)
    filtered = []
    for info in infos:
        sample = nusc.get("sample", info["token"])
        if sample["scene_token"] in allowed:
            filtered.append(info)
    return filtered


def ensure_base_infos(data_root: Path, version: str, info_prefix: str, max_sweeps: int, force: bool) -> Dict[str, Path]:
    """Create the standard info files if they do not already exist."""
    train_info = data_root / f"{info_prefix}_infos_train.pkl"
    val_info = data_root / f"{info_prefix}_infos_val.pkl"
    if force or not (train_info.exists() and val_info.exists()):
        create_nuscenes_infos_skip_missing(
            root_path=str(data_root),
            info_prefix=info_prefix,
            version=version,
            max_sweeps=max_sweeps,
        )
    return {"train": train_info, "val": val_info}


def create_nuscenes_infos_skip_missing(
    root_path: str, info_prefix: str, version: str = "v1.0-trainval", max_sweeps: int = 0
):
    """Drop samples with missing lidar files instead of raising."""
    import os

    import mmengine
    import numpy as np
    from pyquaternion import Quaternion
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    if version == "v1.0-trainval":
        train_split_names = splits.train
        val_split_names = splits.val
    elif version == "v1.0-test":
        train_split_names = splits.test
        val_split_names = []
    elif version == "v1.0-mini":
        train_split_names = splits.mini_train
        val_split_names = splits.mini_val
    else:
        raise ValueError("unknown version")

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_split_names))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_split_names))
    train_scenes = set([available_scenes[available_scene_names.index(s)]["token"] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes])

    def build_infos_for_split(target_scenes: set, is_test: bool):
        infos = []
        for sample in mmengine.track_iter_progress(nusc.sample):
            if sample["scene_token"] not in target_scenes:
                continue

            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
            lidar_path = str(lidar_path)
            if not os.path.isabs(lidar_path):
                lidar_path = os.path.join(root_path, lidar_path)
            if not os.path.exists(lidar_path):
                continue

            sd_rec = nusc.get("sample_data", lidar_token)
            cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])

            info = {
                "lidar_path": lidar_path,
                "num_features": 5,
                "token": sample["token"],
                "sweeps": [],
                "cams": dict(),
                "lidar2ego_translation": cs_record["translation"],
                "lidar2ego_rotation": cs_record["rotation"],
                "ego2global_translation": pose_record["translation"],
                "ego2global_rotation": pose_record["rotation"],
                "timestamp": sample["timestamp"],
            }

            l2e_r = info["lidar2ego_rotation"]
            l2e_t = info["lidar2ego_translation"]
            e2g_r = info["ego2global_rotation"]
            e2g_t = info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            if max_sweeps > 0:
                sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                sweeps = []
                while len(sweeps) < max_sweeps:
                    if sd_rec["prev"] == "":
                        break
                    sweep = obtain_sensor2top(nusc, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar")
                    if not os.path.exists(os.path.join(root_path, sweep["data_path"])):
                        break
                    sweeps.append(sweep)
                    sd_rec = nusc.get("sample_data", sd_rec["prev"])
                info["sweeps"] = sweeps

            if not is_test:
                annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
                locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
                velocity = np.array([nusc.box_velocity(token)[:2] for token in sample["anns"]])
                valid_flag = np.array(
                    [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations],
                    dtype=bool,
                ).reshape(-1)
                for i in range(len(boxes)):
                    velo = np.array([*velocity[i], 0.0])
                    velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    velocity[i] = velo[:2]
                names = [b.name for b in boxes]

                info["gt_boxes"] = np.concatenate([locs, dims, rots], axis=1)
                info["gt_names"] = names
                info["gt_velocity"] = velocity.reshape(-1, 2)
                info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
                info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
                info["valid_flag"] = valid_flag

            infos.append(info)
        return infos

    train_infos = build_infos_for_split(train_scenes, is_test=False)
    val_infos = build_infos_for_split(val_scenes, is_test=False)

    metadata = dict(version=version)
    mmengine.dump({"infos": train_infos, "metadata": metadata}, os.path.join(root_path, f"{info_prefix}_infos_train.pkl"))
    mmengine.dump({"infos": val_infos, "metadata": metadata}, os.path.join(root_path, f"{info_prefix}_infos_val.pkl"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/nuscenes"), help="nuScenes data root")
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"), help="Directory containing split JSONs")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="nuScenes dataset version")
    parser.add_argument("--info_prefix", type=str, default="custom", help="Prefix for intermediate info PKLs")
    parser.add_argument("--max_sweeps", type=int, default=0, help="Number of sweeps to keep (0 = keyframes only)")
    parser.add_argument("--force_rebuild", action="store_true", help="Force regeneration of base info PKLs")
    parser.add_argument(
        "--regenerate_splits",
        action="store_true",
        help="Recreate custom split JSONs before filtering infos",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="custom_val_small.json",
        help="Validation scene list filename within splits_dir",
    )
    args = parser.parse_args()

    split_files = {
        "small_train": args.splits_dir / "custom_small_train.json",
        "full_train": args.splits_dir / "custom_full_train.json",
        "val": args.splits_dir / args.val_split,
    }

    if args.regenerate_splits or not all(p.exists() for p in split_files.values()):
        split_script = ROOT / "mmdetection3d_implementaion" / "scripts" / "create_custom_splits.py"
        cmd = [
            sys.executable,
            str(split_script),
            "--data_root",
            str(args.data_root),
            "--version",
            args.version,
            "--output_dir",
            str(args.splits_dir),
        ]
        print(f"Running split regeneration: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=True)

    base_infos = ensure_base_infos(args.data_root, args.version, args.info_prefix, args.max_sweeps, args.force_rebuild)
    train_info = mmengine.load(base_infos["train"])
    val_info = mmengine.load(base_infos["val"])

    small_train_scenes = load_scene_tokens(split_files["small_train"])
    full_train_scenes = load_scene_tokens(split_files["full_train"])
    val_scenes = load_scene_tokens(split_files["val"])

    def save_filtered(name: str, infos: List[dict]):
        out_path = args.data_root / f"custom_infos_{name}.pkl"
        mmengine.dump({"infos": infos, "metadata": train_info["metadata"]}, out_path)
        print(f"Wrote {len(infos)} samples -> {out_path}")

    train_infos = train_info["infos"]
    val_infos = val_info["infos"]

    save_filtered("small_train", filter_infos_by_scene(train_infos, small_train_scenes, nusc))
    save_filtered("full_train", filter_infos_by_scene(train_infos, full_train_scenes, nusc))
    save_filtered("val", filter_infos_by_scene(val_infos, val_scenes, nusc))


if __name__ == "__main__":
    main()
