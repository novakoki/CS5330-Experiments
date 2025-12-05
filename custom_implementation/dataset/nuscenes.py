"""nuScenes car-only Dataset with minimal lazy metadata loading."""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from dataset.transforms import DatabaseSampler, apply_augmentations


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def create_token_index(records: List[Dict]) -> Dict[str, Dict]:
    return {rec["token"]: rec for rec in records}


def filter_by_range(points: np.ndarray, pc_range: List[float]) -> np.ndarray:
    """Crop points to the configured point cloud range."""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[mask]


def boxes_to_numpy(boxes: List[Dict]) -> np.ndarray:
    """Convert list of box dicts to ndarray [x, y, z, dx, dy, dz, yaw]."""
    arr = []
    for box in boxes:
        q = Quaternion(box["rotation"])
        yaw = q.yaw_pitch_roll[0]
        sz = box["size"]
        ctr = box["translation"]
        # nuScenes stores size as (width, length, height); swap to (length, width, height)
        arr.append([ctr[0], ctr[1], ctr[2], sz[1], sz[0], sz[2], yaw])
    if len(arr) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def transform_box_to_lidar(box: Dict, sd_rec: Dict, cs_rec: Dict, pose_rec: Dict) -> Dict:
    """Transform box dict from global to lidar frame."""
    box = box.copy()
    # Global -> ego
    trans_ego = -np.array(pose_rec["translation"])
    rot_ego = Quaternion(pose_rec["rotation"]).inverse
    box["translation"] = rot_ego.rotate(np.array(box["translation"]) + trans_ego)
    box["rotation"] = (rot_ego * Quaternion(box["rotation"])).elements
    # Ego -> sensor
    trans_cs = -np.array(cs_rec["translation"])
    rot_cs = Quaternion(cs_rec["rotation"]).inverse
    box["translation"] = rot_cs.rotate(np.array(box["translation"]) + trans_cs)
    box["rotation"] = (rot_cs * Quaternion(box["rotation"])).elements
    return box


class NuScenesLite:
    """Minimal loader that keeps only essential tables in memory."""

    def __init__(self, dataroot: str, version: str, scene_tokens: List[str]):
        self.dataroot = Path(dataroot)
        self.version = version
        root = self.dataroot / version
        def log_mem(tag: str):
            try:
                import psutil  # type: ignore
                rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
                print(f"[NuScenesLite][mem] {tag}: {rss_gb:.2f} GB")
            except Exception:
                pass
        # Load base tables (keep only needed keys to save RAM)
        log_mem("before scene.json")
        scenes_raw = load_json(root / "scene.json")
        log_mem("after scene.json")
        self.scenes = {rec["token"]: {"first_sample_token": rec["first_sample_token"]} for rec in scenes_raw}
        samples_raw = load_json(root / "sample.json")
        self.samples = {rec["token"]: {"next": rec["next"]} for rec in samples_raw}
        log_mem("after sample.json")
        sample_data_raw = load_json(root / "sample_data.json")
        log_mem("after sample_data.json")
        self.sample_data = {
            rec["token"]: {
                "token": rec["token"],
                # Keyframe archives may omit is_key_frame; default to True.
                "is_key_frame": rec.get("is_key_frame", True),
                "channel": rec.get("channel", ""),
                "sensor_token": rec.get("sensor_token"),
                "sample_token": rec.get("sample_token"),
                "filename": rec.get("filename"),
                "calibrated_sensor_token": rec.get("calibrated_sensor_token"),
                "ego_pose_token": rec.get("ego_pose_token"),
            }
            for rec in sample_data_raw
        }
        del scenes_raw, samples_raw, sample_data_raw

        # Sensor table is tiny; load for channel inference when missing from sample_data.
        self.sensors = create_token_index(load_json(root / "sensor.json"))
        self.calibrated_sensors = create_token_index(load_json(root / "calibrated_sensor.json"))
        self.ego_poses = create_token_index(load_json(root / "ego_pose.json"))
        log_mem("after calib/ego")

        # Fill missing channel names from sensor table or filename heuristic.
        for sd in self.sample_data.values():
            if sd.get("channel"):
                continue
            sensor_token = sd.get("sensor_token")
            channel = self.sensors.get(sensor_token, {}).get("channel", "")
            if not channel and sd.get("filename"):
                fname = str(sd["filename"]).upper()
                if "LIDAR_TOP" in fname:
                    channel = "LIDAR_TOP"
            sd["channel"] = channel

        self.scene_tokens = scene_tokens
        self.sample_tokens = self._collect_sample_tokens()
        # Pre-index lidar sample_data tokens for our samples
        self.sample_to_lidar: Dict[str, str] = {}
        for sd in self.sample_data.values():
            channel = sd.get("channel", "")
            if channel != "LIDAR_TOP":
                continue
            if not sd.get("is_key_frame", True):
                continue
            stoken = sd.get("sample_token")
            if stoken in self.sample_tokens:
                self.sample_to_lidar[stoken] = sd["token"]
        # Keep only samples that have a LIDAR_TOP keyframe
        self.sample_tokens = [t for t in self.sample_tokens if t in self.sample_to_lidar]
        print(f"[NuScenesLite] collected sample_tokens={len(self.sample_tokens)} with lidar")
        # Filter annotations to our samples and class
        anns = load_json(root / "sample_annotation.json")
        self.sample_annotations: Dict[str, List[Dict]] = {token: [] for token in self.sample_tokens}
        for ann in anns:
            stoken = ann.get("sample_token")
            if stoken in self.sample_annotations:
                self.sample_annotations[stoken].append(
                    {
                        "translation": ann["translation"],
                        "size": ann["size"],
                        "rotation": ann["rotation"],
                        "category_name": ann.get("category_name", ""),
                    }
                )
        log_mem("after annotations")

    def _collect_sample_tokens(self) -> List[str]:
        tokens: List[str] = []
        missing_scenes = 0
        missing_samples = 0
        for scene_token in self.scene_tokens:
            scene = self.scenes.get(scene_token)
            if scene is None:
                missing_scenes += 1
                continue
            sample_token = scene.get("first_sample_token")
            if not sample_token:
                missing_samples += 1
                continue
            while sample_token:
                tokens.append(sample_token)
                sample = self.samples.get(sample_token)
                if sample is None:
                    missing_samples += 1
                    break
                sample_token = sample.get("next")
        if missing_scenes or missing_samples:
            print(
                f"[NuScenesLite] skipped scenes: missing_scenes={missing_scenes} "
                f"missing_samples={missing_samples}"
            )
        return tokens


class NuScenesCarDataset(Dataset):
    """Minimal nuScenes Dataset that emits points and car boxes in LiDAR frame."""

    def __init__(
        self,
        data_root: str,
        scene_list_path: str,
        split: str,
        point_cloud_range: List[float],
        voxel_size: List[float],
        max_points_per_voxel: int,
        max_voxels: int,
        class_name: str = "vehicle.car",
        augmentations: Dict = None,
        version: str = "v1.0-trainval",
        copy_paste_db: str = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.class_name = class_name
        self.augmentations = augmentations or {}
        self.version = version
        with open(scene_list_path, "r") as f:
            self.scene_tokens = json.load(f)
        if len(self.scene_tokens) == 0:
            raise ValueError(
                f"Scene list at {scene_list_path} is empty. Regenerate it with scripts/create_splits.py "
                f"or point the config at the correct splits JSON."
            )
        self.nusc = NuScenesLite(data_root, version, self.scene_tokens)
        self.sample_tokens = self.nusc.sample_tokens
        if len(self.sample_tokens) == 0:
            raise ValueError(
                "No LIDAR keyframes found for the provided scenes. "
                f"Check that {data_root}/{version} contains the nuScenes keyframe metadata and LIDAR files "
                f"and that the scene tokens in {scene_list_path} match the dataset version."
            )
        self.grid_size = self._grid_size()
        copy_paste_path = copy_paste_db or os.path.join(data_root, "dbinfos_car.pkl")
        self.db_sampler = DatabaseSampler(copy_paste_path) if self.augmentations.get("copy_paste") else None
        self.split = split

    def _grid_size(self) -> Tuple[int, int, int]:
        pc_range = self.point_cloud_range
        vs = self.voxel_size
        return (
            int((pc_range[3] - pc_range[0]) / vs[0]),
            int((pc_range[4] - pc_range[1]) / vs[1]),
            int((pc_range[5] - pc_range[2]) / vs[2]),
        )

    def _load_points(self, lidar_token: str) -> np.ndarray:
        sd = self.nusc.sample_data[lidar_token]
        lidar_path = Path(self.data_root) / sd["filename"]
        if not lidar_path.exists():
            raise FileNotFoundError(
                f"LiDAR file not found: {lidar_path}. "
                "Verify that `data_root` points to the extracted nuScenes keyframe blobs "
                "(e.g., v1.0-trainvalXX_keyframes) and that the file layout matches "
                "`data_root/samples/LIDAR_TOP/<file>.pcd.bin`."
            )
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        return points

    def _load_boxes(self, sample_token: str) -> List[Dict]:
        lidar_token = self.nusc.sample_to_lidar[sample_token]
        sd_rec = self.nusc.sample_data[lidar_token]
        cs_rec = self.nusc.calibrated_sensors[sd_rec["calibrated_sensor_token"]]
        pose_rec = self.nusc.ego_poses[sd_rec["ego_pose_token"]]
        anns = []
        for ann in self.nusc.sample_annotations.get(sample_token, []):
            if not ann["category_name"].startswith(self.class_name):
                continue
            box = {
                "translation": ann["translation"],
                "size": ann["size"],
                "rotation": ann["rotation"],
            }
            box = transform_box_to_lidar(box, sd_rec, cs_rec, pose_rec)
            anns.append(box)
        return anns

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> Dict:
        sample_token = self.sample_tokens[idx]
        lidar_token = self.nusc.sample_to_lidar[sample_token]
        points = self._load_points(lidar_token)
        points = filter_by_range(points, self.point_cloud_range)
        boxes = self._load_boxes(sample_token)
        gt_boxes = boxes_to_numpy(boxes)
        if self.augmentations:
            points, gt_boxes = apply_augmentations(
                points,
                gt_boxes,
                self.augmentations,
                sampler=self.db_sampler,
                point_cloud_range=self.point_cloud_range,
            )
        # Filter boxes outside range
        if len(gt_boxes) > 0:
            box_mask = (
                (gt_boxes[:, 0] >= self.point_cloud_range[0])
                & (gt_boxes[:, 0] <= self.point_cloud_range[3])
                & (gt_boxes[:, 1] >= self.point_cloud_range[1])
                & (gt_boxes[:, 1] <= self.point_cloud_range[4])
            )
            gt_boxes = gt_boxes[box_mask]
        targets = {
            "gt_boxes": gt_boxes.astype(np.float32, copy=False),
            "gt_labels": np.ones((len(gt_boxes),), dtype=np.int64),
        }
        meta = {"sample_token": sample_token, "lidar_token": lidar_token}
        return {"points": points.astype(np.float32, copy=False), "targets": targets, "meta": meta}


def collate_batch(batch: List[Dict]) -> Dict[str, List[torch.Tensor]]:
    """Custom collate that keeps variable number of points/boxes."""
    collated = {
        "points": [torch.from_numpy(sample["points"]) for sample in batch],
        "gt_boxes": [torch.from_numpy(sample["targets"]["gt_boxes"]) for sample in batch],
        "gt_labels": [torch.from_numpy(sample["targets"]["gt_labels"]) for sample in batch],
        "meta": [sample["meta"] for sample in batch],
    }
    return collated
