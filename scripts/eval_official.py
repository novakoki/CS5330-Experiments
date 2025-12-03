"""Run inference on Custom-Val and produce nuScenes submission + metrics."""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from torch.utils.data import DataLoader

from dataset.nuscenes import NuScenesCarDataset, collate_batch
from model.pointpillars import PointPillars


def load_config(path: str) -> Dict:
    cfg = {}
    exec(open(path, "r").read(), cfg)
    return cfg["config"]


def lidar_box_to_global(nusc: NuScenes, box: Box, lidar_token: str) -> Box:
    sd_rec = nusc.get("sample_data", lidar_token)
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    box = box.copy()
    box.rotate(Quaternion(cs_rec["rotation"]))
    box.translate(np.array(cs_rec["translation"]))
    box.rotate(Quaternion(pose_rec["rotation"]))
    box.translate(np.array(pose_rec["translation"]))
    return box


def build_model(cfg: Dict, grid_size) -> PointPillars:
    model_cfg = cfg["model"]
    model_cfg["max_points_per_voxel"] = cfg["dataset"]["max_points_per_voxel"]
    model_cfg["max_voxels"] = cfg["dataset"]["max_voxels"]["val"]
    model = PointPillars(
        grid_size=grid_size,
        voxel_size=cfg["dataset"]["voxel_size"],
        point_cloud_range=cfg["dataset"]["point_cloud_range"],
        model_cfg=model_cfg,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output", default="submission.json", help="Path to save submission json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score_thr", type=float, default=0.3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    nusc = NuScenes(version="v1.0-trainval", dataroot=dataset_cfg["data_root"], verbose=False)
    val_dataset = NuScenesCarDataset(
        data_root=dataset_cfg["data_root"],
        scene_list_path=dataset_cfg["val_scenes"],
        split="val",
        point_cloud_range=dataset_cfg["point_cloud_range"],
        voxel_size=dataset_cfg["voxel_size"],
        max_points_per_voxel=dataset_cfg["max_points_per_voxel"],
        max_voxels=dataset_cfg["max_voxels"]["val"],
        class_name=dataset_cfg["class_name"],
        augmentations={"rotation": False, "scaling": False, "flip": False, "copy_paste": False},
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_batch)
    device = torch.device(args.device)

    model = build_model(cfg, val_dataset.grid_size)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            dets = model.predict(outputs, score_thresh=args.score_thr, nms_thresh=0.2)[0]
            lidar_token = batch["meta"][0]["lidar_token"]
            for box_np, score in zip(dets["boxes"].cpu().numpy(), dets["scores"].cpu().numpy()):
                box = Box(
                    center=box_np[:3],
                    size=box_np[3:6],
                    orientation=Quaternion(axis=[0, 0, 1], angle=float(box_np[6])),
                    name="vehicle.car",
                    token=lidar_token,
                )
                box = lidar_box_to_global(nusc, box, lidar_token)
                quat = box.orientation.q
                results.append(
                    {
                        "sample_token": batch["meta"][0]["sample_token"],
                        "translation": box.center.tolist(),
                        "size": [float(box.wlh[0]), float(box.wlh[1]), float(box.wlh[2])],
                        "rotation": [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])],
                        "velocity": [0.0, 0.0],
                        "detection_name": "car",
                        "detection_score": float(score),
                        "attribute_name": "vehicle.parked",
                    }
                )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Aggregate per sample
    submission = {"meta": {"use_camera": False, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False}, "results": {}}
    for res in results:
        submission["results"].setdefault(res["sample_token"], []).append(res)
    with open(args.output, "w") as f:
        json.dump(submission, f)
    print(f"Saved submission to {args.output}")

    eval_cfg = config_factory("detection_cvpr_2019")
    eval_out_dir = str(Path(args.output).parent / "eval")
    evaluator = NuScenesEval(
        nusc,
        config=eval_cfg,
        result_path=args.output,
        eval_set="val",
        output_dir=eval_out_dir,
        verbose=True,
    )
    evaluator.main(render_curves=False)


if __name__ == "__main__":
    main()
