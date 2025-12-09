#!/usr/bin/env python3
"""Quick visualization helper for nuScenes detections.

Generates:
- BEV plot with lidar points and predicted boxes.
- Camera overlay with projected boxes (if camera data is available).

Example:
    python scripts/visualize_predictions.py \\
        --config mmdetection3d_implementaion/configs/pointpillars_car_base.py \\
        --checkpoint outputs/mmdet/exp1_baseline/epoch_1.pth \\
        --info-pkl data/nuscenes/custom_infos_val.pkl \\
        --index 0 \\
        --out-dir outputs/vis \\
        --camera CAM_FRONT
"""

import argparse
import os
import os.path as osp
from typing import Tuple

import matplotlib.pyplot as plt
import mmcv
import mmengine
import numpy as np
import torch
from mmdet3d.apis import inference_detector, init_model
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def build_lidar2img(
        nusc: NuScenes, sample_token: str,
        camera: str) -> Tuple[np.ndarray, str]:
    """Compute lidar->image projection for a given sample/camera."""
    sample = nusc.get('sample', sample_token)
    if camera not in sample['data']:
        raise KeyError(
            f'Camera {camera} not found in sample; '
            f'available keys: {list(sample["data"].keys())}')

    cam_sd = nusc.get('sample_data', sample['data'][camera])
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    cam_cs = nusc.get('calibrated_sensor',
                      cam_sd['calibrated_sensor_token'])
    lidar_cs = nusc.get('calibrated_sensor',
                        lidar_sd['calibrated_sensor_token'])
    cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
    lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    lidar2ego = transform_matrix(lidar_cs['translation'],
                                 Quaternion(lidar_cs['rotation']),
                                 inverse=False)
    ego2global = transform_matrix(lidar_pose['translation'],
                                  Quaternion(lidar_pose['rotation']),
                                  inverse=False)
    global2ego_cam = transform_matrix(cam_pose['translation'],
                                      Quaternion(cam_pose['rotation']),
                                      inverse=True)
    ego2cam = transform_matrix(cam_cs['translation'],
                               Quaternion(cam_cs['rotation']),
                               inverse=True)

    lidar2cam = ego2cam @ global2ego_cam @ ego2global @ lidar2ego
    cam_intrinsic = np.array(cam_cs['camera_intrinsic'])
    lidar2img = cam_intrinsic @ lidar2cam[:3, :]
    img_path = osp.join(nusc.dataroot, cam_sd['filename'])
    return lidar2img, img_path


def _bev_polygon(xy, w, l, yaw):
    """Return 4 corner xy for a box in the BEV plane."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    offsets = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2],
                        [-l / 2, w / 2]])
    return offsets @ rot.T + xy


def save_bev(points: np.ndarray, bboxes, scores: torch.Tensor, score_thr: float,
             out_path: str) -> None:
    """Save a BEV scatter with predicted boxes."""
    plt.figure(figsize=(8, 8))
    xy = points[:, :2]
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        s=0.3,
        c=np.clip(points[:, 2], -3, 2),
        cmap='gray',
        alpha=0.3,
        linewidths=0)

    boxes_np = bboxes.tensor.cpu().numpy()
    scores_np = scores.cpu().numpy()
    for box, score in zip(boxes_np, scores_np):
        if score < score_thr:
            continue
        x, y, _, w, l, _, yaw = box[:7]
        poly = _bev_polygon(np.array([x, y]), w, l, yaw)
        plt.plot(
            np.append(poly[:, 0], poly[0, 0]),
            np.append(poly[:, 1], poly[0, 1]),
            color='tab:red',
            linewidth=1.0)

    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('BEV with predictions')
    plt.grid(True, linestyle='--', linewidth=0.3)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_camera_overlay(img_path: str,
                        bboxes,
                        scores: torch.Tensor,
                        lidar2img: np.ndarray,
                        score_thr: float,
                        out_path: str) -> None:
    """Project boxes to camera image and save."""
    img = mmcv.imread(img_path)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(mmcv.bgr2rgb(img))

    corners = bboxes.corners.cpu().numpy()  # (N, 8, 3)
    scores_np = scores.cpu().numpy()
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    for corner_set, score in zip(corners, scores_np):
        if score < score_thr:
            continue
        homo_corners = np.concatenate(
            [corner_set, np.ones((corner_set.shape[0], 1))], axis=1)
        proj = homo_corners @ lidar2img.T
        if np.any(proj[:, 2] <= 1e-2):
            continue
        pts_2d = proj[:, :2] / proj[:, 2:3]

        for s, e in edges:
            ax.plot(
                [pts_2d[s, 0], pts_2d[e, 0]],
                [pts_2d[s, 1], pts_2d[e, 1]],
                color='tab:red',
                linewidth=1.0)

    ax.axis('off')
    ax.set_title('Projected predictions')
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Model config file')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--info-pkl',
                        default='data/nuscenes/custom_infos_val.pkl',
                        help='Info pkl with sample tokens and lidar paths')
    parser.add_argument('--index',
                        type=int,
                        default=0,
                        help='Sample index in the info pkl to visualize')
    parser.add_argument('--out-dir',
                        default='outputs/vis',
                        help='Directory to save visualizations')
    parser.add_argument('--camera',
                        default='CAM_FRONT',
                        help='Camera name to project onto; set to "" to skip')
    parser.add_argument('--score-thr',
                        type=float,
                        default=0.25,
                        help='Score threshold for drawing boxes')
    parser.add_argument('--device',
                        default=None,
                        help='torch device, defaults to auto CUDA/CPU')
    parser.add_argument('--version',
                        default='v1.0-trainval',
                        help='nuScenes version string')
    parser.add_argument('--dataroot',
                        default=None,
                        help='nuScenes dataroot; inferred from lidar path when unset')
    args = parser.parse_args()

    infos = mmengine.load(args.info_pkl)['infos']
    if args.index >= len(infos):
        raise IndexError(
            f'Index {args.index} out of range for {len(infos)} samples.')
    info = infos[args.index]

    device = args.device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = init_model(args.config, args.checkpoint, device=device)
    result, _ = inference_detector(model, info['lidar_path'])
    preds = result.pred_instances_3d

    points = np.fromfile(
        info['lidar_path'], dtype=np.float32).reshape(-1,
                                                      info['num_features'])

    bev_path = osp.join(args.out_dir,
                        f'sample_{args.index:04d}_bev.png')
    save_bev(points, preds.bboxes_3d, preds.scores_3d, args.score_thr,
             bev_path)

    if args.camera:
        dataroot = args.dataroot
        if dataroot is None:
            # Infer dataroot by walking up from the lidar file until we hit "samples"
            lid_path = osp.abspath(info['lidar_path'])
            parent = osp.dirname(lid_path)
            while parent and osp.basename(parent) != 'samples':
                next_parent = osp.dirname(parent)
                if next_parent == parent:
                    break
                parent = next_parent
            if osp.basename(parent) == 'samples':
                dataroot = osp.dirname(parent)
            else:
                raise RuntimeError(
                    f'Could not infer nuScenes dataroot from {info["lidar_path"]}; '
                    'please pass --dataroot explicitly.')

        nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=False)
        sample_data_keys = list(nusc.get('sample', info['token'])['data'].keys())
        if args.camera not in sample_data_keys:
            raise KeyError(
                f'Camera {args.camera} not in sample; available: {sample_data_keys}')

        lidar2img, img_path = build_lidar2img(nusc, info['token'],
                                              args.camera)
        cam_path = osp.join(args.out_dir,
                            f'sample_{args.index:04d}_{args.camera}.png')
        save_camera_overlay(img_path, preds.bboxes_3d, preds.scores_3d,
                            lidar2img, args.score_thr, cam_path)

    print(f'Saved visualizations to {args.out_dir}')


if __name__ == '__main__':
    main()
