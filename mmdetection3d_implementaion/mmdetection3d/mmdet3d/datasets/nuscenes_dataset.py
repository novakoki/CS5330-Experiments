# Copyright (c) OpenMMLab. All rights reserved.
import os
from os import path as osp
from typing import Callable, List, Union

import numpy as np
from mmengine.fileio import load
from mmengine.logging import print_log

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.convert_utils import NuScenesNameMapping
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class NuScenesDataset(Det3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        with_velocity (bool): Whether to include velocity prediction
            into the experiments. Defaults to True.
        use_valid_flag (bool): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
        'version':
        'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 modality: dict = dict(
                     use_camera=False,
                     use_lidar=True,
                 ),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 with_velocity: bool = True,
                 use_valid_flag: bool = False,
                 **kwargs) -> None:
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity

        # TODO: Redesign multi-view data process in the future
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type

        assert box_type_3d.lower() in ('lidar', 'camera')
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            modality=modality,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def _filter_with_mask(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos.

        Returns:
            dict: Annotations after filtering.
        """
        filtered_annotations = {}
        if self.use_valid_flag:
            filter_mask = ann_info['bbox_3d_isvalid']
        else:
            filter_mask = ann_info['num_lidar_pts'] > 0
        for key in ann_info.keys():
            if key != 'instances':
                filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                filtered_annotations[key] = ann_info[key]
        return filtered_annotations

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is not None:

            ann_info = self._filter_with_mask(ann_info)

            if self.with_velocity:
                gt_bboxes_3d = ann_info['gt_bboxes_3d']
                gt_velocities = ann_info['velocities']
                nan_mask = np.isnan(gt_velocities[:, 0])
                gt_velocities[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocities],
                                              axis=-1)
                ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        else:
            # empty instance
            ann_info = dict()
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['attr_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO: Unify the coordinates
        if self.load_type in ['fov_image_based', 'mv_image_based']:
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5))
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info

    def _strip_prefix(self, path: str, preferred_prefix: str) -> str:
        """Return a path relative to the preferred prefix if possible."""
        abs_root = osp.abspath(self.data_root) if self.data_root else ""
        abs_preferred = osp.abspath(osp.join(abs_root, preferred_prefix)) if preferred_prefix else abs_root
        abs_path = path
        if not osp.isabs(abs_path):
            abs_path = osp.abspath(osp.join(abs_root, path))
        if abs_preferred and abs_path.startswith(abs_preferred):
            return osp.relpath(abs_path, abs_preferred)
        if abs_root and abs_path.startswith(abs_root):
            return osp.relpath(abs_path, abs_root)
        return path

    def _convert_legacy_info(self, legacy_info: dict) -> dict:
        """Adapt legacy ``infos`` format (lidar_path/gt_boxes) to new data_list."""
        info = dict()
        pts_prefix = self.data_prefix.get('pts', '')
        sweeps_prefix = self.data_prefix.get('sweeps', '')

        lidar_path = legacy_info['lidar_path']
        info['lidar_points'] = dict(
            lidar_path=self._strip_prefix(lidar_path, pts_prefix),
            num_pts_feats=legacy_info.get('num_features', 5),
        )

        sweeps = []
        for sweep in legacy_info.get('sweeps', []):
            sweep_path = sweep['data_path']
            sweeps.append(
                dict(
                    lidar_points=dict(lidar_path=self._strip_prefix(sweep_path, sweeps_prefix)),
                    timestamp=sweep.get('timestamp', 0),
                )
            )
        if sweeps:
            info['lidar_sweeps'] = sweeps

        info['timestamp'] = legacy_info.get('timestamp')
        info['token'] = legacy_info.get('token')
        info['ego2global'] = dict(
            rotation=legacy_info.get('ego2global_rotation'),
            translation=legacy_info.get('ego2global_translation'),
        )
        info['lidar2ego'] = dict(
            rotation=legacy_info.get('lidar2ego_rotation'),
            translation=legacy_info.get('lidar2ego_translation'),
        )

        gt_boxes = legacy_info.get('gt_boxes', [])
        gt_names = legacy_info.get('gt_names', [])
        gt_velocity = legacy_info.get('gt_velocity', [])
        num_lidar_pts = legacy_info.get('num_lidar_pts', [])
        num_radar_pts = legacy_info.get('num_radar_pts', [])
        valid_flag = legacy_info.get('valid_flag', [])

        instances = []
        for idx, name in enumerate(gt_names):
            bbox = gt_boxes[idx]
            # Map nuScenes taxonomy to detection classes (e.g., vehicle.car -> car).
            mapped_name = NuScenesNameMapping.get(name, name)
            label = self.metainfo['classes'].index(mapped_name) if mapped_name in self.metainfo['classes'] else -1
            instance = dict(
                bbox_3d=bbox,
                bbox_label_3d=label,
                velocity=gt_velocity[idx] if len(gt_velocity) > idx else np.array([0.0, 0.0]),
                num_lidar_pts=num_lidar_pts[idx] if len(num_lidar_pts) > idx else 0,
                num_radar_pts=num_radar_pts[idx] if len(num_radar_pts) > idx else 0,
                bbox_3d_isvalid=bool(valid_flag[idx]) if len(valid_flag) > idx else True,
            )
            instances.append(instance)
        info['instances'] = instances

        return info

    def load_data_list(self) -> List[dict]:
        """Support both new ``data_list`` format and legacy ``infos`` PKL."""
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file should be a dict, '
                            f'but got {type(annotations)}!')

        if 'data_list' in annotations and 'metainfo' in annotations:
            metainfo = annotations['metainfo']
            raw_data_list = annotations['data_list']
            for k, v in metainfo.items():
                self._metainfo.setdefault(k, v)
            return [self.parse_data_info(info) for info in raw_data_list]

        if 'infos' in annotations:
            metainfo = annotations.get('metadata', {})
            for k, v in metainfo.items():
                self._metainfo.setdefault(k, v)
            print_log('Loading legacy nuScenes infos format.', logger='current')
            return [self.parse_data_info(self._convert_legacy_info(info)) for info in annotations['infos']]

        raise ValueError('Annotation must contain either data_list/metainfo or infos/metadata keys')

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.load_type == 'mv_image_based':
            data_list = []
            if self.modality['use_lidar']:
                info['lidar_points']['lidar_path'] = \
                    osp.join(
                        self.data_prefix.get('pts', ''),
                        info['lidar_points']['lidar_path'])

            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        if cam_id in self.data_prefix:
                            cam_prefix = self.data_prefix[cam_id]
                        else:
                            cam_prefix = self.data_prefix.get('img', '')
                        img_info['img_path'] = osp.join(
                            cam_prefix, img_info['img_path'])

            for idx, (cam_id, img_info) in enumerate(info['images'].items()):
                camera_info = dict()
                camera_info['images'] = dict()
                camera_info['images'][cam_id] = img_info
                if 'cam_instances' in info and cam_id in info['cam_instances']:
                    camera_info['instances'] = info['cam_instances'][cam_id]
                else:
                    camera_info['instances'] = []
                # TODO: check whether to change sample_idx for 6 cameras
                #  in one frame
                camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
                camera_info['token'] = info['token']
                camera_info['ego2global'] = info['ego2global']

                if not self.test_mode:
                    # used in traing
                    camera_info['ann_info'] = self.parse_ann_info(camera_info)
                if self.test_mode and self.load_eval_anns:
                    camera_info['eval_ann_info'] = \
                        self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            data_info = super().parse_data_info(info)
            return data_info
