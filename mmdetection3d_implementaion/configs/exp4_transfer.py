from datetime import datetime
import os

from mmengine.config import read_base

with read_base():
    from .exp3_scaling import *  # noqa: F401,F403,F405

run_suffix = os.environ.get("RUN_SUFFIX", datetime.now().strftime("%Y%m%d-%H%M%S"))
work_dir = f"../outputs/mmdet/exp4_transfer/{run_suffix}"
# Point to a KITTI-pretrained PointPillars checkpoint converted for MMDetection3D.
load_from = "mmdetection3d_implementaion/weights/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.pth"

del os
del datetime
