from mmengine.config import read_base

with read_base():
    from .exp3_scaling import *  # noqa: F401,F403,F405

work_dir = "../outputs/mmdet/exp4_transfer"
# Point to a KITTI-pretrained PointPillars checkpoint converted for MMDetection3D.
load_from = "../weights/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.pth"
