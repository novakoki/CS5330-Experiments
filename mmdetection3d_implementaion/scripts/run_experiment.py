"""Launch an MMDetection3D training run for one of the custom PointPillars experiments."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4], required=True, help="Experiment ID to run")
    parser.add_argument("--mmdet_root", type=Path, default=Path("mmdetection3d"), help="Path to the MMDetection3D repo")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Additional args passed to tools/train.py")
    args = parser.parse_args()

    cfg_map = {
        1: "exp1_baseline.py",
        2: "exp2_augmentation.py",
        3: "exp3_scaling.py",
        4: "exp4_transfer.py",
    }
    repo_root = Path(__file__).resolve().parents[1]
    mmdet_root = args.mmdet_root.resolve()
    cfg_path = (repo_root / "configs" / cfg_map[args.exp]).resolve()
    train_script = (mmdet_root / "tools" / "train.py").resolve()
    cmd = [sys.executable, str(train_script), str(cfg_path)]
    if args.extra_args:
        cmd.extend(args.extra_args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=mmdet_root)


if __name__ == "__main__":
    main()
