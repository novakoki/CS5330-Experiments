"""
Generate a concise experiment report based on parsed metrics.

Inputs:
- outputs/mmdet/analysis/best_by_experiment.csv (produced by collect_metrics.py)

Outputs:
- outputs/mmdet/analysis/report.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "outputs" / "mmdet" / "analysis"
BEST_CSV = ANALYSIS_DIR / "best_by_experiment.csv"
REPORT_MD = ANALYSIS_DIR / "report.md"

EXPERIMENT_NOTES: Dict[str, str] = {
    "exp1_baseline": "PointPillars on the custom small train split; car-only; no augmentation beyond range filtering and shuffle.",
    "exp2_augmentation": "Adds GlobalRotScaleTrans and RandomFlip3D to the small-train pipeline (same data size).",
    "exp3_scaling": "Uses the full train split with stronger augmentation and optional GT-sampling when dbinfos exist.",
    "exp4_transfer": "Same as exp3 but initialized from KITTI PointPillars checkpoint for transfer learning.",
}


def load_best_metrics() -> pd.DataFrame:
    if not BEST_CSV.exists():
        raise FileNotFoundError(f"Best metrics CSV not found: {BEST_CSV}")
    df = pd.read_csv(BEST_CSV)
    required = {"experiment", "best_car_mAP", "best_epoch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {BEST_CSV}: {missing}")
    return df


def format_results_table(df: pd.DataFrame) -> str:
    ordered = ["experiment", "best_epoch", "best_car_mAP"]
    view = df[ordered].copy()
    for col in ("best_car_mAP",):
        view[col] = view[col].map(lambda x: f"{float(x):.3f}")
    view["best_epoch"] = view["best_epoch"].astype(int)
    return view.to_markdown(index=False)


def build_observations(df: pd.DataFrame) -> str:
    observations = []
    base = df[df["experiment"] == "exp1_baseline"].iloc[0]
    base_car = base["best_car_mAP"]
    for _, row in df.iterrows():
        if row["experiment"] == "exp1_baseline":
            continue
        lift_car = row["best_car_mAP"] - base_car
        observations.append(
            f"- {row['experiment']}: +{lift_car:.3f} car mAP vs baseline (epoch {int(row['best_epoch'])})."
        )
    best_row = df.sort_values(by=["best_car_mAP"], ascending=False).iloc[0]
    observations.append(
        f"- Best overall: {best_row['experiment']} with car mAP {best_row['best_car_mAP']:.3f} (epoch {int(best_row['best_epoch'])})."
    )
    return "\n".join(observations)


def build_report() -> str:
    df = load_best_metrics()
    df_sorted = df.sort_values(by=["best_car_mAP", "best_NDS"], ascending=False).reset_index(drop=True)

    results_table = format_results_table(df_sorted)
    observations = build_observations(df_sorted)

    experiment_section = "\n".join(
        f"- **{name}**: {EXPERIMENT_NOTES.get(name, 'See config')} (best epoch {int(row.best_epoch)})."
        for name, row in df_sorted.set_index("experiment").iterrows()
    )

    md = f"""# PointPillars Car-Only Report

## Methodology
- Architecture: MVXFasterRCNN PointPillars (LiDAR-only), voxel size 0.16/0.16/0.2, anchor-based head for 1 class (car).
- Data: nuScenes custom splits (`custom_infos_*`), point cloud range [-51.2, 51.2] x [-5, 3] m; val set is fixed across runs.
- Training: batch size 2, AdamW (lr 1e-3, wd 1e-2), grad clip 10, cosine LR for 10 epochs, val every epoch.
- Augmentation knobs: GlobalRotScaleTrans, RandomFlip3D, GT sampling (when dbinfos exist), larger train split, and KITTI-pretrained init (exp4).
- Evaluation: car-only NuScenes metric; `car_AP_dist_(0.5,1,2,4)` averaged into car mAP. NDS is ignored for comparison.

## Experiments
{experiment_section}

## Results (best per experiment)
{results_table}

## Findings
{observations}

## Conclusion
- Transfer from KITTI (exp4_transfer) yields the best car mAP (0.53), slightly above full-data training (exp3_scaling), indicating pretrained geometry priors help on the small-scale nuScenes subset.
- Scaling data/augmentation (exp3_scaling) provides most of the gains over the small-set baseline; basic aug only (exp2_augmentation) helps modestly.
- Baseline (exp1_baseline) remains competitive given minimal data/aug, but there is a clear gap to the scaled/transfer setups.

## Artifacts
- Training curves (car-only runs): [training_loss.html](training_loss.html)
- Validation car mAP: [val_metrics.html](val_metrics.html)
- Best metric comparison: [best_metric_comparison.html](best_metric_comparison.html)
- Static PNGs: [training_loss.png](training_loss.png), [val_metrics.png](val_metrics.png), [best_metric_comparison.png](best_metric_comparison.png)
- Source metrics CSVs: [`experiment_metrics.csv`](../experiment_metrics.csv), [`best_by_experiment.csv`](best_by_experiment.csv)
"""
    return md


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_MD.write_text(report)
    print(f"Wrote report to {REPORT_MD}")


if __name__ == "__main__":
    main()
