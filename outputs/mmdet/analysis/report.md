# PointPillars Car-Only Report

## Methodology
- Architecture: MVXFasterRCNN PointPillars (LiDAR-only), voxel size 0.16/0.16/0.2, anchor-based head for 1 class (car).
- Data: nuScenes custom splits (`custom_infos_*`), point cloud range [-51.2, 51.2] x [-5, 3] m; val set is fixed across runs.
- Training: batch size 2, AdamW (lr 1e-3, wd 1e-2), grad clip 10, cosine LR for 10 epochs, val every epoch.
- Augmentation knobs: GlobalRotScaleTrans, RandomFlip3D, GT sampling (when dbinfos exist), larger train split, and KITTI-pretrained init (exp4).
- Evaluation: car-only NuScenes metric; `car_AP_dist_(0.5,1,2,4)` averaged into car mAP. NDS is ignored for comparison.

## Experiments
- **exp4_transfer**: Same as exp3 but initialized from KITTI PointPillars checkpoint for transfer learning. (best epoch 10).
- **exp3_scaling**: Uses the full train split with stronger augmentation and optional GT-sampling when dbinfos exist. (best epoch 10).
- **exp2_augmentation**: Adds GlobalRotScaleTrans and RandomFlip3D to the small-train pipeline (same data size). (best epoch 9).
- **exp1_baseline**: PointPillars on the custom small train split; car-only; no augmentation beyond range filtering and shuffle. (best epoch 10).

## Results (best per experiment)
| experiment        |   best_epoch |   best_car_mAP |
|:------------------|-------------:|---------------:|
| exp4_transfer     |           10 |          0.53  |
| exp3_scaling      |           10 |          0.526 |
| exp2_augmentation |            9 |          0.402 |
| exp1_baseline     |           10 |          0.375 |

## Findings
- exp4_transfer: +0.156 car mAP vs baseline (epoch 10).
- exp3_scaling: +0.151 car mAP vs baseline (epoch 10).
- exp2_augmentation: +0.028 car mAP vs baseline (epoch 9).
- Best overall: exp4_transfer with car mAP 0.530 (epoch 10).

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
