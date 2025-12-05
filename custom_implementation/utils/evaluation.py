"""Lightweight BEV metrics and NMS utilities."""
from typing import Dict, List, Tuple

import torch

from utils.anchors import bev_iou


def nms_bev(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5, max_boxes: int = 100) -> List[int]:
    """Axis-aligned BEV NMS."""
    if boxes.numel() == 0:
        return []
    keep: List[int] = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0 and len(keep) < max_boxes:
        current = idxs[0]
        keep.append(int(current))
        if idxs.numel() == 1:
            break
        cur_box = boxes[current].unsqueeze(0)
        rest = boxes[idxs[1:]]
        ious = bev_iou(cur_box, rest).squeeze(0)
        idxs = idxs[1:][ious <= iou_threshold]
    return keep


def average_precision(recalls: List[float], precisions: List[float]) -> float:
    """11-point interpolated AP."""
    ap = 0.0
    for thresh in [t / 10.0 for t in range(0, 11)]:
        prec = max([p for r, p in zip(recalls, precisions) if r >= thresh] or [0])
        ap += prec
    return ap / 11.0


def evaluate_sample(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thr: float = 0.5,
) -> float:
    """Compute AP for a single frame."""
    if gt_boxes.numel() == 0:
        return 1.0 if pred_boxes.numel() == 0 else 0.0
    if pred_boxes.numel() == 0:
        return 0.0
    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    matched = torch.zeros((gt_boxes.shape[0],), device=gt_boxes.device, dtype=torch.bool)
    tp = []
    fp = []
    for i, box in enumerate(pred_boxes):
        ious = bev_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        max_iou, gt_idx = ious.max(0)
        if max_iou >= iou_thr and not matched[gt_idx]:
            tp.append(1)
            fp.append(0)
            matched[gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    tp_cum = torch.cumsum(torch.tensor(tp, dtype=torch.float32), dim=0)
    fp_cum = torch.cumsum(torch.tensor(fp, dtype=torch.float32), dim=0)
    recalls = (tp_cum / max(len(gt_boxes), 1)).tolist()
    precisions = (tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)).tolist()
    return average_precision(recalls, precisions)


def evaluate_map(
    preds: List[Dict[str, torch.Tensor]],
    gts: List[Dict[str, torch.Tensor]],
    iou_thr: float = 0.5,
) -> float:
    """Compute mean AP across samples."""
    scores = []
    for pred, gt in zip(preds, gts):
        ap = evaluate_sample(pred["boxes"], pred["scores"], gt["boxes"], iou_thr=iou_thr)
        scores.append(ap)
    return float(torch.tensor(scores).mean())

