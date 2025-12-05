"""Main training script for custom PointPillars car detector."""
import argparse
import os
import runpy
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.nuscenes import NuScenesCarDataset, collate_batch
from model.pointpillars import PointPillars
from utils.evaluation import evaluate_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config python file.")
    parser.add_argument("--work_dir", type=str, default=None, help="Override work_dir from config.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def load_config(path: str) -> Dict:
    cfg = runpy.run_path(path)
    if "config" not in cfg:
        raise ValueError(f"No `config` object found in {path}")
    return cfg["config"]


def build_dataloaders(cfg: Dict):
    dataset_cfg = cfg["dataset"]
    num_workers = cfg.get("num_workers", 0)
    pin_memory = cfg.get("pin_memory", True)
    prefetch_factor = cfg.get("prefetch_factor", 2)
    print(f"[build_dataloaders] loading splits train={dataset_cfg['train_scenes']} val={dataset_cfg['val_scenes']}")
    train_dataset = NuScenesCarDataset(
        data_root=dataset_cfg["data_root"],
        scene_list_path=dataset_cfg["train_scenes"],
        split="train",
        point_cloud_range=dataset_cfg["point_cloud_range"],
        voxel_size=dataset_cfg["voxel_size"],
        max_points_per_voxel=dataset_cfg["max_points_per_voxel"],
        max_voxels=dataset_cfg["max_voxels"]["train"],
        class_name=dataset_cfg["class_name"],
        augmentations=dataset_cfg.get("augmentations", {}),
    )
    val_dataset = None
    if dataset_cfg.get("load_val", True):
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
        print(f"[build_dataloaders] dataset sizes: train={len(train_dataset)} val={len(val_dataset) if val_dataset else 0}")
    print(
        f"[build_dataloaders] grid_size={train_dataset.grid_size} "
        f"num_workers={num_workers} pin_memory={pin_memory} prefetch_factor={prefetch_factor if num_workers > 0 else 'off'}"
    )
    loader_kwargs = dict(
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_batch,
        drop_last=True,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        **loader_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader_kwargs = loader_kwargs.copy()
        val_loader_kwargs.update({"shuffle": False, "drop_last": False})
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["train"]["batch_size"],
            **val_loader_kwargs,
        )
    return train_loader, val_loader, train_dataset.grid_size


def run_validation(model: PointPillars, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds: List[Dict] = []
    gts: List[Dict] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v for k, v in batch.items()}
            outputs = model(batch)
            dets = model.predict(outputs, score_thresh=0.3, nms_thresh=0.2)
            for i, det in enumerate(dets):
                preds.append({"boxes": det["boxes"].detach(), "scores": det["scores"].detach()})
                gts.append({"boxes": batch["gt_boxes"][i].to(device)})
    if len(preds) == 0:
        return 0.0
    return evaluate_map(preds, gts, iou_thr=0.5)


def save_checkpoint(state: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    work_dir = Path(cfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(work_dir / "tb"))
    set_seed(cfg.get("seed", 42), deterministic=cfg.get("deterministic", False))
    device = torch.device(args.device)

    def log_mem(tag: str):
        try:
            import psutil  # type: ignore

            process = psutil.Process()
            rss_gb = process.memory_info().rss / (1024 ** 3)
            print(f"[mem] {tag}: RSS={rss_gb:.2f} GB")
        except Exception:
            pass

    log_mem("start")
    train_loader, val_loader, grid_size = build_dataloaders(cfg)
    log_mem("after_dataloaders")
    model_cfg = cfg["model"]
    model_cfg["max_points_per_voxel"] = cfg["dataset"]["max_points_per_voxel"]
    model_cfg["max_voxels"] = cfg["dataset"]["max_voxels"]["train"]
    model = PointPillars(
        grid_size=grid_size,
        voxel_size=cfg["dataset"]["voxel_size"],
        point_cloud_range=cfg["dataset"]["point_cloud_range"],
        model_cfg=model_cfg,
    ).to(device)
    if cfg.get("pretrained"):
        ckpt = torch.load(cfg["pretrained"], map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["base_lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["train"]["onecycle_max_lr"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["train"]["epochs"],
        pct_start=cfg["train"]["pct_start"],
        div_factor=cfg["train"]["div_factor"],
        final_div_factor=cfg["train"]["final_div_factor"],
    )
    scaler = GradScaler(enabled=cfg.get("amp", True) and device.type == "cuda")

    best_map = 0.0
    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        data_start = time.perf_counter()
        for i, batch in enumerate(train_loader):
            data_time = time.perf_counter() - data_start
            iter_start = time.perf_counter()
            optimizer.zero_grad()
            with autocast(enabled=cfg.get("amp", True) and device.type == "cuda"):
                preds = model(batch)
                loss_dict = model.loss(preds, batch, cfg["model"]["loss"])
                loss = loss_dict["total_loss"]
            scaler.scale(loss).backward()
            if cfg.get("grad_clip", None):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            iter_time = time.perf_counter() - iter_start
            epoch_loss += loss.item()
            global_step += 1
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            writer.add_scalar("train/loss_cls", loss_dict["cls_loss"].item(), global_step)
            writer.add_scalar("train/loss_box", loss_dict["box_loss"].item(), global_step)
            writer.add_scalar("train/loss_dir", loss_dict["dir_loss"].item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("time/data_ms", data_time * 1000, global_step)
            writer.add_scalar("time/iter_ms", iter_time * 1000, global_step)
            if global_step % cfg.get("log_interval", 10) == 0:
                gpu_mem = 0.0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(
                    f"Epoch [{epoch+1}/{cfg['train']['epochs']}], Step {i}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, cls: {loss_dict['cls_loss']:.4f}, "
                    f"box: {loss_dict['box_loss']:.4f}, dir: {loss_dict['dir_loss']:.4f}, "
                    f"data: {data_time*1000:.1f}ms, iter: {iter_time*1000:.1f}ms, "
                    f"lr: {scheduler.get_last_lr()[0]:.6f}, gpu_mem: {gpu_mem:.2f} GB"
                )
            data_start = time.perf_counter()
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)
        print(f"Epoch {epoch+1} done in {time.time()-start:.1f}s, avg loss {avg_epoch_loss:.4f}")

        if val_loader is not None and (epoch + 1) % cfg.get("val_interval", 1) == 0:
            val_map = run_validation(model, val_loader, device)
            print(f"Validation mAP@0.5: {val_map:.4f}")
            writer.add_scalar("val/mAP@0.5", val_map, global_step)
            if val_map > best_map:
                best_map = val_map
                save_checkpoint(model.state_dict(), str(work_dir / "best.pth"))
        if (epoch + 1) % cfg.get("save_every", 1) == 0:
            save_checkpoint(model.state_dict(), str(work_dir / f"epoch_{epoch+1}.pth"))

    save_checkpoint(model.state_dict(), str(work_dir / "last.pth"))
    print(f"Training complete. Best val mAP: {best_map:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
