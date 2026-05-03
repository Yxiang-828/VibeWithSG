#!/usr/bin/env python3
"""Config-driven training entrypoint for CSOT phases P1+."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg.utils import set_rocm_defaults

set_rocm_defaults()

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from seg.constants import CLASS_NAMES, NUM_CLASSES
from seg.data.dataset import convert_raw_mask_to_class_ids
from seg.metrics import iou_from_confusion
from seg.models import DinoDPTSegmentation
from seg.utils import seed_everything, timestamp, write_json


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("inherits"):
        inherited = Path(config["inherits"])
        if inherited.is_absolute():
            parent_path = inherited
        elif inherited.exists():
            parent_path = inherited
        else:
            parent_path = path.parent / inherited
        parent = load_config(parent_path.resolve())
        return deep_merge(parent, config)
    return config


class SegmentationTensorDataset(Dataset):
    def __init__(self, data_dir: str | Path, image_size: tuple[int, int], use_precomputed_masks: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "Color_Images"
        classid_dir = self.data_dir / "Segmentation_classid"
        raw_dir = self.data_dir / "Segmentation"
        self.mask_dir = classid_dir if use_precomputed_masks and classid_dir.exists() else raw_dir
        self.uses_precomputed_masks = self.mask_dir.name == "Segmentation_classid"
        self.ids = sorted(p.name for p in self.image_dir.glob("*.png"))
        self.image_transform = v2.Compose(
            [
                v2.Resize(image_size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.mask_transform = v2.Compose(
            [
                v2.Resize(image_size, interpolation=v2.InterpolationMode.NEAREST_EXACT),
                v2.PILToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.ids[idx]
        with Image.open(self.image_dir / name) as im:
            image = im.convert("RGB")
        with Image.open(self.mask_dir / name) as im:
            mask = im.copy()
        if not self.uses_precomputed_masks:
            mask = convert_raw_mask_to_class_ids(mask)
        image_t = self.image_transform(image)
        mask_t = self.mask_transform(mask).squeeze(0).long()
        return image_t, mask_t


def update_confusion(cm: np.ndarray, pred: torch.Tensor, target: torch.Tensor) -> None:
    pred_np = pred.detach().cpu().numpy().astype(np.int64)
    target_np = target.detach().cpu().numpy().astype(np.int64)
    valid = (target_np >= 0) & (target_np < NUM_CLASSES)
    encoded = NUM_CLASSES * target_np[valid] + pred_np[valid]
    cm += np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = True,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_pixels = 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        if training:
            loss.backward()
            optimizer.step()

        batch_pixels = masks.numel()
        total_loss += float(loss.detach()) * batch_pixels
        total_pixels += batch_pixels
        update_confusion(cm, logits.argmax(dim=1), masks)

    miou, per_class_iou = iou_from_confusion(cm)
    accuracy = float(np.diag(cm).sum() / max(cm.sum(), 1))
    return {
        "loss": total_loss / max(total_pixels, 1),
        "miou": miou,
        "accuracy": accuracy,
        "per_class_iou": per_class_iou.tolist(),
    }


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    if model_cfg["decoder"] != "dpt":
        raise ValueError(f"scripts/train.py currently supports decoder=dpt, got {model_cfg['decoder']}")
    return DinoDPTSegmentation(
        backbone_name=model_cfg["backbone"],
        num_classes=NUM_CLASSES,
        freeze_backbone=bool(config["training"].get("freeze_backbone", True)),
        decoder_dim=int(model_cfg.get("decoder_dim", 256)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Override config epoch count for sanity gates.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a last.pt or best.pt checkpoint.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    seed_everything(int(config.get("seed", 0)))
    phase = config.get("phase", "phase")
    run_name = args.run_name or f"{timestamp()}_{phase}_{config['model']['backbone']}_{config['model']['decoder']}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / args.config.name)
    write_json(run_dir / "resolved_config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    image_size = tuple(config["model"]["image_size"])
    train_ds = SegmentationTensorDataset(
        config["data"]["train_dir"],
        image_size=image_size,
        use_precomputed_masks=bool(config["data"].get("use_precomputed_masks", True)),
    )
    val_ds = SegmentationTensorDataset(
        config["data"]["val_dir"],
        image_size=image_size,
        use_precomputed_masks=bool(config["data"].get("use_precomputed_masks", True)),
    )
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(config).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = str(config["training"].get("optimizer", "adamw")).lower()
    lr = float(config["training"]["lr"])
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    class_weights_mode = config["training"].get("class_weights")
    if class_weights_mode == "median_frequency":
        # Median-frequency balancing using per-class pixel shares from
        # docs/DATASET_AUDIT.md (median = Trees at 3.5%); capped at 50x per
        # configs/03_lovasz_copypaste.yaml policy.
        weights_tensor = torch.tensor(
            [28.69, 1.00, 0.59, 0.185, 3.18, 0.80, 1.25, 44.87, 2.92, 0.143, 0.093],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print(f"loss=ce_class_weighted mode={class_weights_mode} weights={weights_tensor.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("loss=ce_uniform")
    use_amp = str(config["training"].get("amp", "")).lower() in {"bf16", "true", "1"}

    start_epoch = 1
    best_miou = -1.0
    best_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_miou = float(checkpoint.get("val_miou", -1.0))
        best_epoch = int(checkpoint.get("epoch", 0))
        print(f"resumed={args.resume} start_epoch={start_epoch} best={best_miou:.4f}@{best_epoch}")

    csv_path = run_dir / "metrics.csv"
    history = []
    start = time.time()
    append_csv = args.resume is not None and csv_path.exists()
    with csv_path.open("a" if append_csv else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_miou", "val_loss", "val_miou", "val_accuracy"])
        if not append_csv:
            writer.writeheader()
        for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
            epoch_start = time.time()
            train_stats = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, use_amp=use_amp)
            with torch.no_grad():
                val_stats = run_epoch(model, val_loader, criterion, device, optimizer=None, use_amp=use_amp)
            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_miou": train_stats["miou"],
                "val_loss": val_stats["loss"],
                "val_miou": val_stats["miou"],
                "val_accuracy": val_stats["accuracy"],
            }
            writer.writerow(row)
            f.flush()
            history.append(
                {
                    **row,
                    "seconds": time.time() - epoch_start,
                    "val_per_class_iou": {
                        name: val_stats["per_class_iou"][idx]
                        for idx, name in enumerate(CLASS_NAMES)
                    },
                }
            )
            if val_stats["miou"] > best_miou:
                best_miou = val_stats["miou"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": config,
                        "val_miou": best_miou,
                        "val_per_class_iou": val_stats["per_class_iou"],
                    },
                    run_dir / "best.pt",
                )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                    "val_miou": val_stats["miou"],
                    "val_per_class_iou": val_stats["per_class_iou"],
                },
                run_dir / "last.pt",
            )
            print(
                f"epoch {epoch:03d} "
                f"train_loss={train_stats['loss']:.4f} train_miou={train_stats['miou']:.4f} "
                f"val_loss={val_stats['loss']:.4f} val_miou={val_stats['miou']:.4f} "
                f"best={best_miou:.4f}@{best_epoch} time={time.time() - epoch_start:.1f}s",
                flush=True,
            )

    summary = {
        "run_dir": str(run_dir),
        "config": str(args.config),
        "device": str(device),
        "epochs": int(config["training"]["epochs"]),
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "wall_clock_seconds": time.time() - start,
        "history": history,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"best_val_miou={best_miou:.4f} best_epoch={best_epoch} run_dir={run_dir}")


if __name__ == "__main__":
    main()
