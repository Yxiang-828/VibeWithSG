"""Run segmentation inference on a directory of images.

Supports:
- Single checkpoint or ensemble (averaged softmax across multiple checkpoints).
- Horizontal-flip test-time augmentation (HFlip TTA).
- Three output PNG formats: `class_id` (0..10 grayscale), `raw_id`
  (0, 100, 200, ... uint16) which matches the dataset's ground-truth format,
  or `color` (RGB visualisation using `seg.constants.COLOR_PALETTE`).

Designed for the `Offroad_Segmentation_testImages/` set but works on any
folder of `.png` / `.jpg` images.

Usage:
    python scripts/predict.py \\
        --checkpoints runs/P2_cloud/best.pt runs/P1_cloud/best.pt \\
        --image-dir Offroad_Segmentation_testImages/Color_Images \\
        --output-dir final_package/masks \\
        --image-size 532 952 \\
        --hflip-tta \\
        --save-format raw_id
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from seg.constants import (  # noqa: E402
    CLASS_TO_RAW_LUT,
    COLOR_PALETTE,
    NUM_CLASSES,
)
from seg.models.dpt import DinoDPTSegmentation  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="One or more checkpoint .pt files. Multiple checkpoints are ensembled (softmax averaged).",
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[532, 952],
        metavar=("H", "W"),
        help="Resize input to this (H, W) before forward. Default 532x952 (full res).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitl14_reg",
        help="Backbone name; must match the checkpoints' training config.",
    )
    parser.add_argument(
        "--hflip-tta",
        action="store_true",
        help="Average predictions of original and horizontally-flipped input.",
    )
    parser.add_argument(
        "--multiscale",
        type=int,
        nargs="+",
        default=None,
        metavar="H",
        help="Multi-scale TTA: list of base heights (widths scale proportionally, "
             "rounded to multiples of 14). Example: --multiscale 420 532 644. "
             "If omitted, only --image-size is used.",
    )
    parser.add_argument(
        "--save-format",
        choices=["class_id", "raw_id", "color"],
        default="raw_id",
        help="Output PNG format. Default raw_id matches the dataset's ground-truth encoding.",
    )
    parser.add_argument(
        "--save-color",
        action="store_true",
        help="Additionally write an RGB color visualisation alongside the primary output.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of output classes; must match training.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, backbone: str, num_classes: int, device: torch.device) -> DinoDPTSegmentation:
    model = DinoDPTSegmentation(backbone_name=backbone, num_classes=num_classes, freeze_backbone=True)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  warn: missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  warn: unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.to(device).eval()
    return model


def build_transform(image_size: tuple[int, int]) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(image_size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _round14(x: int) -> int:
    return max(14, (int(x) // 14) * 14)


@torch.no_grad()
def predict_one(
    pil_image: "Image.Image",
    models: list[DinoDPTSegmentation],
    hflip_tta: bool,
    base_size: tuple[int, int],
    scales: list[tuple[int, int]] | None,
    device: torch.device,
) -> torch.Tensor:
    """Returns averaged softmax probabilities at `base_size` (C, H, W).

    `scales` is a list of (H, W) to run in addition to (or instead of) base_size.
    Each scale's softmax is bilinearly resampled to `base_size` before averaging.
    """
    sizes = scales if scales else [base_size]
    accum = None
    n = 0
    for (h, w) in sizes:
        tf = v2.Compose([
            v2.Resize((h, w), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        batch = tf(pil_image).unsqueeze(0).to(device)
        for model in models:
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            if (h, w) != base_size:
                probs = F.interpolate(probs, size=base_size, mode="bilinear", align_corners=False)
            accum = probs if accum is None else accum + probs
            n += 1
            if hflip_tta:
                logits_f = model(torch.flip(batch, dims=[3]))
                probs_f = torch.flip(F.softmax(logits_f, dim=1), dims=[3])
                if (h, w) != base_size:
                    probs_f = F.interpolate(probs_f, size=base_size, mode="bilinear", align_corners=False)
                accum = accum + probs_f
                n += 1
    assert accum is not None
    return (accum / n).squeeze(0)  # (C, H, W)


def encode_mask(class_mask: np.ndarray, save_format: str) -> Image.Image:
    if save_format == "class_id":
        return Image.fromarray(class_mask.astype(np.uint8), mode="L")
    if save_format == "raw_id":
        raw = CLASS_TO_RAW_LUT[class_mask]  # (H, W) uint16, values like 0, 100, ..., 10000
        return Image.fromarray(raw.astype(np.uint16), mode="I;16")
    if save_format == "color":
        rgb = COLOR_PALETTE[class_mask]
        return Image.fromarray(rgb, mode="RGB")
    raise ValueError(save_format)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    image_size = tuple(args.image_size)

    print(f"device={device}")
    print(f"image_size={image_size}")
    print(f"checkpoints={[str(p) for p in args.checkpoints]}")
    print(f"hflip_tta={args.hflip_tta}")
    print(f"save_format={args.save_format}")

    models = []
    for ckpt in args.checkpoints:
        print(f"loading {ckpt}")
        models.append(load_model(ckpt, args.backbone, args.num_classes, device))
    print(f"loaded {len(models)} model(s); ensemble={'yes' if len(models) > 1 else 'no'}")

    base_h, base_w = image_size
    if args.multiscale:
        aspect = base_w / base_h
        scales = [(_round14(h), _round14(int(round(h * aspect)))) for h in args.multiscale]
        print(f"multiscale={scales}")
    else:
        scales = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_color:
        (args.output_dir / "color").mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        print(f"ERROR: no images found in {args.image_dir}")
        return 1
    print(f"found {len(image_paths)} images")

    start = time.time()
    for idx, img_path in enumerate(image_paths, 1):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_size = img.size  # (W, H)
            probs = predict_one(
                img, models,
                hflip_tta=args.hflip_tta,
                base_size=(base_h, base_w),
                scales=scales,
                device=device,
            )  # (C, H, W) at base_size
        # Resize predictions back to original resolution for submission compliance.
        probs = F.interpolate(
            probs.unsqueeze(0),
            size=(orig_size[1], orig_size[0]),  # (H, W)
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        class_mask = probs.argmax(dim=0).cpu().numpy().astype(np.int64)

        out_path = args.output_dir / img_path.name
        encode_mask(class_mask, args.save_format).save(out_path)

        if args.save_color:
            color_img = Image.fromarray(COLOR_PALETTE[class_mask], mode="RGB")
            color_img.save(args.output_dir / "color" / img_path.name)

        if idx % 50 == 0 or idx == len(image_paths):
            elapsed = time.time() - start
            rate = idx / elapsed
            eta = (len(image_paths) - idx) / rate if rate > 0 else 0
            print(f"  [{idx}/{len(image_paths)}] {rate:.1f} img/s  eta={eta:.0f}s")

    elapsed = time.time() - start
    n = len(image_paths)
    print(f"done: {n} images in {elapsed:.1f}s ({n/elapsed:.1f} img/s, {1000*elapsed/n:.1f} ms/img)")
    print(f"masks written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
