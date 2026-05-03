"""Datasets and mask conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from seg.constants import RAW_TO_CLASS_LUT


def convert_raw_mask_to_class_ids(mask: Image.Image) -> Image.Image:
    arr = np.asarray(mask)
    if arr.max(initial=0) >= len(RAW_TO_CLASS_LUT):
        bad = sorted(set(np.unique(arr).tolist()) - set(range(len(RAW_TO_CLASS_LUT))))
        raise ValueError(f"Mask contains raw ids outside LUT range: {bad[:20]}")
    return Image.fromarray(RAW_TO_CLASS_LUT[arr], mode="L")


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    from seg.constants import COLOR_PALETTE

    return COLOR_PALETTE[mask.astype(np.uint8)]


class OffroadSegmentationDataset:
    """Lazy PIL dataset used by the torch scripts.

    The class avoids importing torch at module import time so audit/precompute
    scripts work before the training environment is installed.
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_transform: Callable | None = None,
        mask_transform: Callable | None = None,
        use_precomputed_masks: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "Color_Images"
        classid_dir = self.data_dir / "Segmentation_classid"
        raw_dir = self.data_dir / "Segmentation"
        self.mask_dir = classid_dir if use_precomputed_masks and classid_dir.exists() else raw_dir
        self.uses_precomputed_masks = self.mask_dir.name == "Segmentation_classid"
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.data_ids = sorted(p.name for p in self.image_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.data_ids)

    def __getitem__(self, idx: int):
        data_id = self.data_ids[idx]
        image_path = self.image_dir / data_id
        mask_path = self.mask_dir / data_id

        with Image.open(image_path) as im:
            image = im.convert("RGB")
        with Image.open(mask_path) as im:
            mask = im.copy()

        if not self.uses_precomputed_masks:
            mask = convert_raw_mask_to_class_ids(mask)

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask

