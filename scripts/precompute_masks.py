#!/usr/bin/env python3
"""Convert raw int32 masks to uint8 class-id masks under Segmentation_classid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from seg.constants import RAW_TO_CLASS, RAW_TO_CLASS_LUT


def convert_split(split_dir: Path, overwrite: bool) -> int:
    src_dir = split_dir / "Segmentation"
    dst_dir = split_dir / "Segmentation_classid"
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for src in tqdm(sorted(src_dir.glob("*.png")), desc=f"Precomputing {split_dir.name}", unit="mask"):
        dst = dst_dir / src.name
        if dst.exists() and not overwrite:
            continue
        with Image.open(src) as mask:
            arr = np.asarray(mask)
        unexpected = sorted(set(np.unique(arr).tolist()) - set(RAW_TO_CLASS))
        if unexpected:
            raise ValueError(f"{src} contains unexpected raw ids: {unexpected}")
        Image.fromarray(RAW_TO_CLASS_LUT[arr], mode="L").save(dst)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=Path("Offroad_Segmentation_Training_Dataset"))
    parser.add_argument("--test-root", type=Path, default=Path("Offroad_Segmentation_testImages"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_dirs = [args.train_root / "train", args.train_root / "val", args.test_root]
    total = sum(convert_split(path, args.overwrite) for path in split_dirs)
    print(f"Converted {total} masks to Segmentation_classid/")


if __name__ == "__main__":
    main()
