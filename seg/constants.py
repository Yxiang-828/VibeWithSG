"""Shared label metadata for the offroad segmentation task."""

from __future__ import annotations

import numpy as np

RAW_TO_CLASS = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    600: 6,
    700: 7,
    800: 8,
    7100: 9,
    10000: 10,
}

CLASS_TO_RAW = {v: k for k, v in RAW_TO_CLASS.items()}

CLASS_NAMES = [
    "Background",
    "Trees",
    "Lush Bushes",
    "Dry Grass",
    "Dry Bushes",
    "Ground Clutter",
    "Flowers",
    "Logs",
    "Rocks",
    "Landscape",
    "Sky",
]

NUM_CLASSES = len(CLASS_NAMES)

COLOR_PALETTE = np.array(
    [
        [0, 0, 0],
        [34, 139, 34],
        [0, 220, 0],
        [210, 180, 80],
        [139, 90, 43],
        [128, 128, 0],
        [255, 0, 255],
        [139, 69, 19],
        [128, 128, 128],
        [160, 82, 45],
        [135, 206, 235],
    ],
    dtype=np.uint8,
)

MAX_RAW_ID = max(RAW_TO_CLASS)
RAW_TO_CLASS_LUT = np.zeros(MAX_RAW_ID + 1, dtype=np.uint8)
for raw_id, class_id in RAW_TO_CLASS.items():
    RAW_TO_CLASS_LUT[raw_id] = class_id

CLASS_TO_RAW_LUT = np.zeros(NUM_CLASSES, dtype=np.uint16)
for class_id, raw_id in CLASS_TO_RAW.items():
    CLASS_TO_RAW_LUT[class_id] = raw_id

