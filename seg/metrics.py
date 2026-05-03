"""Segmentation metrics."""

from __future__ import annotations

import numpy as np


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (target >= 0) & (target < num_classes)
    encoded = num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_from_confusion(cm: np.ndarray) -> tuple[float, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    per_class = np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)
    return float(np.nanmean(per_class)), per_class


def dice_from_confusion(cm: np.ndarray) -> tuple[float, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    per_class = np.divide(2 * tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)
    return float(np.nanmean(per_class)), per_class

