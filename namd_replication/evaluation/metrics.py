from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class PerformanceReport:
    auroc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    accuracy: float
    f1: float
    threshold: float


def _safe_divide(num: float, denom: float) -> float:
    return float(num / denom) if denom > 0 else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> PerformanceReport:
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=float)
    if yt.shape != ys.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape}, y_score {ys.shape}")
    if yt.ndim != 1:
        raise ValueError(f"Inputs must be 1-D; got {yt.ndim}-D")
    if not set(np.unique(yt).tolist()).issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0/1; got {np.unique(yt)}")

    y_pred = (ys >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()

    return PerformanceReport(
        auroc=float(roc_auc_score(yt, ys)),
        sensitivity=_safe_divide(tp, tp + fn),
        specificity=_safe_divide(tn, tn + fp),
        ppv=_safe_divide(tp, tp + fp),
        npv=_safe_divide(tn, tn + fn),
        accuracy=float(accuracy_score(yt, y_pred)),
        f1=float(f1_score(yt, y_pred, zero_division=0)),
        threshold=float(threshold),
    )
