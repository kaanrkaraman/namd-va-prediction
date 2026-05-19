from __future__ import annotations

import numpy as np
from scipy import stats


def _kernel_matrix(scores_pos: np.ndarray, scores_neg: np.ndarray) -> np.ndarray:
    pos = scores_pos[:, np.newaxis]
    neg = scores_neg[np.newaxis, :]
    k = np.where(neg < pos, 1.0, 0.0) + 0.5 * (neg == pos)
    return np.asarray(k, dtype=float)


def _empirical_auc(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    k = _kernel_matrix(scores_pos, scores_neg)
    auc = float(k.mean())
    v10 = k.mean(axis=1)
    v01 = k.mean(axis=0)
    return auc, v10, v01


def _cov_term(
    v_a: np.ndarray,
    v_b: np.ndarray,
    auc_a: float,
    auc_b: float,
) -> float:
    return float(((v_a - auc_a) * (v_b - auc_b)).sum() / (len(v_a) - 1))


def delong_roc_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> tuple[float, float]:
    yt = np.asarray(y_true).astype(bool)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)
    if not (yt.shape == pa.shape == pb.shape):
        raise ValueError(
            f"Shape mismatch: y_true {yt.shape}, pred_a {pa.shape}, pred_b {pb.shape}"
        )
    if yt.ndim != 1:
        raise ValueError(f"All inputs must be 1-D; got {yt.ndim}-D")

    pos = yt
    neg = ~yt
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Need at least 2 positives and 2 negatives; got n_pos={n_pos}, n_neg={n_neg}"
        )

    auc_a, v10_a, v01_a = _empirical_auc(pa[pos], pa[neg])
    auc_b, v10_b, v01_b = _empirical_auc(pb[pos], pb[neg])

    var_a = (
        _cov_term(v10_a, v10_a, auc_a, auc_a) / n_pos
        + _cov_term(v01_a, v01_a, auc_a, auc_a) / n_neg
    )
    var_b = (
        _cov_term(v10_b, v10_b, auc_b, auc_b) / n_pos
        + _cov_term(v01_b, v01_b, auc_b, auc_b) / n_neg
    )
    cov_ab = (
        _cov_term(v10_a, v10_b, auc_a, auc_b) / n_pos
        + _cov_term(v01_a, v01_b, auc_a, auc_b) / n_neg
    )

    denom_squared = var_a + var_b - 2 * cov_ab
    if denom_squared <= 0:
        if abs(auc_a - auc_b) < 1e-12:
            return 0.0, 1.0
        raise FloatingPointError(
            "Non-positive DeLong variance estimate: "
            f"var_a={var_a} var_b={var_b} cov={cov_ab}"
        )

    z = (auc_a - auc_b) / float(np.sqrt(denom_squared))
    p_two_sided = 2.0 * float(stats.norm.sf(abs(z)))
    return z, p_two_sided
