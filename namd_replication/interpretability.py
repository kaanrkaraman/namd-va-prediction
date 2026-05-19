from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from autogluon.tabular import TabularPredictor
from xgboost import XGBClassifier

from namd_replication.data.encoder import (
    AGEGROUP_COLUMN_NAMES,
    ETHNICITY_LEVELS,
    FeatureEncoder,
)
from namd_replication.models.autogluon_model import (
    predict_proba_above,
    select_autogluon_columns,
)

log = logging.getLogger(__name__)


PAPER_FEATURES: tuple[str, ...] = (
    "baseline_va",
    "agegroup",
    "ethnicity",
    "gender",
    "vol_irf",
    "vol_rpe",
    "vol_srf",
    "vol_ped",
    "vol_shrm",
    "vol_hrf",
)

ENCODED_TO_PAPER: dict[str, str] = {
    "baseline_va": "baseline_va",
    "vol_irf": "vol_irf",
    "vol_rpe": "vol_rpe",
    "vol_srf": "vol_srf",
    "vol_ped": "vol_ped",
    "vol_shrm": "vol_shrm",
    "vol_hrf": "vol_hrf",
    "gender_male": "gender",
}
for _lvl in ETHNICITY_LEVELS:
    ENCODED_TO_PAPER[f"ethnicity_{_lvl}"] = "ethnicity"
for _col in AGEGROUP_COLUMN_NAMES:
    ENCODED_TO_PAPER[_col] = "agegroup"


PredictRawFn = Callable[[pd.DataFrame], np.ndarray]


def make_xgb_raw_predictor(
    model: XGBClassifier,
    encoder: FeatureEncoder,
) -> PredictRawFn:
    def predict_raw(df_raw: pd.DataFrame) -> np.ndarray:
        x = encoder.transform(df_raw)
        proba = model.predict_proba(x)[:, 1]
        return np.asarray(proba, dtype=float)

    return predict_raw


def make_autogluon_raw_predictor(predictor: TabularPredictor) -> PredictRawFn:
    def predict_raw(df_raw: pd.DataFrame) -> np.ndarray:
        df_ag = select_autogluon_columns(df_raw, include_label=False)
        return predict_proba_above(predictor, df_ag)

    return predict_raw


def _shap_values_array(model: XGBClassifier, x_test: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_test)
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values, dtype=float)


def compute_xgb_shap_grouped_importance(
    model: XGBClassifier,
    x_test: pd.DataFrame,
) -> dict[str, float]:
    values = _shap_values_array(model, x_test)

    importance: dict[str, float] = {}
    for paper_feat in PAPER_FEATURES:
        cols = [c for c, p in ENCODED_TO_PAPER.items() if p == paper_feat]
        col_idx = np.asarray([x_test.columns.get_loc(c) for c in cols], dtype=int)
        net_per_sample = values[:, col_idx].sum(axis=1)
        importance[paper_feat] = float(np.mean(np.abs(net_per_sample)))
    return importance


def compute_autogluon_permutation_importance(
    predictor: TabularPredictor,
    test_df_with_label: pd.DataFrame,
    num_shuffle_sets: int = 10,
) -> dict[str, float]:
    fi = predictor.feature_importance(
        data=test_df_with_label,
        num_shuffle_sets=num_shuffle_sets,
    )
    importance: dict[str, float] = {}
    for paper_feat in PAPER_FEATURES:
        if paper_feat in fi.index:
            val = float(fi.loc[paper_feat, "importance"])
            importance[paper_feat] = max(val, 0.0)
        else:
            importance[paper_feat] = 0.0
    return importance


def normalize_to_rfi(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    if total <= 0:
        return {k: 1.0 / len(raw) for k in raw}
    return {k: v / total for k, v in raw.items()}


def plot_feature_importance_comparison(
    xgb_rfi: dict[str, float],
    ag_rfi: dict[str, float],
    out_path: Path,
) -> None:
    features_sorted = sorted(PAPER_FEATURES, key=lambda f: xgb_rfi[f], reverse=True)
    xgb_vals = [xgb_rfi[f] for f in features_sorted]
    ag_vals = [ag_rfi[f] for f in features_sorted]

    y = np.arange(len(features_sorted))
    h = 0.4

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y - h / 2, xgb_vals, h, label="XGBoost (bespoke)", color="#1f77b4")
    ax.barh(y + h / 2, ag_vals, h, label="AutoGluon (AutoML)", color="#ff7f0e")
    ax.set_yticks(y)
    ax.set_yticklabels(features_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Relative feature importance (sum = 1.0)")
    ax.set_title("Feature importance: XGBoost vs AutoGluon")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary_xgboost(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    out_path: Path,
) -> None:
    values = _shap_values_array(model, x_test)

    plt.figure(figsize=(8, 8))
    shap.summary_plot(values, x_test, show=False)
    plt.title("XGBoost SHAP beeswarm (test set, 17 encoded features)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_pdp_continuous(
    predict_raw: PredictRawFn,
    df_raw: pd.DataFrame,
    feature: str,
    grid: np.ndarray,
) -> np.ndarray:
    pdp = np.empty(len(grid), dtype=float)
    for i, v in enumerate(grid):
        df_v = df_raw.copy()
        df_v[feature] = v
        pdp[i] = float(predict_raw(df_v).mean())
    return pdp


def compute_pdp_categorical(
    predict_raw: PredictRawFn,
    df_raw: pd.DataFrame,
    feature: str,
    levels: list[str],
) -> list[float]:
    pdp: list[float] = []
    for lvl in levels:
        df_v = df_raw.copy()
        df_v[feature] = lvl
        pdp.append(float(predict_raw(df_v).mean()))
    return pdp


def plot_pdp_continuous(
    grid: np.ndarray,
    xgb_pdp: np.ndarray,
    ag_pdp: np.ndarray,
    feature_label: str,
    out_path: Path,
    threshold: float = 0.5,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid, xgb_pdp, "-o", color="#1f77b4", label="XGBoost", markersize=4)
    ax.plot(grid, ag_pdp, "-s", color="#ff7f0e", label="AutoGluon", markersize=4)
    ax.axhline(
        threshold,
        color="black",
        linewidth=0.8,
        linestyle="--",
        label=f"Threshold ({threshold})",
    )
    ax.set_xlabel(feature_label)
    ax.set_ylabel("Mean P(Above) across test set")
    ax.set_title(f"Partial dependence: {feature_label}")
    ax.legend(loc="best")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_ylim(0.0, 1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pdp_categorical(
    levels: list[str],
    xgb_pdp: list[float],
    ag_pdp: list[float],
    feature_label: str,
    out_path: Path,
    threshold: float = 0.5,
) -> None:
    x = np.arange(len(levels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, xgb_pdp, w, color="#1f77b4", label="XGBoost")
    ax.bar(x + w / 2, ag_pdp, w, color="#ff7f0e", label="AutoGluon")
    ax.axhline(
        threshold,
        color="black",
        linewidth=0.8,
        linestyle="--",
        label=f"Threshold ({threshold})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=20, ha="right")
    ax.set_ylabel("Mean P(Above) across test set")
    ax.set_title(f"Partial dependence: {feature_label}")
    ax.legend(loc="best")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
