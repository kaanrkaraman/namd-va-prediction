from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

log = logging.getLogger(__name__)


AUTOGLUON_INPUT_COLUMNS: tuple[str, ...] = (
    "agegroup",
    "gender",
    "ethnicity",
    "baseline_va",
    "vol_irf",
    "vol_rpe",
    "vol_srf",
    "vol_ped",
    "vol_shrm",
    "vol_hrf",
)

AUTOGLUON_LABEL_COLUMN: str = "outcome"
AUTOGLUON_POSITIVE_CLASS: str = "Above"

FAST_PRESET: str = "medium_quality"
GRID_PRESET: str = "best_quality"

DEFAULT_TIME_LIMIT_FAST: int = 120
DEFAULT_TIME_LIMIT_GRID: int = 1800

EXCLUDED_MODEL_TYPES: tuple[str, ...] = ("XGB",)


def select_autogluon_columns(df: pd.DataFrame, *, include_label: bool) -> pd.DataFrame:
    required = set(AUTOGLUON_INPUT_COLUMNS)
    if include_label:
        required.add(AUTOGLUON_LABEL_COLUMN)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Frame is missing required columns: {sorted(missing)}")

    columns = list(AUTOGLUON_INPUT_COLUMNS)
    if include_label:
        columns.append(AUTOGLUON_LABEL_COLUMN)
    return df.loc[:, columns].copy()


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_autogluon(
    train_df: pd.DataFrame,
    *,
    seed: int,
    mode: Literal["fast", "grid"],
    model_dir: Path,
    time_limit: int | None = None,
) -> TabularPredictor:
    if AUTOGLUON_LABEL_COLUMN not in train_df.columns:
        raise KeyError(
            f"train_df must contain label column {AUTOGLUON_LABEL_COLUMN!r}; "
            f"got {list(train_df.columns)}"
        )

    if mode == "fast":
        preset = FAST_PRESET
        limit = time_limit if time_limit is not None else DEFAULT_TIME_LIMIT_FAST
    elif mode == "grid":
        preset = GRID_PRESET
        limit = time_limit if time_limit is not None else DEFAULT_TIME_LIMIT_GRID
    else:
        raise ValueError(f"mode must be 'fast' or 'grid'; got {mode!r}")

    _seed_everything(seed)

    model_dir.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label=AUTOGLUON_LABEL_COLUMN,
        path=str(model_dir),
        problem_type="binary",
        eval_metric="roc_auc",
        verbosity=2,
    )

    log.info(
        "AutoGluon fit: preset=%s time_limit=%ds n_train=%d",
        preset,
        limit,
        len(train_df),
    )
    predictor.fit(
        train_data=train_df,
        presets=preset,
        time_limit=limit,
        excluded_model_types=list(EXCLUDED_MODEL_TYPES),
        ag_args_fit={"random_seed": seed},
    )
    return predictor


def predict_proba_above(predictor: TabularPredictor, df: pd.DataFrame) -> np.ndarray:
    if AUTOGLUON_POSITIVE_CLASS not in predictor.class_labels:
        raise RuntimeError(
            f"Predictor's class labels {predictor.class_labels} do not include "
            f"{AUTOGLUON_POSITIVE_CLASS!r}"
        )
    proba = predictor.predict_proba(df, as_pandas=True, as_multiclass=True)
    return np.asarray(proba[AUTOGLUON_POSITIVE_CLASS].to_numpy(), dtype=float)


def save_predictor(predictor: TabularPredictor, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.save()


def load_predictor(model_dir: Path) -> TabularPredictor:
    loaded: Any = TabularPredictor.load(str(model_dir))
    if not isinstance(loaded, TabularPredictor):
        raise TypeError(f"Expected TabularPredictor, got {type(loaded).__name__}")
    return loaded
