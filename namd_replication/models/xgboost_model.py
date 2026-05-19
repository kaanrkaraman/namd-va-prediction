from __future__ import annotations

import logging
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

log = logging.getLogger(__name__)


SUPP_TABLE_4_HPARAMS: dict[str, Any] = {
    "n_estimators": 50,
    "learning_rate": 0.1,
    "max_depth": 2,
    "min_child_weight": 3.5,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "gamma": 0.8,
    "objective": "binary:logistic",
}

COARSE_GRID: dict[str, list[Any]] = {
    "objective": ["binary:logistic"],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [1, 2, 3],
    "min_child_weight": [1, 2, 3],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "n_estimators": [5, 10, 50, 75, 100],
    "gamma": [0.0, 1.0, 2.0, 3.0],
}

FINE_GRID: dict[str, list[Any]] = {
    "objective": ["binary:logistic"],
    "learning_rate": [0.08, 0.1, 0.12],
    "max_depth": [1, 2, 3],
    "min_child_weight": [2.5, 3, 3.5],
    "subsample": [0.95, 1.0],
    "colsample_bytree": [0.75, 0.8, 0.85],
    "n_estimators": [40, 50, 60],
    "gamma": [0.8, 1.0, 1.2],
}


def _grid_size(grid: dict[str, list[Any]]) -> int:
    return reduce(mul, (len(v) for v in grid.values()), 1)


def make_classifier(seed: int, n_jobs: int = 1) -> XGBClassifier:
    return XGBClassifier(
        **SUPP_TABLE_4_HPARAMS,
        random_state=seed,
        n_jobs=n_jobs,
        eval_metric="logloss",
    )


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    n_jobs: int = 1,
) -> XGBClassifier:
    model = make_classifier(seed=seed, n_jobs=n_jobs)
    model.fit(x_train, y_train)
    return model


def grid_search_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    coarse_grid: dict[str, list[Any]] | None = None,
    fine_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 10,
    n_jobs: int = -1,
) -> tuple[XGBClassifier, dict[str, Any]]:
    coarse = coarse_grid if coarse_grid is not None else COARSE_GRID
    fine = fine_grid if fine_grid is not None else FINE_GRID

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base = XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss")

    log.info(
        "Coarse grid: %d combinations x %d-fold CV = %d fits",
        _grid_size(coarse),
        n_splits,
        _grid_size(coarse) * n_splits,
    )
    coarse_search = GridSearchCV(
        base,
        coarse,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        refit=False,
        verbose=0,
    )
    coarse_search.fit(x_train, y_train)
    log.info(
        "Coarse winner CV AUROC=%.4f  params=%s",
        coarse_search.best_score_,
        coarse_search.best_params_,
    )

    log.info(
        "Fine grid: %d combinations x %d-fold CV = %d fits",
        _grid_size(fine),
        n_splits,
        _grid_size(fine) * n_splits,
    )
    fine_search = GridSearchCV(
        base,
        fine,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    fine_search.fit(x_train, y_train)
    log.info(
        "Fine winner CV AUROC=%.4f  params=%s",
        fine_search.best_score_,
        fine_search.best_params_,
    )

    info: dict[str, Any] = {
        "coarse_best_params": coarse_search.best_params_,
        "coarse_best_cv_auroc": float(coarse_search.best_score_),
        "fine_best_params": fine_search.best_params_,
        "fine_best_cv_auroc": float(fine_search.best_score_),
        "n_splits": n_splits,
        "n_combinations_coarse": _grid_size(coarse),
        "n_combinations_fine": _grid_size(fine),
    }
    return fine_search.best_estimator_, info


def save_model(model: XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(path))
    return model
