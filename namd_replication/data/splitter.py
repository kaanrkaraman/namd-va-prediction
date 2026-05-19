from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    outcome_col: str,
    train_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if outcome_col not in df.columns:
        raise KeyError(
            f"Outcome column {outcome_col!r} not in frame columns {list(df.columns)}"
        )
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}")

    train_idx, test_idx = train_test_split(
        df.index,
        train_size=train_fraction,
        stratify=df[outcome_col],
        random_state=seed,
        shuffle=True,
    )
    train = cast(pd.DataFrame, df.loc[train_idx].copy())
    test = cast(pd.DataFrame, df.loc[test_idx].copy())
    return train, test


def persist_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet")
    test.to_parquet(out_dir / "test.parquet")
    manifest = {
        "n_train": len(train),
        "n_test": len(test),
        "train_indices": [_jsonable(i) for i in train.index],
        "test_indices": [_jsonable(i) for i in test.index],
    }
    (out_dir / "split_indices.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _jsonable(idx: object) -> int | str:
    if isinstance(idx, int | bool):
        return int(idx)
    return str(idx)
