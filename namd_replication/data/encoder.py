from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

ETHNICITY_LEVELS: tuple[str, ...] = ("White", "Asian", "Black", "Other", "Unknown")

AGEGROUP_TO_COLUMN: dict[str, str] = {
    "50-59 years": "age_50_59",
    "60-69 years": "age_60_69",
    "70-79 years": "age_70_79",
    "80 years and above": "age_80_plus",
}
AGEGROUP_LEVELS: tuple[str, ...] = tuple(AGEGROUP_TO_COLUMN.keys())
AGEGROUP_COLUMN_NAMES: tuple[str, ...] = tuple(AGEGROUP_TO_COLUMN.values())

NUMERIC_PASS_THROUGH: tuple[str, ...] = (
    "baseline_va",
    "vol_irf",
    "vol_rpe",
    "vol_srf",
    "vol_ped",
    "vol_shrm",
    "vol_hrf",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    *NUMERIC_PASS_THROUGH,
    "gender_male",
    *[f"ethnicity_{lvl}" for lvl in ETHNICITY_LEVELS],
    *AGEGROUP_COLUMN_NAMES,
)


@dataclass
class FeatureEncoder:
    is_fit: bool = False
    ethnicity_levels: tuple[str, ...] = field(default_factory=lambda: ETHNICITY_LEVELS)
    agegroup_levels: tuple[str, ...] = field(default_factory=lambda: AGEGROUP_LEVELS)

    def fit(self, df: pd.DataFrame) -> FeatureEncoder:
        _validate_required_columns(df)
        unknown_e = set(df["ethnicity"].unique()) - set(self.ethnicity_levels)
        if unknown_e:
            raise ValueError(f"Unknown ethnicity levels: {sorted(unknown_e)}")
        unknown_a = set(df["agegroup"].unique()) - set(self.agegroup_levels)
        if unknown_a:
            raise ValueError(f"Unknown agegroup levels: {sorted(unknown_a)}")
        unknown_g = set(df["gender"].unique()) - {"Male", "Female"}
        if unknown_g:
            raise ValueError(f"Unknown gender levels: {sorted(unknown_g)}")
        self.is_fit = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fit:
            raise RuntimeError("FeatureEncoder.transform called before fit")
        _validate_required_columns(df)

        out = pd.DataFrame(index=df.index)
        for col in NUMERIC_PASS_THROUGH:
            out[col] = df[col].astype(float)
        out["gender_male"] = (df["gender"] == "Male").astype(int)
        for lvl in self.ethnicity_levels:
            out[f"ethnicity_{lvl}"] = (df["ethnicity"] == lvl).astype(int)
        for lvl, colname in AGEGROUP_TO_COLUMN.items():
            out[colname] = (df["agegroup"] == lvl).astype(int)
        return out.loc[:, list(OUTPUT_COLUMNS)]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> FeatureEncoder:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected FeatureEncoder, got {type(loaded).__name__}")
        return loaded


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = {"agegroup", "gender", "ethnicity", *NUMERIC_PASS_THROUGH}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Cohort frame is missing required columns: {sorted(missing)}")


def encode_outcome(df: pd.DataFrame) -> pd.Series:
    if "outcome" not in df.columns:
        raise KeyError("Cohort frame is missing the 'outcome' column")
    mapping = {"Above": 1, "Below": 0}
    encoded = df["outcome"].map(mapping)
    if encoded.isna().any():
        unknown = sorted(set(df.loc[encoded.isna(), "outcome"].unique()))
        raise ValueError(f"Unknown outcome values: {unknown}")
    return encoded.astype(int)
