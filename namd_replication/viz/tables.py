from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

CONTINUOUS_FEATURES: tuple[str, ...] = (
    "baseline_va",
    "vol_irf",
    "vol_rpe",
    "vol_srf",
    "vol_ped",
    "vol_shrm",
    "vol_hrf",
)


@dataclass(frozen=True)
class CategoricalSpec:
    column: str
    levels: tuple[str, ...]


CATEGORICAL_SPECS: tuple[CategoricalSpec, ...] = (
    CategoricalSpec("gender", ("Male", "Female")),
    CategoricalSpec("ethnicity", ("White", "Asian", "Black", "Other", "Unknown")),
    CategoricalSpec(
        "agegroup",
        (
            "50-59 years",
            "60-69 years",
            "70-79 years",
            "80 years and above",
        ),
    ),
)


def _format_continuous(values: pd.Series, dp: int) -> str:
    if len(values) == 0:
        return "n/a"
    median = float(values.median())
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return f"{median:.{dp}f} ({q1:.{dp}f}-{q3:.{dp}f})"


def _format_categorical(count: int, total: int) -> str:
    pct = 100.0 * count / total if total > 0 else 0.0
    return f"{count} ({pct:.1f}%)"


def generate_table1(
    cohort: pd.DataFrame,
    outcome_col: str,
    out_path: Path | None = None,
) -> pd.DataFrame:
    if outcome_col not in cohort.columns:
        raise KeyError(f"Outcome column {outcome_col!r} not in cohort")

    above = cohort[cohort[outcome_col] == "Above"]
    below = cohort[cohort[outcome_col] == "Below"]
    if len(above) == 0 or len(below) == 0:
        raise ValueError(
            f"Need at least one row per outcome class; "
            f"got n_above={len(above)} n_below={len(below)}"
        )

    rows: list[dict[str, str | float]] = []

    for col in CONTINUOUS_FEATURES:
        if col not in cohort.columns:
            continue
        dp = 5 if col == "vol_hrf" else 2
        _, p = mannwhitneyu(above[col], below[col], alternative="two-sided")
        rows.append(
            {
                "variable": col,
                "level": "",
                "total": _format_continuous(cohort[col], dp),
                "above": _format_continuous(above[col], dp),
                "below": _format_continuous(below[col], dp),
                "p_value": float(p),
                "test": "Mann-Whitney U",
            }
        )

    for spec in CATEGORICAL_SPECS:
        if spec.column not in cohort.columns:
            continue
        for level in spec.levels:
            n_total = int((cohort[spec.column] == level).sum())
            n_above = int((above[spec.column] == level).sum())
            n_below = int((below[spec.column] == level).sum())
            table = [
                [n_above, n_below],
                [len(above) - n_above, len(below) - n_below],
            ]
            _, p = fisher_exact(table, alternative="two-sided")
            rows.append(
                {
                    "variable": spec.column,
                    "level": level,
                    "total": _format_categorical(n_total, len(cohort)),
                    "above": _format_categorical(n_above, len(above)),
                    "below": _format_categorical(n_below, len(below)),
                    "p_value": float(p),
                    "test": "Fisher's exact",
                }
            )

    table_df = pd.DataFrame(rows)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(out_path, index=False)

    return table_df
