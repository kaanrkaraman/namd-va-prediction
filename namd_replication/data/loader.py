from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from namd_replication.config import settings

log = logging.getLogger(__name__)


ETHNICITY_MAP: dict[str, str] = {
    "caucasian": "White",
    "asian": "Asian",
    "afrocaribbean": "Black",
    "Other": "Other",
    "unknown": "Unknown",
}


@dataclass(frozen=True)
class CohortStep:
    label: str
    n_eyes: int
    n_patients: int


def _n_eyes(df: pd.DataFrame) -> int:
    return int(df[["anonid", "eye"]].drop_duplicates().shape[0])


def _n_patients(df: pd.DataFrame) -> int:
    return int(df["anonid"].nunique())


def load_dryad_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    log.info("Loaded raw CSV: %d rows by %d cols", len(df), df.shape[1])
    return df


def assign_timepoint_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank"] = (
        out.groupby(["anonid", "eye"])["time"].rank(method="first").astype(int)
    )
    return out


def filter_complete_timepoints(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby(["anonid", "eye"]).size().rename("n_rows").reset_index()
    keep = sizes[sizes["n_rows"] == 5][["anonid", "eye"]]
    return df.merge(keep, on=["anonid", "eye"], how="inner")


def filter_valid_baseline(df: pd.DataFrame) -> pd.DataFrame:
    rank1 = df[df["rank"] == 1]
    keep = rank1[rank1["time"] == 0][["anonid", "eye"]]
    return df.merge(keep, on=["anonid", "eye"], how="inner")


def filter_followup_window(
    df: pd.DataFrame,
    min_days: int,
    max_days: int,
) -> pd.DataFrame:
    rank5 = df[df["rank"] == 5]
    keep = rank5[(rank5["time"] >= min_days) & (rank5["time"] <= max_days)][
        ["anonid", "eye"]
    ]
    return df.merge(keep, on=["anonid", "eye"], how="inner")


def pivot_to_eye_level(df: pd.DataFrame, va_threshold: int) -> pd.DataFrame:
    baseline = df[df["rank"] == 1].set_index(["anonid", "eye"])
    follow = df[df["rank"] == 5].set_index(["anonid", "eye"])

    out = pd.DataFrame(index=baseline.index)
    out["agegroup"] = baseline["agegroup"]
    out["gender"] = baseline["gender"]
    out["ethnicity_raw"] = baseline["ethnicity"]
    out["ethnicity"] = baseline["ethnicity"].map(ETHNICITY_MAP)
    out["baseline_va"] = baseline["va"]
    out["vol_irf"] = baseline["vol_irf"]
    out["vol_rpe"] = baseline["vol_rpe"]
    out["vol_srf"] = baseline["vol_srf"]
    out["vol_ped"] = baseline["vol_ped"]
    out["vol_shrm"] = baseline["vol_shrm"]
    out["vol_hrf"] = baseline["vol_intrarethyperreflect"]
    out["followup_va"] = follow["va"]
    out["followup_days"] = follow["time"]
    out["outcome"] = (
        (out["followup_va"] >= va_threshold)
        .map({True: "Above", False: "Below"})
        .astype("string")
    )

    out = out.reset_index()
    out["eye_key"] = out["anonid"].astype(str) + ":" + out["eye"].astype(str)

    column_order = [
        "eye_key",
        "anonid",
        "eye",
        "agegroup",
        "gender",
        "ethnicity_raw",
        "ethnicity",
        "baseline_va",
        "vol_irf",
        "vol_rpe",
        "vol_srf",
        "vol_ped",
        "vol_shrm",
        "vol_hrf",
        "followup_va",
        "followup_days",
        "outcome",
    ]
    return out.loc[:, column_order].reset_index(drop=True)


def build_analysis_cohort(
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, list[CohortStep]]:
    path = csv_path if csv_path is not None else settings.data_raw_dir / "dataframe.csv"

    steps: list[CohortStep] = []

    raw = load_dryad_raw(path)
    steps.append(CohortStep("Dryad CSV loaded", _n_eyes(raw), _n_patients(raw)))

    ranked = assign_timepoint_rank(raw)

    complete = filter_complete_timepoints(ranked)
    steps.append(
        CohortStep("Has all 5 timepoint rows", _n_eyes(complete), _n_patients(complete))
    )

    with_baseline = filter_valid_baseline(complete)
    steps.append(
        CohortStep(
            "Rank-1 row at time == 0 (true baseline)",
            _n_eyes(with_baseline),
            _n_patients(with_baseline),
        )
    )

    in_window = filter_followup_window(
        with_baseline,
        settings.followup_window_min_days,
        settings.followup_window_max_days,
    )
    steps.append(
        CohortStep(
            f"Rank-5 time in [{settings.followup_window_min_days}, "
            f"{settings.followup_window_max_days}] days",
            _n_eyes(in_window),
            _n_patients(in_window),
        )
    )

    cohort = pivot_to_eye_level(in_window, settings.va_threshold_etdrs)
    steps.append(
        CohortStep(
            "Pivot to one row per eye",
            len(cohort),
            int(cohort["anonid"].nunique()),
        )
    )

    for step in steps:
        log.info(
            "step: %-44s  n_eyes=%4d  n_patients=%4d",
            step.label,
            step.n_eyes,
            step.n_patients,
        )

    return cohort, steps
