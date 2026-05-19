from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NAMD_", frozen=True)

    project_root: Path = _project_root()
    data_raw_dir: Path = _project_root() / "data" / "raw"
    data_processed_dir: Path = _project_root() / "data" / "processed"
    data_interim_dir: Path = _project_root() / "data" / "interim"
    outputs_dir: Path = _project_root() / "outputs"

    random_seed: int = 42
    train_fraction: float = 0.85
    va_threshold_etdrs: int = 70

    followup_window_min_days: int = 335
    followup_window_max_days: int = 395
    followup_target_days: int = 365


settings = Settings()
