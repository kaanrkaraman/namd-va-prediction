from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import sys
import time
from importlib.metadata import version

import pandas as pd

from namd_replication.config import settings
from namd_replication.data.encoder import encode_outcome
from namd_replication.evaluation.metrics import compute_metrics
from namd_replication.models.autogluon_model import (
    AUTOGLUON_INPUT_COLUMNS,
    DEFAULT_TIME_LIMIT_FAST,
    DEFAULT_TIME_LIMIT_GRID,
    FAST_PRESET,
    GRID_PRESET,
    predict_proba_above,
    select_autogluon_columns,
    train_autogluon,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("train_autogluon")


PAPER_TARGETS_TABLE_2_AUTOML: dict[str, float] = {
    "auroc": 0.849,
    "sensitivity": 0.69,
    "specificity": 0.821,
    "ppv": 0.726,
    "npv": 0.793,
    "accuracy": 0.767,
    "f1": 0.71,
}


def _library_versions() -> dict[str, str]:
    libs = [
        "autogluon.tabular",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "xgboost",
        "lightgbm",
    ]
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for lib in libs:
        try:
            out[lib] = version(lib)
        except Exception:
            out[lib] = "not-installed"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["fast", "grid"],
        default="fast",
        help="fast = medium_quality preset; grid = best_quality preset",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Override AutoGluon fit time limit in seconds",
    )
    args = parser.parse_args()

    train_path = settings.data_processed_dir / "train.parquet"
    test_path = settings.data_processed_dir / "test.parquet"
    for p in (train_path, test_path):
        if not p.is_file():
            log.error("Missing %s. Run scripts/02_build_cohort.py first.", p)
            return 1

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    train_ag = select_autogluon_columns(train, include_label=True)
    test_ag = select_autogluon_columns(test, include_label=False)
    y_test = encode_outcome(test)

    log.info(
        "Loaded train=%d rows  test=%d rows  AutoGluon features=%d",
        len(train_ag),
        len(test_ag),
        len(AUTOGLUON_INPUT_COLUMNS),
    )
    log.info(
        "Outcome distribution: train Above=%d Below=%d  test Above=%d Below=%d",
        int((train["outcome"] == "Above").sum()),
        int((train["outcome"] == "Below").sum()),
        int(y_test.sum()),
        int((y_test == 0).sum()),
    )

    model_dir = settings.outputs_dir / "models" / "autogluon"
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fast":
        preset = FAST_PRESET
        limit = (
            args.time_limit if args.time_limit is not None else DEFAULT_TIME_LIMIT_FAST
        )
    else:
        preset = GRID_PRESET
        limit = (
            args.time_limit if args.time_limit is not None else DEFAULT_TIME_LIMIT_GRID
        )

    log.info("Mode: %s (preset=%s, time_limit=%ds)", args.mode, preset, limit)

    t0 = time.perf_counter()
    predictor = train_autogluon(
        train_ag,
        seed=settings.random_seed,
        mode=args.mode,
        model_dir=model_dir,
        time_limit=limit,
    )
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.2f seconds", elapsed)

    feature_names = sorted(predictor.feature_metadata_in.get_features())
    expected = sorted(AUTOGLUON_INPUT_COLUMNS)
    if feature_names != expected:
        log.error(
            "Leakage guardrail failed: predictor saw features %s, expected %s",
            feature_names,
            expected,
        )
        return 2
    log.info(
        "Leakage guardrail OK: %d features = %s", len(feature_names), feature_names
    )

    leaderboard = predictor.leaderboard(silent=True)
    log.info("\nAutoGluon leaderboard:\n%s", leaderboard.to_string(index=False))

    y_score = predict_proba_above(predictor, test_ag)
    report = compute_metrics(y_test.to_numpy(), y_score)
    log.info(
        "Test: AUROC=%.4f  Sens=%.3f  Spec=%.3f  PPV=%.3f  NPV=%.3f  Acc=%.3f  F1=%.3f",
        report.auroc,
        report.sensitivity,
        report.specificity,
        report.ppv,
        report.npv,
        report.accuracy,
        report.f1,
    )

    pred_dir = settings.outputs_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "autogluon_test_predictions.parquet"
    pd.DataFrame(
        {
            "eye_key": test["eye_key"].to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_score": y_score,
        }
    ).to_parquet(pred_path)
    log.info("Wrote test predictions -> %s", pred_path)

    leaderboard_path = settings.outputs_dir / "tables" / "autogluon_leaderboard.csv"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(leaderboard_path, index=False)
    log.info("Wrote leaderboard -> %s", leaderboard_path)

    manifest = {
        "mode": args.mode,
        "preset": preset,
        "time_limit_seconds": limit,
        "seed": settings.random_seed,
        "n_train": len(train_ag),
        "n_test": len(test_ag),
        "n_features": len(AUTOGLUON_INPUT_COLUMNS),
        "feature_names": list(AUTOGLUON_INPUT_COLUMNS),
        "training_seconds": round(elapsed, 3),
        "best_model": str(predictor.model_best),
        "metrics": {
            "auroc": report.auroc,
            "sensitivity": report.sensitivity,
            "specificity": report.specificity,
            "ppv": report.ppv,
            "npv": report.npv,
            "accuracy": report.accuracy,
            "f1": report.f1,
            "threshold": report.threshold,
        },
        "paper_targets_table_2_automl": PAPER_TARGETS_TABLE_2_AUTOML,
        "library_versions": _library_versions(),
        "paths": {
            "train_parquet": str(train_path),
            "test_parquet": str(test_path),
            "model_dir": str(model_dir),
            "predictions_parquet": str(pred_path),
            "leaderboard_csv": str(leaderboard_path),
        },
    }
    manifest_path = settings.data_processed_dir / "autogluon_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Wrote training manifest -> %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
