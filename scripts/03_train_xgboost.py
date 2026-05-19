from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from importlib.metadata import version

import pandas as pd

from namd_replication.config import settings
from namd_replication.data.encoder import FeatureEncoder, encode_outcome
from namd_replication.evaluation.metrics import compute_metrics
from namd_replication.models.xgboost_model import (
    SUPP_TABLE_4_HPARAMS,
    save_model,
    train_xgboost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("train_xgboost")


def _library_versions() -> dict[str, str]:
    libs = ["xgboost", "numpy", "pandas", "scikit-learn", "scipy", "shap"]
    return {"python": sys.version.split()[0]} | {lib: version(lib) for lib in libs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["fast", "grid"],
        default="fast",
        help="fast = fixed hparams; grid = 6,318-combination CV search",
    )
    args = parser.parse_args()

    train_path = settings.data_processed_dir / "train.parquet"
    test_path = settings.data_processed_dir / "test.parquet"
    encoder_path = settings.outputs_dir / "models" / "encoder.joblib"
    for p in (train_path, test_path, encoder_path):
        if not p.is_file():
            log.error("Missing %s. Run scripts/02_build_cohort.py first.", p)
            return 1

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    encoder = FeatureEncoder.load(encoder_path)

    x_train = encoder.transform(train)
    x_test = encoder.transform(test)
    y_train = encode_outcome(train)
    y_test = encode_outcome(test)

    log.info(
        "Loaded train=%s test=%s n_features=%d",
        x_train.shape,
        x_test.shape,
        x_train.shape[1],
    )
    log.info(
        "Outcome distribution: train Above=%d Below=%d  test Above=%d Below=%d",
        int(y_train.sum()),
        int((y_train == 0).sum()),
        int(y_test.sum()),
        int((y_test == 0).sum()),
    )

    t0 = time.perf_counter()
    if args.mode == "fast":
        log.info("Mode: fast (fixed hparams)")
        model = train_xgboost(x_train, y_train, seed=settings.random_seed)
    else:
        from namd_replication.models.xgboost_model import grid_search_xgboost

        log.info("Mode: grid (6,318 combinations x 10-fold CV)")
        model, _ = grid_search_xgboost(x_train, y_train, seed=settings.random_seed)
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.2f seconds", elapsed)

    y_score = model.predict_proba(x_test)[:, 1]
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

    model_path = settings.outputs_dir / "models" / "xgboost.json"
    save_model(model, model_path)
    log.info("Wrote model -> %s", model_path)

    pred_dir = settings.outputs_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "test_predictions.parquet"
    pd.DataFrame(
        {
            "eye_key": test["eye_key"].to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_score": y_score,
        }
    ).to_parquet(pred_path)
    log.info("Wrote test predictions -> %s", pred_path)

    manifest = {
        "mode": args.mode,
        "seed": settings.random_seed,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": int(x_train.shape[1]),
        "training_seconds": round(elapsed, 3),
        "hparams": SUPP_TABLE_4_HPARAMS,
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
        "library_versions": _library_versions(),
        "paths": {
            "train_parquet": str(train_path),
            "test_parquet": str(test_path),
            "encoder_joblib": str(encoder_path),
            "model_json": str(model_path),
            "predictions_parquet": str(pred_path),
        },
    }
    manifest_path = settings.data_processed_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Wrote training manifest -> %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
