from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict

import pandas as pd

from namd_replication.config import settings
from namd_replication.evaluation.delong import delong_roc_test
from namd_replication.evaluation.metrics import PerformanceReport, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("compare_models")


PAPER_DELONG_P: float = 0.71


def _report_row(model: str, report: PerformanceReport) -> dict[str, object]:
    return {"model": model, **asdict(report)}


def main() -> int:
    pred_dir = settings.outputs_dir / "predictions"
    xgb_path = pred_dir / "test_predictions.parquet"
    ag_path = pred_dir / "autogluon_test_predictions.parquet"

    for p in (xgb_path, ag_path):
        if not p.is_file():
            log.error(
                "Missing %s. Run scripts/03_train_xgboost.py and "
                "scripts/04_train_autogluon.py first.",
                p,
            )
            return 1

    xgb = pd.read_parquet(xgb_path).sort_values("eye_key").reset_index(drop=True)
    ag = pd.read_parquet(ag_path).sort_values("eye_key").reset_index(drop=True)

    if not xgb["eye_key"].equals(ag["eye_key"]):
        log.error("eye_key mismatch between XGBoost and AutoGluon prediction files")
        return 2
    if not xgb["y_true"].equals(ag["y_true"]):
        log.error("y_true mismatch between XGBoost and AutoGluon prediction files")
        return 3

    y_true = xgb["y_true"].to_numpy().astype(int)
    xgb_score = xgb["y_score"].to_numpy().astype(float)
    ag_score = ag["y_score"].to_numpy().astype(float)

    log.info(
        "Loaded n=%d aligned predictions (Above=%d  Below=%d)",
        len(y_true),
        int(y_true.sum()),
        int((y_true == 0).sum()),
    )

    xgb_report = compute_metrics(y_true, xgb_score)
    ag_report = compute_metrics(y_true, ag_score)

    log.info(
        "XGBoost: AUROC=%.4f  Sens=%.3f  Spec=%.3f  PPV=%.3f  NPV=%.3f  Acc=%.3f  F1=%.3f",
        xgb_report.auroc,
        xgb_report.sensitivity,
        xgb_report.specificity,
        xgb_report.ppv,
        xgb_report.npv,
        xgb_report.accuracy,
        xgb_report.f1,
    )
    log.info(
        "AutoGluon: AUROC=%.4f  Sens=%.3f  Spec=%.3f  PPV=%.3f  NPV=%.3f  Acc=%.3f  F1=%.3f",
        ag_report.auroc,
        ag_report.sensitivity,
        ag_report.specificity,
        ag_report.ppv,
        ag_report.npv,
        ag_report.accuracy,
        ag_report.f1,
    )

    z, p = delong_roc_test(y_true, ag_score, xgb_score)
    significant = p < 0.05
    log.info(
        "DeLong (AutoGluon vs XGBoost): z=%+.4f  p=%.4f  %s",
        z,
        p,
        "significantly different" if significant else "not significantly different",
    )
    log.info(
        "Paper Table 2 / Fig 3: DeLong p=%.2f  not significantly different",
        PAPER_DELONG_P,
    )

    comparison_rows = [
        _report_row("xgboost", xgb_report),
        _report_row("autogluon", ag_report),
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    tables_dir = settings.outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = tables_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    log.info("Wrote per-model metrics -> %s", comparison_path)

    summary = {
        "n_test": len(y_true),
        "n_above": int(y_true.sum()),
        "n_below": int((y_true == 0).sum()),
        "xgboost": {
            "auroc": xgb_report.auroc,
            "sensitivity": xgb_report.sensitivity,
            "specificity": xgb_report.specificity,
            "ppv": xgb_report.ppv,
            "npv": xgb_report.npv,
            "accuracy": xgb_report.accuracy,
            "f1": xgb_report.f1,
        },
        "autogluon": {
            "auroc": ag_report.auroc,
            "sensitivity": ag_report.sensitivity,
            "specificity": ag_report.specificity,
            "ppv": ag_report.ppv,
            "npv": ag_report.npv,
            "accuracy": ag_report.accuracy,
            "f1": ag_report.f1,
        },
        "delong_autogluon_vs_xgboost": {
            "z": float(z),
            "p_two_sided": float(p),
            "significant_at_0.05": bool(significant),
        },
        "auroc_diff": float(ag_report.auroc - xgb_report.auroc),
        "paper_reference": {
            "delong_p_automl_vs_xgboost": PAPER_DELONG_P,
            "conclusion": "not significantly different",
        },
    }
    summary_path = tables_dir / "model_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("Wrote comparison summary -> %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
