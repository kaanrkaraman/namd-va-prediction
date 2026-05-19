from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import logging
import sys

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from namd_replication.config import settings
from namd_replication.data.encoder import FeatureEncoder
from namd_replication.interpretability import (
    PAPER_FEATURES,
    compute_autogluon_permutation_importance,
    compute_pdp_categorical,
    compute_pdp_continuous,
    compute_xgb_shap_grouped_importance,
    make_autogluon_raw_predictor,
    make_xgb_raw_predictor,
    normalize_to_rfi,
    plot_feature_importance_comparison,
    plot_pdp_categorical,
    plot_pdp_continuous,
    plot_shap_summary_xgboost,
)
from namd_replication.models.autogluon_model import select_autogluon_columns
from namd_replication.models.xgboost_model import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("interpret_models")


PAPER_XGBOOST_RFI: dict[str, float] = {
    "baseline_va": 0.556,
    "agegroup": 0.117,
    "ethnicity": 0.077,
    "vol_irf": 0.069,
    "vol_ped": 0.058,
}

PAPER_AUTOML_RFI: dict[str, float] = {
    "baseline_va": 0.498,
    "agegroup": 0.112,
    "ethnicity": 0.103,
    "vol_ped": 0.076,
    "vol_irf": 0.054,
}


def main() -> int:
    test_path = settings.data_processed_dir / "test.parquet"
    if not test_path.is_file():
        log.error("Missing %s. Run scripts/02_build_cohort.py first.", test_path)
        return 1
    test_df = pd.read_parquet(test_path)
    log.info("Loaded test set: n=%d", len(test_df))

    xgb_path = settings.outputs_dir / "models" / "xgboost.json"
    encoder_path = settings.outputs_dir / "models" / "encoder.joblib"
    for p in (xgb_path, encoder_path):
        if not p.is_file():
            log.error("Missing %s. Run scripts/03_train_xgboost.py first.", p)
            return 2
    xgb_model = load_model(xgb_path)
    encoder = FeatureEncoder.load(encoder_path)
    x_test = encoder.transform(test_df)

    ag_dir = settings.outputs_dir / "models" / "autogluon"
    if not ag_dir.is_dir():
        log.error("Missing %s. Run scripts/04_train_autogluon.py first.", ag_dir)
        return 3
    ag_predictor = TabularPredictor.load(str(ag_dir))

    figures_dir = settings.outputs_dir / "figures"
    tables_dir = settings.outputs_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    log.info("Computing XGBoost SHAP feature importance ...")
    xgb_raw_imp = compute_xgb_shap_grouped_importance(xgb_model, x_test)
    xgb_rfi = normalize_to_rfi(xgb_raw_imp)
    log.info("XGBoost RFI: %s", {k: round(v, 4) for k, v in xgb_rfi.items()})

    log.info("Computing AutoGluon permutation feature importance ...")
    np.random.seed(settings.random_seed)
    test_df_ag_with_label = select_autogluon_columns(test_df, include_label=True)
    ag_raw_imp = compute_autogluon_permutation_importance(
        ag_predictor, test_df_ag_with_label
    )
    ag_rfi = normalize_to_rfi(ag_raw_imp)
    log.info("AutoGluon RFI: %s", {k: round(v, 4) for k, v in ag_rfi.items()})

    fi_path = figures_dir / "feature_importance.png"
    plot_feature_importance_comparison(xgb_rfi, ag_rfi, fi_path)
    log.info("Wrote feature importance plot -> %s", fi_path)

    fi_csv = tables_dir / "feature_importance.csv"
    pd.DataFrame(
        {
            "feature": list(PAPER_FEATURES),
            "xgboost_raw_shap": [xgb_raw_imp[f] for f in PAPER_FEATURES],
            "xgboost_rfi": [xgb_rfi[f] for f in PAPER_FEATURES],
            "autogluon_raw_perm": [ag_raw_imp[f] for f in PAPER_FEATURES],
            "autogluon_rfi": [ag_rfi[f] for f in PAPER_FEATURES],
            "paper_xgboost_rfi": [
                PAPER_XGBOOST_RFI.get(f, float("nan")) for f in PAPER_FEATURES
            ],
            "paper_automl_rfi": [
                PAPER_AUTOML_RFI.get(f, float("nan")) for f in PAPER_FEATURES
            ],
        }
    ).to_csv(fi_csv, index=False)
    log.info("Wrote feature importance CSV -> %s", fi_csv)

    log.info("Plotting SHAP beeswarm for XGBoost ...")
    shap_path = figures_dir / "shap_summary_xgboost.png"
    plot_shap_summary_xgboost(xgb_model, x_test, shap_path)
    log.info("Wrote SHAP beeswarm -> %s", shap_path)

    xgb_predict = make_xgb_raw_predictor(xgb_model, encoder)
    ag_predict = make_autogluon_raw_predictor(ag_predictor)

    log.info("Computing PDPs for baseline_va ...")
    va_grid = np.linspace(
        float(test_df["baseline_va"].min()),
        float(test_df["baseline_va"].max()),
        30,
    )
    xgb_pdp_va = compute_pdp_continuous(xgb_predict, test_df, "baseline_va", va_grid)
    ag_pdp_va = compute_pdp_continuous(ag_predict, test_df, "baseline_va", va_grid)
    pdp_va_path = figures_dir / "pdp_baseline_va.png"
    plot_pdp_continuous(
        va_grid,
        xgb_pdp_va,
        ag_pdp_va,
        "Baseline VA (ETDRS letters)",
        pdp_va_path,
    )
    log.info("Wrote baseline_va PDP -> %s", pdp_va_path)

    log.info("Computing PDPs for agegroup ...")
    age_levels = [
        "50-59 years",
        "60-69 years",
        "70-79 years",
        "80 years and above",
    ]
    xgb_pdp_age = compute_pdp_categorical(xgb_predict, test_df, "agegroup", age_levels)
    ag_pdp_age = compute_pdp_categorical(ag_predict, test_df, "agegroup", age_levels)
    pdp_age_path = figures_dir / "pdp_agegroup.png"
    plot_pdp_categorical(age_levels, xgb_pdp_age, ag_pdp_age, "Age group", pdp_age_path)
    log.info("Wrote agegroup PDP -> %s", pdp_age_path)

    log.info("Computing PDPs for vol_irf ...")
    irf_p95 = float(np.quantile(test_df["vol_irf"], 0.95))
    irf_grid = np.linspace(0.0, max(irf_p95, 0.6), 30)
    xgb_pdp_irf = compute_pdp_continuous(xgb_predict, test_df, "vol_irf", irf_grid)
    ag_pdp_irf = compute_pdp_continuous(ag_predict, test_df, "vol_irf", irf_grid)
    pdp_irf_path = figures_dir / "pdp_vol_irf.png"
    plot_pdp_continuous(
        irf_grid, xgb_pdp_irf, ag_pdp_irf, "IRF volume (mm³)", pdp_irf_path
    )
    log.info("Wrote vol_irf PDP -> %s", pdp_irf_path)

    log.info("Computing PDPs for vol_ped ...")
    ped_p95 = float(np.quantile(test_df["vol_ped"], 0.95))
    ped_grid = np.linspace(0.0, max(ped_p95, 3.0), 30)
    xgb_pdp_ped = compute_pdp_continuous(xgb_predict, test_df, "vol_ped", ped_grid)
    ag_pdp_ped = compute_pdp_continuous(ag_predict, test_df, "vol_ped", ped_grid)
    pdp_ped_path = figures_dir / "pdp_vol_ped.png"
    plot_pdp_continuous(
        ped_grid, xgb_pdp_ped, ag_pdp_ped, "PED volume (mm³)", pdp_ped_path
    )
    log.info("Wrote vol_ped PDP -> %s", pdp_ped_path)

    manifest = {
        "n_test": len(test_df),
        "seed": settings.random_seed,
        "feature_importance": {
            "xgboost_rfi": xgb_rfi,
            "autogluon_rfi": ag_rfi,
            "paper_xgboost_rfi_partial": PAPER_XGBOOST_RFI,
            "paper_automl_rfi_partial": PAPER_AUTOML_RFI,
        },
        "pdp_grids": {
            "baseline_va": va_grid.tolist(),
            "vol_irf": irf_grid.tolist(),
            "vol_ped": ped_grid.tolist(),
            "agegroup": age_levels,
        },
        "pdp_values": {
            "baseline_va_xgb": xgb_pdp_va.tolist(),
            "baseline_va_ag": ag_pdp_va.tolist(),
            "agegroup_xgb": xgb_pdp_age,
            "agegroup_ag": ag_pdp_age,
            "vol_irf_xgb": xgb_pdp_irf.tolist(),
            "vol_irf_ag": ag_pdp_irf.tolist(),
            "vol_ped_xgb": xgb_pdp_ped.tolist(),
            "vol_ped_ag": ag_pdp_ped.tolist(),
        },
        "figures": {
            "feature_importance": str(fi_path),
            "shap_summary_xgboost": str(shap_path),
            "pdp_baseline_va": str(pdp_va_path),
            "pdp_agegroup": str(pdp_age_path),
            "pdp_vol_irf": str(pdp_irf_path),
            "pdp_vol_ped": str(pdp_ped_path),
        },
    }
    manifest_path = settings.data_processed_dir / "interpretation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Wrote interpretation manifest -> %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
