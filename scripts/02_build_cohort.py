from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict

from namd_replication.config import settings
from namd_replication.data.encoder import FeatureEncoder
from namd_replication.data.loader import build_analysis_cohort
from namd_replication.data.splitter import persist_split, stratified_split
from namd_replication.viz.figures import plot_consort
from namd_replication.viz.tables import generate_table1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("build_cohort")


def main() -> int:
    raw_csv = settings.data_raw_dir / "dataframe.csv"
    if not raw_csv.is_file():
        log.error("Missing %s. Run scripts/01_download_data.py first.", raw_csv)
        return 1

    log.info("Building analysis cohort from %s", raw_csv)
    cohort, steps = build_analysis_cohort(raw_csv)
    log.info(
        "Final cohort: n_eyes=%d  n_patients=%d  (Above=%d, Below=%d)",
        len(cohort),
        int(cohort["anonid"].nunique()),
        int((cohort["outcome"] == "Above").sum()),
        int((cohort["outcome"] == "Below").sum()),
    )

    processed_dir = settings.data_processed_dir
    figures_dir = settings.outputs_dir / "figures"
    tables_dir = settings.outputs_dir / "tables"
    models_dir = settings.outputs_dir / "models"
    for d in (processed_dir, figures_dir, tables_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    cohort_path = processed_dir / "cohort.parquet"
    cohort.to_parquet(cohort_path)
    log.info("Wrote cohort -> %s (%d rows)", cohort_path, len(cohort))

    table1_path = tables_dir / "table_01_cohort.csv"
    table1 = generate_table1(cohort, outcome_col="outcome", out_path=table1_path)
    log.info("Wrote Table 1 -> %s (%d rows)", table1_path, len(table1))

    consort_path = figures_dir / "exclusion_flow.png"
    plot_consort(steps, consort_path)
    log.info("Wrote CONSORT diagram -> %s", consort_path)

    train, test = stratified_split(
        cohort,
        outcome_col="outcome",
        train_fraction=settings.train_fraction,
        seed=settings.random_seed,
    )
    persist_split(train, test, processed_dir)
    log.info(
        "Stratified split: n_train=%d  n_test=%d  (seed=%d)",
        len(train),
        len(test),
        settings.random_seed,
    )

    encoder = FeatureEncoder().fit(cohort)
    encoder_path = models_dir / "encoder.joblib"
    encoder.save(encoder_path)
    log.info("Wrote encoder -> %s", encoder_path)

    manifest = {
        "seed": settings.random_seed,
        "train_fraction": settings.train_fraction,
        "va_threshold_etdrs": settings.va_threshold_etdrs,
        "followup_window_days": [
            settings.followup_window_min_days,
            settings.followup_window_max_days,
        ],
        "n_eyes_final": len(cohort),
        "n_patients_final": int(cohort["anonid"].nunique()),
        "n_train": len(train),
        "n_test": len(test),
        "outcome_counts": {
            "Above": int((cohort["outcome"] == "Above").sum()),
            "Below": int((cohort["outcome"] == "Below").sum()),
        },
        "exclusion_log": [asdict(step) for step in steps],
        "paths": {
            "raw_csv": str(raw_csv),
            "cohort_parquet": str(cohort_path),
            "train_parquet": str(processed_dir / "train.parquet"),
            "test_parquet": str(processed_dir / "test.parquet"),
            "split_indices_json": str(processed_dir / "split_indices.json"),
            "encoder_joblib": str(encoder_path),
            "table1_csv": str(table1_path),
            "consort_png": str(consort_path),
        },
    }
    manifest_path = processed_dir / "cohort_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Wrote cohort manifest -> %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
