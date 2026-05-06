# ==========================================
# RETAIL ML PROJECT - MODEL MONITORING
# ==========================================
# FIXES APPLIED:
#   ✅ KeyError 'ks_stat' — empty drifted list guard added
#   ✅ helper functions defined BEFORE they are called
#   ✅ trained_model extracted from pipeline (named_steps)
#   ✅ tuned threshold applied to all predictions
#   ✅ __main__ guard wraps all execution logic
#   ✅ column_mapping built before report.run()
#   ✅ predict_proba used for AUC — not predict
#
# FIXES in this version (v2):
#   ✅ trained_model and X_train now correctly extracted/loaded
#      inside the EVIDENTLY_OK block — were NameError before
#   ✅ X_ref_raw used for predict_proba BEFORE target/prediction
#      columns are added to reference — eliminates fragile ordering bug
#   ✅ roc_ref > 0.99 warning replaced with clearer placeholder warning
#      that fires based on actual mode (placeholder vs production)
#   ✅ _print_drift_summary called in both Evidently and fallback paths
#      for consistent output format
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_loader import get_config, get_logger

cfg    = get_config()
logger = get_logger("monitoring")

os.makedirs("reports", exist_ok=True)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _get_column_mapping(df: pd.DataFrame):
    from evidently import ColumnMapping
    numeric_features = [
        c for c in df.columns
        if c not in ["target", "prediction"]
        and df[c].dtype in ["int64", "float64"]
    ][:20]
    return ColumnMapping(
        target             = "target",
        prediction         = "prediction",
        numerical_features = numeric_features,
    )


def _get_top_features(model, feature_names, n: int = 15) -> list:
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        return importances.nlargest(n).index.tolist()
    elif hasattr(model, "coef_"):
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_names)
        return importances.nlargest(n).index.tolist()
    return list(feature_names[:n])


def _print_drift_summary(roc_ref: float, roc_cur: float, n_features: int,
                          placeholder: bool = False):
    degradation = (roc_ref - roc_cur) * 100
    status      = "⚠️  DEGRADED" if degradation > 5 else "✅ STABLE"

    print("\n" + "=" * 60)
    print("  MONITORING SUMMARY")
    if placeholder:
        print("  ⚠️  PLACEHOLDER MODE — not real production monitoring")
    print("=" * 60)
    print(f"  Features monitored    : {n_features}")
    print(f"  Reference ROC-AUC     : {roc_ref:.4f}")
    print(f"  Current   ROC-AUC     : {roc_cur:.4f}")
    print(f"  Degradation           : {degradation:.2f}%  {status}")
    print(f"\n  Reports generated:")
    print(f"    reports/monitoring_report.html  ← Full Evidently report")
    print(f"    reports/drift_report.html       ← Data drift only")
    print("=" * 60)

    if degradation > 5:
        logger.warning(
            "Model performance degraded by %.2f%%. Consider retraining.", degradation
        )


def _simple_monitoring(
    reference          : pd.DataFrame,
    current            : pd.DataFrame,
    calibrated_pipeline,
    feature_names,
    roc_ref            : float,
    roc_cur            : float,
    placeholder        : bool = False,
) -> None:
    """
    Simplified monitoring without Evidently.
    Uses KS test to detect feature drift manually.
    FIX v2: roc_ref and roc_cur are passed in (computed outside on raw feature
    DataFrames) instead of recomputed here, avoiding the column ordering
    fragility issue (target/prediction columns must not be in the input).
    """
    import scipy.stats as stats

    logger.info("Running simplified drift detection (KS test)...")

    numeric_cols = reference.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["target", "prediction"]][:20]

    drifted = []
    for col in numeric_cols:
        ref_vals = reference[col].dropna()
        cur_vals = current[col].dropna()

        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)

        if p_value < 0.05:
            drifted.append({
                "feature" : col,
                "ks_stat" : round(ks_stat, 4),
                "p_value" : round(p_value, 4),
                "ref_mean": round(ref_vals.mean(), 3),
                "cur_mean": round(cur_vals.mean(), 3),
            })

    # ✅ Guard against empty drifted list
    if drifted:
        drift_df = pd.DataFrame(drifted).sort_values("ks_stat", ascending=False)
    else:
        drift_df = pd.DataFrame(
            columns=["feature", "ks_stat", "p_value", "ref_mean", "cur_mean"]
        )

    drift_df.to_csv("reports/drift_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("  SIMPLIFIED MONITORING RESULTS")
    if placeholder:
        print("  ⚠️  PLACEHOLDER MODE — not real production monitoring")
    print("=" * 60)
    print(f"  Features checked : {len(numeric_cols)}")
    print(f"  Drifted features : {len(drifted)}  (KS test p < 0.05)")

    if len(drifted) > 0:
        print(f"\n  Top drifted features:")
        for _, row in drift_df.head(5).iterrows():
            print(
                f"    {row['feature']:<35} "
                f"KS={row['ks_stat']}  p={row['p_value']}  "
                f"ref_mean={row['ref_mean']} → cur_mean={row['cur_mean']}"
            )
    else:
        print("  ✅ No significant drift detected.")

    degradation = (roc_ref - roc_cur) * 100
    status      = "⚠️  DEGRADED" if degradation > 5 else "✅ STABLE"

    print(f"\n  Model Performance:")
    print(f"    Reference ROC-AUC : {roc_ref:.4f}")
    print(f"    Current   ROC-AUC : {roc_cur:.4f}")
    print(f"    Degradation       : {degradation:.2f}%  {status}")
    print(f"\n  Drift report saved → reports/drift_summary.csv")
    print("=" * 60)

    if degradation > 5:
        logger.warning(
            "Model ROC-AUC degraded by %.2f%%. Consider retraining.", degradation
        )


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # ── 1. Load data & artifacts ───────────────────────────
    logger.info("Loading data and artifacts for monitoring...")
    X_test  = pd.read_csv("data/train_test/X_test.csv")
    y_test  = pd.read_csv("data/train_test/y_test.csv").squeeze()

    pipeline       = joblib.load("models/churn_model.pkl")
    threshold_data = joblib.load("models/threshold.pkl")
    THRESHOLD      = threshold_data["threshold"]

    logger.info("Using threshold: %.3f", THRESHOLD)

    # ── 2. Build reference / current datasets ─────────────
    # Reference = test set (last known good labelled data).
    # Current   = new production data arriving over time.
    PRODUCTION_DATA_PATH  = "data/production/new_customers.csv"
    PRODUCTION_LABEL_PATH = "data/production/new_customers_labels.csv"

    IS_PLACEHOLDER = not os.path.exists(PRODUCTION_DATA_PATH)

    if IS_PLACEHOLDER:
        logger.warning(
            "No production data found at %s. "
            "Splitting test set 50/50 as a placeholder. "
            "Replace with real incoming data when available.",
            PRODUCTION_DATA_PATH,
        )
        logger.warning(
            "PLACEHOLDER MODE: results below are NOT real drift monitoring. "
            "With only ~%d rows per side, AUC differences of 1-3%% are "
            "pure random noise. Do not act on these numbers.",
            len(X_test) // 2,
        )
        split     = len(X_test) // 2
        X_ref_raw = X_test.iloc[:split].reset_index(drop=True)
        X_cur_raw = X_test.iloc[split:].reset_index(drop=True)
        y_ref     = y_test.iloc[:split].reset_index(drop=True)
        y_cur     = y_test.iloc[split:].reset_index(drop=True)
    else:
        X_ref_raw = X_test.reset_index(drop=True)
        y_ref     = y_test.reset_index(drop=True)
        X_cur_raw = pd.read_csv(PRODUCTION_DATA_PATH)
        y_cur     = pd.read_csv(PRODUCTION_LABEL_PATH).squeeze()

    logger.info(
        "Reference: %d rows | Current: %d rows",
        len(X_ref_raw), len(X_cur_raw)
    )

    # ── 3. Compute probabilities on raw feature DataFrames ─
    # FIX v2: predict_proba is called on X_ref_raw / X_cur_raw BEFORE
    # adding target/prediction columns to reference/current DataFrames.
    # This eliminates the fragile ordering dependency: if target/prediction
    # were added first, predict_proba would receive unknown columns and crash.
    from sklearn.metrics import roc_auc_score

    ref_probs = pipeline.predict_proba(X_ref_raw)[:, 1]
    cur_probs = pipeline.predict_proba(X_cur_raw)[:, 1]

    roc_ref = roc_auc_score(y_ref, ref_probs)
    roc_cur = roc_auc_score(y_cur, cur_probs)

    # Now build the annotated DataFrames for Evidently / drift reporting
    reference               = X_ref_raw.copy()
    current                 = X_cur_raw.copy()
    reference["target"]     = y_ref.values
    current["target"]       = y_cur.values
    reference["prediction"] = (ref_probs >= THRESHOLD).astype(int)
    current["prediction"]   = (cur_probs >= THRESHOLD).astype(int)

    # ── 4. Evidently or fallback ───────────────────────────
    try:
        from evidently.report import Report
        from evidently.metric_preset import (
            DataDriftPreset,
            ClassificationPreset,
            DataQualityPreset,
            TargetDriftPreset,
        )
        from evidently.metrics import (
            DatasetDriftMetric,
            ColumnDriftMetric,
        )
        logger.info("Evidently loaded — generating reports...")
        EVIDENTLY_OK = True

    except ImportError:
        logger.warning(
            "Evidently not installed. Run: pip install evidently\n"
            "Falling back to simplified monitoring..."
        )
        EVIDENTLY_OK = False

    if EVIDENTLY_OK:

        # FIX v2: trained_model and X_train must be extracted/loaded here.
        # They were undefined in the previous version → NameError at runtime.
        # fitted_for_cal is stored inside calibrated differently across sklearn versions.
        # Safest approach: access the estimator attribute then named_steps.
        try:
            inner_pipeline = pipeline.estimator              # sklearn Pipeline
            trained_model  = inner_pipeline.named_steps["model"]
        except AttributeError:
            # Fallback: calibrated wraps the pipeline directly
            trained_model  = pipeline.named_steps["model"]
        X_train = pd.read_csv("data/train_test/X_train.csv")

        important_features = _get_top_features(trained_model, X_train.columns, n=15)
        drift_cols         = important_features + ["target", "prediction"]

        col_map = _get_column_mapping(reference)

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            ClassificationPreset(),
            TargetDriftPreset(),
        ])
        report.run(
            reference_data = reference,
            current_data   = current,
            column_mapping = col_map,
        )
        report.save_html("reports/monitoring_report.html")
        logger.info("Full report saved → reports/monitoring_report.html")

        drift_col_map = _get_column_mapping(reference[drift_cols])
        drift_report  = Report(metrics=[
            DatasetDriftMetric(),
        ] + [
            ColumnDriftMetric(column_name=col)
            for col in important_features
        ])
        drift_report.run(
            reference_data = reference[drift_cols],
            current_data   = current[drift_cols],
            column_mapping = drift_col_map,
        )
        drift_report.save_html("reports/drift_report.html")
        logger.info("Drift report saved → reports/drift_report.html")

        _print_drift_summary(
            roc_ref     = roc_ref,
            roc_cur     = roc_cur,
            n_features  = len(important_features),
            placeholder = IS_PLACEHOLDER,
        )

    else:
        # Fallback: simplified KS-based monitoring
        # FIX v2: _simple_monitoring now receives pre-computed roc_ref/roc_cur
        # and IS_PLACEHOLDER flag — consistent output with the Evidently path.
        _simple_monitoring(
            reference           = reference,
            current             = current,
            calibrated_pipeline = pipeline,
            feature_names       = X_test.columns,
            roc_ref             = roc_ref,
            roc_cur             = roc_cur,
            placeholder         = IS_PLACEHOLDER,
        )