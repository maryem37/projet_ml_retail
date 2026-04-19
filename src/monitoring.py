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


def _print_drift_summary(roc_ref: float, roc_cur: float, n_features: int):
    degradation = (roc_ref - roc_cur) * 100
    status      = "⚠️  DEGRADED" if degradation > 5 else "✅ STABLE"

    print("\n" + "=" * 60)
    print("  MONITORING SUMMARY")
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
    reference    : pd.DataFrame,
    current      : pd.DataFrame,
    trained_model,
    feature_names,
) -> None:
    """
    Simplified monitoring without Evidently.
    Uses KS test to detect feature drift manually.
    """
    import scipy.stats as stats
    from sklearn.metrics import roc_auc_score

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

    # ✅ FIX: guard against empty drifted list
    #         pd.DataFrame([]) has no columns → .sort_values("ks_stat") crashes
    if drifted:
        drift_df = pd.DataFrame(drifted).sort_values("ks_stat", ascending=False)
    else:
        drift_df = pd.DataFrame(
            columns=["feature", "ks_stat", "p_value", "ref_mean", "cur_mean"]
        )

    drift_df.to_csv("reports/drift_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("  SIMPLIFIED MONITORING RESULTS")
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

    # Model performance — drop target/prediction before scoring
    ref_features = reference.drop(columns=["target", "prediction"], errors="ignore")
    cur_features = current.drop(columns=["target", "prediction"],   errors="ignore")

    ref_probs = trained_model.predict_proba(ref_features)[:, 1]
    cur_probs = trained_model.predict_proba(cur_features)[:, 1]

    ref_auc     = roc_auc_score(reference["target"], ref_probs)
    cur_auc     = roc_auc_score(current["target"],   cur_probs)
    degradation = (ref_auc - cur_auc) * 100
    status      = "⚠️  DEGRADED" if degradation > 5 else "✅ STABLE"

    print(f"\n  Model Performance:")
    print(f"    Reference ROC-AUC : {ref_auc:.4f}")
    print(f"    Current   ROC-AUC : {cur_auc:.4f}")
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

    X_train = pd.read_csv("data/train_test/X_train.csv")
    X_test  = pd.read_csv("data/train_test/X_test.csv")
    y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/train_test/y_test.csv").squeeze()

    pipeline       = joblib.load("models/churn_model.pkl")
    threshold_data = joblib.load("models/threshold.pkl")

    # Extract model step — SMOTE not needed at inference
    trained_model = pipeline.estimator.named_steps["model"]   # ← fixed
    THRESHOLD     = threshold_data["threshold"]

    logger.info(
        "Reference (train): %d rows | Current (test): %d rows",
        len(X_train), len(X_test)
    )
    logger.info("Using threshold: %.3f", THRESHOLD)

    # ── 2. Build reference & current datasets ─────────────
    reference = X_train.copy()
    current   = X_test.copy()

    reference["target"] = y_train.values
    current["target"]   = y_test.values

    # Apply tuned threshold — not default 0.5
    ref_probs = trained_model.predict_proba(X_train)[:, 1]
    cur_probs = trained_model.predict_proba(X_test)[:, 1]

    reference["prediction"] = (ref_probs >= THRESHOLD).astype(int)
    current["prediction"]   = (cur_probs >= THRESHOLD).astype(int)

    from sklearn.metrics import roc_auc_score
    roc_ref = roc_auc_score(y_train, ref_probs)
    roc_cur = roc_auc_score(y_test,  cur_probs)

    # ── 3. Evidently or fallback ───────────────────────────
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

        important_features = _get_top_features(trained_model, X_train.columns, n=15)
        drift_cols         = important_features + ["target", "prediction"]
        drift_col_map      = _get_column_mapping(reference[drift_cols])

        drift_report = Report(metrics=[
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

        _print_drift_summary(roc_ref, roc_cur, len(important_features))

    else:
        _simple_monitoring(reference, current, trained_model, X_train.columns)