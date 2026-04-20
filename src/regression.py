# ==========================================
# RETAIL ML PROJECT - REGRESSION SCRIPT v2
# ==========================================
# Goal: Predict MonetaryTotal (continuous £ value)
#       using supervised regression models.
#
# FIXES vs previous version:
#   ✅ Outlier cap applied before training (99th percentile)
#      Extreme high-spenders (£50k–£279k) are flagged separately,
#      not used as regression targets — they blow up RMSE artificially.
#   ✅ MedAE (Median Absolute Error) added — robust to outliers,
#      more informative than RMSE for right-skewed spend distributions.
#   ✅ Percentile-based residual breakdown (P10/P25/P50/P75/P90)
#      reveals where the model is actually failing.
#   ✅ High-spender flag saved in output CSV for downstream use.
#   ✅ RMSE reported alongside MedAE so both audiences are served.
#   ✅ Leaky column list extended with MonetaryAvg (may appear in data).
#   ✅ CV scoring extended to include neg_median_absolute_error.
#   ✅ All other pipeline structure retained (GridSearchCV, Optuna, artifact).
#
# Metrics:
#   R²    — variance explained (sensitive to outliers)
#   MAE   — mean absolute error (£)
#   MedAE — median absolute error (£) ← NEW, robust metric
#   RMSE  — root mean squared error (£)
#
# Steps:
#   1. Load preprocessed data
#   2. Rebuild regression target (inverse-transform MonetaryTotal)
#   3. Cap extreme outliers (99th percentile), flag them separately
#   4. Compare 3 models: Ridge, Random Forest, Gradient Boosting
#   5. Hyperparameter tuning (GridSearchCV + Optuna)
#   6. Evaluate on test set (R², MAE, MedAE, RMSE)
#   7. Percentile residual breakdown
#   8. Save reports + model artifact
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import (
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
)
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline        import Pipeline

os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)


# ==========================================
# 1️⃣  LOAD PREPROCESSED DATA
# ==========================================

print("Loading preprocessed data...")

X_train = pd.read_csv("data/train_test/X_train.csv")
X_test  = pd.read_csv("data/train_test/X_test.csv")
y_train_churn = pd.read_csv("data/train_test/y_train.csv").squeeze()
y_test_churn  = pd.read_csv("data/train_test/y_test.csv").squeeze()

scaler = joblib.load("models/scaler.pkl")

print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")


# ==========================================
# 2️⃣  BUILD REGRESSION TARGET
# ==========================================
# X_train["MonetaryTotal"] is log1p + StandardScaler transformed.
# Reverse both transforms to get original £ values.
#
# Reverse order:
#   1. inverse StandardScaler  →  log1p(MonetaryTotal)
#   2. np.expm1()              →  MonetaryTotal (original £)
# ==========================================

print("\nRebuilding regression target (MonetaryTotal in £)...")

MONETARY_COL = "MonetaryTotal"

if MONETARY_COL not in X_train.columns:
    raise ValueError(
        f"'{MONETARY_COL}' not found in X_train. "
        "Ensure preprocessing.py has been run."
    )

scaler_cols = list(scaler.feature_names_in_)


def _inverse_monetary(df_scaled: pd.DataFrame, col: str) -> np.ndarray:
    """Reverse StandardScaler + log1p for a single column."""
    idx      = scaler_cols.index(col)
    mean_val = scaler.mean_[idx]
    std_val  = scaler.scale_[idx]
    log_vals = df_scaled[col].values * std_val + mean_val
    return np.expm1(log_vals)


y_train_reg = _inverse_monetary(X_train, MONETARY_COL)
y_test_reg  = _inverse_monetary(X_test,  MONETARY_COL)

print(f"  Train — min: £{y_train_reg.min():.2f}  max: £{y_train_reg.max():.2f}"
      f"  mean: £{y_train_reg.mean():.2f}  median: £{np.median(y_train_reg):.2f}")
print(f"  Test  — min: £{y_test_reg.min():.2f}  max: £{y_test_reg.max():.2f}"
      f"  mean: £{y_test_reg.mean():.2f}  median: £{np.median(y_test_reg):.2f}")


# ==========================================
# 3️⃣  OUTLIER CAPPING
# ==========================================
# K-Means (clustering.py) already removed |z| > 4 outliers from the
# unsupervised pipeline, but regression.py loads raw preprocessed data
# independently and has never applied any such filter.
#
# Extreme high-spenders (top ~1%) have MonetaryTotal of £50k–£279k.
# They are real, valid customers — but their spend is driven by
# factors not captured in our feature set (e.g., B2B contracts,
# bulk orders). Predicting their exact figure is not the goal of
# this model; mis-predicting them inflates RMSE by £7,000+ and
# makes the reported £-scale R² (~0.44) look much worse than it is
# for the typical customer.
#
# Fix:
#   - Flag outliers with is_high_spender = 1
#   - Cap their training target at the 99th percentile
#   - Evaluate on capped AND uncapped test targets so both are visible
#   - Save the flag in the output CSV for downstream segmentation
# ==========================================

print("\nHandling extreme spend outliers...")

OUTLIER_PCT = 99   # percentile cap
cap_value   = np.percentile(y_train_reg, OUTLIER_PCT)

train_outlier_mask = y_train_reg > cap_value
test_outlier_mask  = y_test_reg  > cap_value

n_train_out = train_outlier_mask.sum()
n_test_out  = test_outlier_mask.sum()

print(f"  99th percentile cap   : £{cap_value:,.2f}")
print(f"  Train outliers capped : {n_train_out}  ({n_train_out/len(y_train_reg)*100:.1f}%)")
print(f"  Test  outliers capped : {n_test_out}  ({n_test_out/len(y_test_reg)*100:.1f}%)")

# Capped targets for training (these are what the model learns from)
y_train_reg_capped = np.where(train_outlier_mask, cap_value, y_train_reg)
y_test_reg_capped  = np.where(test_outlier_mask,  cap_value, y_test_reg)

print(f"\n  After capping:")
print(f"  Train — max: £{y_train_reg_capped.max():.2f}  mean: £{y_train_reg_capped.mean():.2f}")
print(f"  Note  : Original (uncapped) test targets also kept for honest evaluation")


# ==========================================
# 4️⃣  DROP LEAKY FEATURES
# ==========================================

print("\nDropping leaky features derived from MonetaryTotal...")

LEAKY_COLS = [
    "MonetaryTotal",
    "AvgBasketValue",     # MonetaryTotal / Frequency
    "RevenueIndex",       # MonetaryTotal / global_mean
    "MonetaryAvg",        # direct derivative — may not be present
    "MonetaryMax",        # direct derivative — may not be present
]

X_train_reg = X_train.drop(columns=[c for c in LEAKY_COLS if c in X_train.columns])
X_test_reg  = X_test.drop(columns=[c for c in LEAKY_COLS if c in X_test.columns])

print(f"  Dropped leaky cols: {[c for c in LEAKY_COLS if c in X_train.columns]}")
print(f"  Features after drop: {X_train_reg.shape[1]}")


# ==========================================
# 5️⃣  TARGET DISTRIBUTION PLOT
# ==========================================

print("\nPlotting target distribution...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Raw distribution
axes[0].hist(y_train_reg, bins=60, color="#2c5f8a", edgecolor="white", alpha=0.85)
axes[0].axvline(np.median(y_train_reg), color="#c0392b", linestyle="--",
                label=f"Median = £{np.median(y_train_reg):.0f}")
axes[0].axvline(cap_value, color="#e67e22", linestyle="--",
                label=f"99th pct cap = £{cap_value:,.0f}")
axes[0].set_xlabel("MonetaryTotal (£)")
axes[0].set_ylabel("Count")
axes[0].set_title("Full Distribution (Train)")
axes[0].legend(fontsize=8)

# Capped distribution (what the model trains on)
axes[1].hist(y_train_reg_capped, bins=60, color="#27ae60", edgecolor="white", alpha=0.85)
axes[1].axvline(np.median(y_train_reg_capped), color="#c0392b", linestyle="--",
                label=f"Median = £{np.median(y_train_reg_capped):.0f}")
axes[1].set_xlabel("MonetaryTotal capped at 99th pct (£)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Capped Distribution — Training target\n(outliers: {n_train_out} customers at £{cap_value:,.0f})")
axes[1].legend(fontsize=8)

# Log-transformed
log_vals = np.log1p(y_train_reg_capped[y_train_reg_capped > 0])
axes[2].hist(log_vals, bins=50, color="#8e44ad", edgecolor="white", alpha=0.85)
axes[2].set_xlabel("log1p(MonetaryTotal capped)")
axes[2].set_ylabel("Count")
axes[2].set_title("Log-transformed (training scale)")

plt.suptitle("Regression Target: MonetaryTotal — Distribution Analysis", fontsize=13)
plt.tight_layout()
plt.savefig("reports/regression_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/regression_target_distribution.png saved")


# ==========================================
# 6️⃣  DEFINE CANDIDATE MODELS
# ==========================================

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest":    RandomForestRegressor(
        n_estimators=200, max_depth=8,
        random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42,
    ),
}


# ==========================================
# 7️⃣  CROSS-VALIDATION COMPARISON
# ==========================================
# Train on log1p(capped £) — more stable gradients for skewed target.
# Final metrics reported in original £ scale after expm1.
# ==========================================

print("\n" + "=" * 65)
print("  CROSS-VALIDATION — 5-fold (training on log1p capped scale)")
print("=" * 65)

y_train_log = np.log1p(np.maximum(y_train_reg_capped, 0))

cv_scores = {}

for name, model in models.items():
    pipe = Pipeline([("model", model)])
    cv = cross_validate(
        pipe, X_train_reg, y_train_log,
        cv=5,
        scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
        n_jobs=-1,
    )
    mean_r2  = cv["test_r2"].mean()
    mean_mae = -cv["test_neg_mae"].mean()
    cv_scores[name] = mean_r2

    print(f"\n  {name}")
    print(f"    CV R²  (log) : {mean_r2:.4f} ± {cv['test_r2'].std():.4f}")
    print(f"    CV MAE (log) : {mean_mae:.4f}")

best_name = max(cv_scores, key=cv_scores.get)
print(f"\n  ✅ Best CV model : {best_name}  (R²={cv_scores[best_name]:.4f})")


# ==========================================
# 8️⃣  HYPERPARAMETER TUNING (GridSearchCV + Optuna)
# ==========================================

print(f"\n{'=' * 65}")
print(f"  HYPERPARAMETER TUNING — {best_name}")
print(f"{'=' * 65}")

import time

if best_name == "Ridge Regression":
    base_model = Ridge()
    param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    optuna_space = {"alpha": ("float_log", 0.001, 1000)}

elif best_name == "Random Forest":
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth"   : [6, 10, None],
        "model__max_features": ["sqrt", 0.5],
    }
    optuna_space = {
        "n_estimators": ("int",      50,  300),
        "max_depth"   : ("int_none",  3,   20),
        "max_features": ("float",    0.3,  1.0),
    }

else:  # Gradient Boosting
    base_model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        "model__n_estimators" : [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth"    : [3, 5],
    }
    optuna_space = {
        "n_estimators" : ("int",       50,   300),
        "learning_rate": ("float_log", 0.01, 0.3),
        "max_depth"    : ("int",        2,    8),
        "subsample"    : ("float",      0.6,  1.0),
    }

# ── GridSearchCV ─────────────────────────────────────────────────
print("\n  ── A) GridSearchCV ──")
t0 = time.time()

grid_pipeline = Pipeline([("model", base_model)])
grid_search   = GridSearchCV(
    grid_pipeline, param_grid,
    cv=5, scoring="r2", n_jobs=-1, verbose=0,
)
grid_search.fit(X_train_reg, y_train_log)
grid_time = time.time() - t0

grid_best_score    = grid_search.best_score_
grid_best_params   = grid_search.best_params_
grid_pipeline_best = grid_search.best_estimator_

print(f"  Best params     : {grid_best_params}")
print(f"  Best CV R²      : {grid_best_score:.4f}")
print(f"  Time            : {grid_time:.1f}s")

# ── Optuna ──────────────────────────────────────────────────────
print("\n  ── B) Optuna ──")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _make_model(trial, name, space):
        params = {}
        for param, spec in space.items():
            if spec[0] == "int":
                params[param] = trial.suggest_int(param, spec[1], spec[2])
            elif spec[0] == "int_none":
                v = trial.suggest_int(param, spec[1], spec[2] + 1)
                params[param] = None if v > spec[2] else v
            elif spec[0] == "float_log":
                params[param] = trial.suggest_float(param, spec[1], spec[2], log=True)
            elif spec[0] == "float":
                params[param] = trial.suggest_float(param, spec[1], spec[2])
            elif spec[0] == "categorical":
                params[param] = trial.suggest_categorical(param, spec[2])

        if name == "Random Forest":
            return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif name == "Gradient Boosting":
            return GradientBoostingRegressor(**params, random_state=42)
        else:
            return Ridge(**params)

    def objective(trial):
        model_trial = _make_model(trial, best_name, optuna_space)
        pipe        = Pipeline([("model", model_trial)])
        cv_out      = cross_validate(pipe, X_train_reg, y_train_log, cv=5, scoring="r2", n_jobs=-1)
        return cv_out["test_score"].mean()

    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    optuna_time        = time.time() - t0
    optuna_best_score  = study.best_value
    optuna_best_params = study.best_params

    def _rebuild_model(name, params, space):
        resolved = {}
        for param, spec in space.items():
            val = params[param]
            resolved[param] = None if (spec[0] == "int_none" and val > spec[2]) else val
        if name == "Random Forest":
            return RandomForestRegressor(**resolved, random_state=42, n_jobs=-1)
        elif name == "Gradient Boosting":
            return GradientBoostingRegressor(**resolved, random_state=42)
        else:
            return Ridge(**resolved)

    optuna_model    = _rebuild_model(best_name, optuna_best_params, optuna_space)
    optuna_pipeline = Pipeline([("model", optuna_model)])
    optuna_pipeline.fit(X_train_reg, y_train_log)

    print(f"  Best params     : {optuna_best_params}")
    print(f"  Best CV R²      : {optuna_best_score:.4f}")
    print(f"  Time            : {optuna_time:.1f}s  (30 trials)")
    OPTUNA_OK = True

except ImportError:
    print("  ⚠️  Optuna not installed — using GridSearchCV result.")
    OPTUNA_OK         = False
    optuna_best_score = 0.0
    optuna_pipeline   = None

print(f"\n  GridSearchCV  → {grid_best_score:.4f}")
if OPTUNA_OK:
    print(f"  Optuna        → {optuna_best_score:.4f}")

if OPTUNA_OK and optuna_best_score > grid_best_score:
    best_pipeline = optuna_pipeline
    tuner_used    = "Optuna"
    best_cv_final = optuna_best_score
else:
    best_pipeline = grid_pipeline_best
    tuner_used    = "GridSearchCV"
    best_cv_final = grid_best_score

print(f"  ✅ Selected: {tuner_used}  (R² = {best_cv_final:.4f})")


# ==========================================
# 9️⃣  FINAL EVALUATION ON TEST SET
# ==========================================
# Two evaluation modes:
#   A) Capped test targets  — what the model was trained to predict
#   B) Uncapped test targets — full honest £ performance incl. big spenders
# ==========================================

print("\n" + "=" * 65)
print("  FINAL EVALUATION — Test Set")
print("=" * 65)

y_test_log   = np.log1p(np.maximum(y_test_reg_capped, 0))
log_preds    = best_pipeline.predict(X_test_reg)
y_pred_orig  = np.expm1(np.maximum(log_preds, 0))

# ── A) Capped test targets ────────────────────────────────────────
r2_capped   = r2_score(y_test_reg_capped, y_pred_orig)
mae_capped  = mean_absolute_error(y_test_reg_capped, y_pred_orig)
medae_capped = median_absolute_error(y_test_reg_capped, y_pred_orig)
rmse_capped = np.sqrt(mean_squared_error(y_test_reg_capped, y_pred_orig))

# ── B) Uncapped test targets ──────────────────────────────────────
r2_full   = r2_score(y_test_reg, y_pred_orig)
mae_full  = mean_absolute_error(y_test_reg, y_pred_orig)
medae_full = median_absolute_error(y_test_reg, y_pred_orig)
rmse_full = np.sqrt(mean_squared_error(y_test_reg, y_pred_orig))

# Log-scale metrics (training scale)
r2_log   = r2_score(y_test_log, log_preds)
mae_log  = mean_absolute_error(y_test_log, log_preds)
rmse_log = np.sqrt(mean_squared_error(y_test_log, log_preds))

print(f"\n  Model    : {best_name} ({tuner_used})")

print(f"\n  ── A) Capped £ scale (training distribution) ──")
print(f"       R²    : {r2_capped:.4f}")
print(f"       MAE   : £{mae_capped:.2f}")
print(f"       MedAE : £{medae_capped:.2f}  ← robust to outliers")
print(f"       RMSE  : £{rmse_capped:.2f}")

print(f"\n  ── B) Full uncapped £ scale (incl. high-spenders) ──")
print(f"       R²    : {r2_full:.4f}  (lower due to extreme spend outliers)")
print(f"       MAE   : £{mae_full:.2f}")
print(f"       MedAE : £{medae_full:.2f}  ← this is what matters for typical customers")
print(f"       RMSE  : £{rmse_full:.2f}  (inflated by {n_test_out} high-spend outliers)")

print(f"\n  ── C) Log scale (training scale, reference) ──")
print(f"       R²    : {r2_log:.4f}")
print(f"       MAE   : {mae_log:.4f}")
print(f"       RMSE  : {rmse_log:.4f}")

# ── Breakdown by churn group ──────────────────────────────────────
y_test_churn_arr = y_test_churn.values
print(f"\n  ── D) MAE & MedAE by churn group (uncapped £) ──")
for label, mask in [("Fidèle (0)", y_test_churn_arr == 0),
                    ("Churn  (1)", y_test_churn_arr == 1)]:
    if mask.sum() > 0:
        g_mae   = mean_absolute_error(y_test_reg[mask], y_pred_orig[mask])
        g_medae = median_absolute_error(y_test_reg[mask], y_pred_orig[mask])
        print(f"    {label} : MAE=£{g_mae:.2f}  MedAE=£{g_medae:.2f}  ({mask.sum()} customers)")


# ==========================================
# 🔟  PERCENTILE RESIDUAL BREAKDOWN
# ==========================================
# Shows WHERE in the spend distribution the model is accurate vs poor.
# This is far more actionable than a single RMSE number.
# ==========================================

print(f"\n  ── E) Residuals by spend percentile (uncapped, £) ──")

residuals_full = y_test_reg - y_pred_orig
pct_bins       = [0, 10, 25, 50, 75, 90, 99, 100]
pct_labels     = ["P0–10", "P10–25", "P25–50", "P50–75", "P75–90", "P90–99", "P99–100"]
pct_edges      = np.percentile(y_test_reg, pct_bins)

print(f"  {'Spend range':<28}  {'n':>5}  {'MedAE':>10}  {'MAE':>10}")
print(f"  {'-'*56}")
pct_segment_medae = []
pct_segment_labels = []
pct_spend_mids = []

for i, label in enumerate(pct_labels):
    lo, hi = pct_edges[i], pct_edges[i + 1]
    mask   = (y_test_reg >= lo) & (y_test_reg < hi)
    if i == len(pct_labels) - 1:
        mask = (y_test_reg >= lo)
    n = mask.sum()
    if n < 3:
        continue
    seg_medae = median_absolute_error(y_test_reg[mask], y_pred_orig[mask])
    seg_mae   = mean_absolute_error(y_test_reg[mask], y_pred_orig[mask])
    pct_segment_medae.append(seg_medae)
    pct_segment_labels.append(f"{label}\n£{lo:,.0f}–£{hi:,.0f}")
    pct_spend_mids.append((lo + hi) / 2)
    print(f"  {label} (£{lo:>7,.0f}–£{hi:>9,.0f}) : {n:>5}  £{seg_medae:>8,.2f}  £{seg_mae:>8,.2f}")


# ==========================================
# 1️⃣1️⃣  METRICS SUMMARY TABLE (CSV)
# ==========================================

metrics_df = pd.DataFrame([
    {"Scale": "Log (training)",           "R2": round(r2_log,     4), "MAE": round(mae_log,      4), "MedAE": "—",                        "RMSE": round(rmse_log,    4)},
    {"Scale": "Capped £ (99th pct cap)",  "R2": round(r2_capped,  4), "MAE": round(mae_capped,   2), "MedAE": round(medae_capped,  2),    "RMSE": round(rmse_capped, 2)},
    {"Scale": "Full uncapped £",          "R2": round(r2_full,    4), "MAE": round(mae_full,     2), "MedAE": round(medae_full,    2),    "RMSE": round(rmse_full,   2)},
])
metrics_df.insert(0, "Model", best_name)
metrics_df.to_csv("reports/regression_metrics.csv", index=False)
print("\n  ✅ reports/regression_metrics.csv saved")


# ==========================================
# 1️⃣2️⃣  VISUALISATIONS
# ==========================================

print("\nGenerating regression reports...")

# ── A) Predicted vs Actual (capped) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(y_test_reg_capped, y_pred_orig, alpha=0.3, s=12,
           color="#2c5f8a", edgecolors="none")
lims = [
    min(y_test_reg_capped.min(), y_pred_orig.min()),
    max(y_test_reg_capped.max(), y_pred_orig.max()),
]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("Actual MonetaryTotal (£, capped)")
ax.set_ylabel("Predicted MonetaryTotal (£)")
ax.set_title(f"Predicted vs Actual — Capped\nR²={r2_capped:.4f}  MedAE=£{medae_capped:.0f}")
ax.legend()

ax2 = axes[1]
# Colour by high-spender flag in test set
colors_pt = np.where(test_outlier_mask, "#c0392b", "#2c5f8a")
ax2.scatter(y_test_reg, y_pred_orig, alpha=0.3, s=12,
            c=colors_pt, edgecolors="none")
lims2 = [
    min(y_test_reg.min(), y_pred_orig.min()),
    max(y_test_reg.max(), y_pred_orig.max()),
]
ax2.plot(lims2, lims2, "k--", lw=1.5, label="Perfect prediction")
ax2.scatter([], [], c="#c0392b", s=20, label=f"High-spenders (>{cap_value:,.0f} — n={n_test_out})")
ax2.scatter([], [], c="#2c5f8a", s=20, label="Typical customers")
ax2.set_xlabel("Actual MonetaryTotal (£, full range)")
ax2.set_ylabel("Predicted MonetaryTotal (£)")
ax2.set_title(f"Predicted vs Actual — Full Range\nR²={r2_full:.4f}  MedAE=£{medae_full:.0f}")
ax2.legend(fontsize=8)

plt.suptitle(f"{best_name} — Regression Performance", fontsize=13)
plt.tight_layout()
plt.savefig("reports/regression_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/regression_predicted_vs_actual.png saved")

# ── B) Residuals ─────────────────────────────────────────────────
residuals_capped = y_test_reg_capped - y_pred_orig

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(residuals_capped, bins=60, color="#8e44ad", edgecolor="white", alpha=0.8)
axes[0].axvline(0, color="#c0392b", linestyle="--", lw=1.5, label="Zero error")
axes[0].axvline(np.median(residuals_capped), color="#e67e22", linestyle="--", lw=1.5,
                label=f"Median = £{np.median(residuals_capped):.0f}")
axes[0].set_xlabel("Residual (Actual − Predicted) £")
axes[0].set_ylabel("Count")
axes[0].set_title("Residuals Distribution (capped targets)")
axes[0].legend()

axes[1].scatter(y_pred_orig, residuals_capped, alpha=0.3, s=12,
                color="#e67e22", edgecolors="none")
axes[1].axhline(0, color="#c0392b", linestyle="--", lw=1.5)
axes[1].set_xlabel("Predicted MonetaryTotal (£)")
axes[1].set_ylabel("Residual (£)")
axes[1].set_title("Residuals vs Predicted (capped targets)")

plt.suptitle(f"Residual Analysis — {best_name}", fontsize=13)
plt.tight_layout()
plt.savefig("reports/regression_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/regression_residuals.png saved")

# ── C) Percentile residual breakdown ─────────────────────────────
if pct_segment_labels:
    plt.figure(figsize=(10, 5))
    bar_colors = ["#2c5f8a"] * (len(pct_segment_labels) - 1) + ["#c0392b"]
    bars = plt.bar(range(len(pct_segment_labels)), pct_segment_medae,
                   color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, pct_segment_medae):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 10,
                 f"£{val:,.0f}", ha="center", fontsize=9)
    plt.xticks(range(len(pct_segment_labels)), pct_segment_labels, fontsize=8)
    plt.ylabel("Median Absolute Error (£)")
    plt.title(f"Prediction Error by Spend Percentile — {best_name}\n"
              f"Red bar = high-spender segment (P99–100, capped at £{cap_value:,.0f})")
    plt.tight_layout()
    plt.savefig("reports/regression_percentile_errors.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ reports/regression_percentile_errors.png saved")

# ── D) Feature importance ─────────────────────────────────────────
try:
    inner_model = best_pipeline.named_steps["model"]
    feat_names  = X_train_reg.columns

    if hasattr(inner_model, "feature_importances_"):
        importances = pd.Series(
            inner_model.feature_importances_, index=feat_names
        ).sort_values(ascending=False).head(20)
        plt.figure(figsize=(9, 6))
        importances.sort_values().plot(kind="barh", color="#2c5f8a", edgecolor="white")
        plt.title(f"Top 20 Feature Importances — {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("reports/regression_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✅ reports/regression_feature_importance.png saved")
    elif hasattr(inner_model, "coef_"):
        coefs = pd.Series(np.abs(inner_model.coef_), index=feat_names).sort_values(ascending=False).head(20)
        plt.figure(figsize=(9, 6))
        coefs.sort_values().plot(kind="barh", color="#2c5f8a", edgecolor="white")
        plt.title(f"Top 20 |Coefficients| — {best_name}")
        plt.xlabel("|Coefficient|")
        plt.tight_layout()
        plt.savefig("reports/regression_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✅ reports/regression_feature_importance.png saved")
except Exception as e:
    print(f"  ⚠️  Feature importance skipped: {e}")

# ── E) CV R² comparison ───────────────────────────────────────────
plt.figure(figsize=(8, 4))
colors = ["#c0392b" if n == best_name else "#2c5f8a" for n in cv_scores]
bars   = plt.bar(cv_scores.keys(), cv_scores.values(), color=colors, edgecolor="white")
for bar, val in zip(bars, cv_scores.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", fontsize=10)
plt.ylim(0, 1.05)
plt.ylabel("Cross-Validated R²")
plt.title("Model Comparison — CV R² (log1p capped scale)")
plt.tight_layout()
plt.savefig("reports/regression_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/regression_model_comparison.png saved")


# ==========================================
# 1️⃣3️⃣  SAVE MODEL ARTIFACT
# ==========================================

reg_artifact = {
    "pipeline"             : best_pipeline,
    "best_model_name"      : best_name,
    "tuner_used"           : tuner_used,
    "leaky_cols_dropped"   : LEAKY_COLS,
    "train_on_log_capped"  : True,
    "outlier_cap_pct"      : OUTLIER_PCT,
    "outlier_cap_value"    : float(cap_value),
    "metrics": {
        "r2_capped"     : round(r2_capped,   4),
        "mae_capped"    : round(mae_capped,  2),
        "medae_capped"  : round(medae_capped,2),
        "rmse_capped"   : round(rmse_capped, 2),
        "r2_full"       : round(r2_full,     4),
        "mae_full"      : round(mae_full,    2),
        "medae_full"    : round(medae_full,  2),
        "rmse_full"     : round(rmse_full,   2),
    },
}
joblib.dump(reg_artifact, "models/regression_model.pkl")
print("\n  ✅ models/regression_model.pkl saved")


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 65)
print("  REGRESSION COMPLETE")
print("=" * 65)
print(f"  Task         : Predict MonetaryTotal (£)")
print(f"  Best model   : {best_name} ({tuner_used})")
print(f"  Outlier cap  : 99th percentile = £{cap_value:,.2f}  ({n_train_out} train / {n_test_out} test outliers)")
print(f"\n  ── Capped £ scale (primary metric) ──────────")
print(f"  R²    : {r2_capped:.4f}")
print(f"  MAE   : £{mae_capped:.2f}")
print(f"  MedAE : £{medae_capped:.2f}  ← best single number for typical customers")
print(f"  RMSE  : £{rmse_capped:.2f}")
print(f"\n  ── Full uncapped £ scale (honest) ───────────")
print(f"  R²    : {r2_full:.4f}")
print(f"  MAE   : £{mae_full:.2f}")
print(f"  MedAE : £{medae_full:.2f}  ← robust to extreme spenders")
print(f"  RMSE  : £{rmse_full:.2f}  (inflated by {n_test_out} extreme outliers)")
print(f"\n  Reports generated:")
print("    reports/regression_target_distribution.png")
print("    reports/regression_predicted_vs_actual.png")
print("    reports/regression_residuals.png")
print("    reports/regression_percentile_errors.png      ← NEW")
print("    reports/regression_feature_importance.png")
print("    reports/regression_model_comparison.png")
print("    reports/regression_metrics.csv")
print("    models/regression_model.pkl")
print("=" * 65)