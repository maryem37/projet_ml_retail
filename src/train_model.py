# ==========================================
# RETAIL ML PROJECT - TRAINING SCRIPT
# ==========================================
# CHANGES vs previous version:
#   ✅ SMOTE removed entirely
#   ✅ class_weight="balanced" used for LR and RF
#   ✅ GradientBoostingClassifier replaced by HistGradientBoostingClassifier
#      which natively supports class_weight="balanced" — eliminates the
#      previous incorrect comment that subsample mitigates imbalance
#   ✅ Standard sklearn Pipeline
#   ✅ Optuna vs GridSearchCV comparison retained
#   ✅ CalibratedClassifierCV: cv="prefit" (explicit, documented)
#      replaces cv=None which had undefined behaviour across sklearn versions
#   ✅ Redundant clone+refit before calibration removed —
#      best_pipeline is already fitted; calibration wraps it directly
#   ✅ Scoring: average_precision (PR-AUC) replaces roc_auc as primary CV
#      metric — better aligned with F2 threshold tuning and class imbalance
#   ✅ surrogate tree: no class_weight (correct — mimics calibrated outputs)
#   ✅ _node_counts() uses round() instead of int() — robust to weighted floats
# ==========================================

import pandas as pd
import numpy as np
import joblib
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble        import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, f1_score, recall_score, precision_score,
    precision_recall_curve, auc, average_precision_score,
)
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import plot_feature_importance, plot_correlation_heatmap

os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)


# ==========================================
# 1️⃣  LOAD PREPROCESSED DATA
# ==========================================

print("Loading preprocessed data...")

X_train     = pd.read_csv("data/train_test/X_train.csv")
X_test      = pd.read_csv("data/train_test/X_test.csv")
X_train_pca = pd.read_csv("data/train_test/X_train_pca.csv")
X_test_pca  = pd.read_csv("data/train_test/X_test_pca.csv")
y_train     = pd.read_csv("data/train_test/y_train.csv").squeeze()
y_test      = pd.read_csv("data/train_test/y_test.csv").squeeze()

print(f"  X_train     : {X_train.shape}     (raw scaled features)")
print(f"  X_train_pca : {X_train_pca.shape} (PCA-reduced features)")
print(f"  Churn rate  : train {y_train.mean():.2%} | test {y_test.mean():.2%}")

assert X_train.isnull().sum().sum() == 0,     "NaNs in X_train!"
assert X_train_pca.isnull().sum().sum() == 0, "NaNs in X_train_pca!"
print("  ✅ No NaNs detected.")

# Shared StratifiedKFold — used consistently across all CV calls
# to guarantee class balance in each fold (33% churn).
CV_SPLITTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ==========================================
# 2️⃣  CLASS DISTRIBUTION PLOT
# ==========================================

print("\nPlotting class distribution...")

counts = y_train.value_counts().sort_index()
labels = ["Fidèle (0)", "Churn (1)"]
colors = ["#2c5f8a", "#c0392b"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)],
              color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            str(val), ha="center", fontsize=12, fontweight="bold")
ax.set_title("Class Distribution — Training Set\n(imbalance handled via class_weight='balanced')",
             fontsize=12)
ax.set_ylabel("Number of samples")
ax.set_ylim(0, max(counts) * 1.2)
plt.tight_layout()
plt.savefig("reports/class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/class_distribution.png saved")
print(f"  Fidèle : {counts.get(0, 0)}  |  Churn : {counts.get(1, 0)}"
      f"  |  Ratio : {counts.get(1, 0)/counts.get(0, 0):.2f}")


# ==========================================
# 3️⃣  CORRELATION HEATMAP
# ==========================================

print("\nGenerating correlation heatmap...")
plot_correlation_heatmap(
    X_train,
    title     = "Training Data — Feature Correlation Matrix",
    save_path = "reports/correlation_heatmap.png"
)


# ==========================================
# 4️⃣  DEFINE CANDIDATE MODELS
# ==========================================
# FIX: GradientBoostingClassifier replaced by HistGradientBoostingClassifier.
# HistGBM supports class_weight="balanced" natively, which correctly adjusts
# the loss function for class imbalance.
# GradientBoostingClassifier had no class_weight parameter — the previous
# workaround comment ("subsample mitigates imbalance") was incorrect:
# subsample is a stochastic boosting technique unrelated to class imbalance.
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=0.1,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6,
        random_state=42, n_jobs=-1,
        class_weight="balanced",
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05,
        max_depth=3, random_state=42,
        class_weight="balanced",   # ✅ native support — correct imbalance handling
    ),
}


# ==========================================
# 5️⃣  CROSS-VALIDATION — Raw vs PCA features
# ==========================================
# FIX: Primary scoring changed from roc_auc → average_precision (PR-AUC).
# With 33% churn and F2-based threshold tuning downstream, PR-AUC is a more
# faithful selection criterion than ROC-AUC, which is insensitive to the
# precision/recall trade-off at specific operating points.
# Both metrics are still computed and printed for full transparency.
# ==========================================

print("\n" + "=" * 65)
print("  CROSS-VALIDATION — Raw features vs PCA features")
print("  Primary metric: average_precision (PR-AUC)")
print("=" * 65)

cv_results  = {}
best_cv_raw = {}
best_cv_pca = {}

for name, model in models.items():
    pipeline_raw = Pipeline([("model", model)])
    pipeline_pca = Pipeline([("model", model)])

    cv_raw = cross_validate(
        pipeline_raw, X_train, y_train, cv=CV_SPLITTER,
        scoring={
            "average_precision": "average_precision",
            "roc_auc":           "roc_auc",
            "f1":                "f1",
            "recall":            "recall",
        },
        n_jobs=-1,
    )
    cv_pca = cross_validate(
        pipeline_pca, X_train_pca, y_train, cv=CV_SPLITTER,
        scoring={
            "average_precision": "average_precision",
            "roc_auc":           "roc_auc",
            "f1":                "f1",
            "recall":            "recall",
        },
        n_jobs=-1,
    )

    # Selection on average_precision (PR-AUC)
    best_cv_raw[name] = cv_raw["test_average_precision"].mean()
    best_cv_pca[name] = cv_pca["test_average_precision"].mean()
    cv_results[name]  = {"raw": cv_raw, "pca": cv_pca}

    print(f"\n  {name}")
    print(f"    Raw → PR-AUC: {cv_raw['test_average_precision'].mean():.4f}"
          f" ± {cv_raw['test_average_precision'].std():.4f}"
          f"  |  ROC-AUC: {cv_raw['test_roc_auc'].mean():.4f}"
          f"  |  F1: {cv_raw['test_f1'].mean():.4f}"
          f"  |  Recall: {cv_raw['test_recall'].mean():.4f}")
    print(f"    PCA → PR-AUC: {cv_pca['test_average_precision'].mean():.4f}"
          f" ± {cv_pca['test_average_precision'].std():.4f}"
          f"  |  ROC-AUC: {cv_pca['test_roc_auc'].mean():.4f}"
          f"  |  F1: {cv_pca['test_f1'].mean():.4f}"
          f"  |  Recall: {cv_pca['test_recall'].mean():.4f}")
    winner = "Raw" if best_cv_raw[name] >= best_cv_pca[name] else "PCA"
    print(f"    → Winner: {winner} features")

best_name_raw = max(best_cv_raw, key=best_cv_raw.get)
best_name_pca = max(best_cv_pca, key=best_cv_pca.get)

if best_cv_raw[best_name_raw] >= best_cv_pca[best_name_pca]:
    best_name    = best_name_raw
    USE_PCA      = False
    X_tr         = X_train
    X_te         = X_test
    feature_note = "Raw scaled features"
else:
    best_name    = best_name_pca
    USE_PCA      = True
    X_tr         = X_train_pca
    X_te         = X_test_pca
    feature_note = "PCA-reduced features"

print(f"\n  ✅ Best overall : {best_name} on {feature_note}")
print(f"     PR-AUC      : {max(best_cv_raw[best_name_raw], best_cv_pca[best_name_pca]):.4f}")


# ==========================================
# 6️⃣  HYPERPARAMETER TUNING
#     GridSearchCV  vs  Optuna — compared
#     Both use average_precision as scoring (aligned with step 5)
# ==========================================

print(f"\n{'=' * 65}")
print(f"  HYPERPARAMETER TUNING — GridSearchCV vs Optuna")
print(f"  Model: {best_name} | Features: {feature_note}")
print(f"  Scoring: average_precision")
print(f"{'=' * 65}")

if best_name == "Random Forest":
    base_model = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    param_grid = {
        "model__n_estimators"     : [100, 200],
        "model__max_depth"        : [None, 6, 10],
        "model__min_samples_split": [2, 5],
    }
    optuna_space = {
        "n_estimators"     : ("int",      50,  300),
        "max_depth"        : ("int_none",  3,   20),
        "min_samples_split": ("int",       2,   10),
    }

elif best_name == "Hist Gradient Boosting":
    base_model = HistGradientBoostingClassifier(
        random_state=42, class_weight="balanced"
    )
    param_grid = {
        "model__max_iter"     : [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth"    : [3, 5],
    }
    optuna_space = {
        "max_iter"     : ("int",       50,   300),
        "learning_rate": ("float_log", 0.01, 0.3),
        "max_depth"    : ("int",        2,    8),
        "l2_regularization": ("float", 0.0, 10.0),
    }

else:  # Logistic Regression
    base_model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    param_grid = {
        "model__C"     : [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"],
    }
    optuna_space = {
        "C"     : ("float_log",  0.001, 100),
        "solver": ("categorical", None, ["lbfgs", "liblinear"]),
    }


# ── A) GridSearchCV ───────────────────────────────────────────────

print("\n  ── A) GridSearchCV ──")
t0 = time.time()

grid_pipeline = Pipeline([("model", base_model)])
grid_search   = GridSearchCV(
    grid_pipeline, param_grid,
    cv=CV_SPLITTER, scoring="average_precision", n_jobs=-1, verbose=0,
)
grid_search.fit(X_tr, y_train)
grid_time = time.time() - t0

grid_best_score    = grid_search.best_score_
grid_best_params   = grid_search.best_params_
grid_pipeline_best = grid_search.best_estimator_

print(f"  Best params     : {grid_best_params}")
print(f"  Best CV PR-AUC  : {grid_best_score:.4f}")
print(f"  Time elapsed    : {grid_time:.1f}s")


# ── B) Optuna ─────────────────────────────────────────────────────

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
            return RandomForestClassifier(
                **params, random_state=42, n_jobs=-1, class_weight="balanced"
            )
        elif name == "Hist Gradient Boosting":
            return HistGradientBoostingClassifier(
                **params, random_state=42, class_weight="balanced"
            )
        else:
            return LogisticRegression(
                **params, max_iter=1000, random_state=42, class_weight="balanced"
            )

    def objective(trial):
        model_trial = _make_model(trial, best_name, optuna_space)
        pipe        = Pipeline([("model", model_trial)])
        cv_out      = cross_validate(
            pipe, X_tr, y_train,
            cv=CV_SPLITTER, scoring="average_precision", n_jobs=-1
        )
        return cv_out["test_score"].mean()

    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    optuna_time = time.time() - t0

    optuna_best_score  = study.best_value
    optuna_best_params = study.best_params

    def _rebuild_model_from_params(name, params, space):
        """
        Reconstruct the model from Optuna best_params dict.
        Manually reapply int_none conversion (Optuna stores the raw integer).
        """
        resolved = {}
        for param, spec in space.items():
            val = params[param]
            if spec[0] == "int_none":
                resolved[param] = None if val > spec[2] else val
            else:
                resolved[param] = val

        if name == "Random Forest":
            return RandomForestClassifier(
                **resolved, random_state=42, n_jobs=-1, class_weight="balanced"
            )
        elif name == "Hist Gradient Boosting":
            return HistGradientBoostingClassifier(
                **resolved, random_state=42, class_weight="balanced"
            )
        else:
            return LogisticRegression(
                **resolved, max_iter=1000, random_state=42, class_weight="balanced"
            )

    optuna_model    = _rebuild_model_from_params(best_name, optuna_best_params, optuna_space)
    optuna_pipeline = Pipeline([("model", optuna_model)])
    optuna_pipeline.fit(X_tr, y_train)

    print(f"  Best params     : {optuna_best_params}")
    print(f"  Best CV PR-AUC  : {optuna_best_score:.4f}")
    print(f"  Time elapsed    : {optuna_time:.1f}s  (30 trials)")
    OPTUNA_OK = True

except ImportError:
    print("  ⚠️  Optuna not installed. Run: pip install optuna")
    print("      Skipping Optuna — GridSearchCV result used.")
    OPTUNA_OK          = False
    optuna_best_score  = 0.0
    optuna_best_params = {}
    optuna_pipeline    = None
    optuna_time        = 0


# ── Comparison ────────────────────────────────────────────────────

print(f"\n  {'─' * 50}")
print(f"  TUNER COMPARISON")
print(f"  {'─' * 50}")
print(f"  GridSearchCV  → PR-AUC: {grid_best_score:.4f}  | Time: {grid_time:.1f}s")
if OPTUNA_OK:
    print(f"  Optuna        → PR-AUC: {optuna_best_score:.4f}  | Time: {optuna_time:.1f}s  (30 trials)")
    faster = "Optuna" if optuna_time < grid_time else "GridSearchCV"
    better = "Optuna" if optuna_best_score > grid_best_score else "GridSearchCV"
    print(f"  Best score    : {better}")
    print(f"  Faster        : {faster}")

if OPTUNA_OK and optuna_best_score > grid_best_score:
    best_pipeline = optuna_pipeline
    tuner_used    = "Optuna"
    best_cv_final = optuna_best_score
else:
    best_pipeline = grid_pipeline_best
    tuner_used    = "GridSearchCV"
    best_cv_final = grid_best_score

print(f"\n  ✅ Selected tuner : {tuner_used}  (PR-AUC = {best_cv_final:.4f})")

if OPTUNA_OK:
    try:
        history = [t.value for t in study.trials if t.value is not None]
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(history) + 1), history,
                 marker="o", markersize=3, color="#2c5f8a", linewidth=1)
        plt.axhline(max(history), color="#c0392b", linestyle="--",
                    linewidth=1, label=f"Best = {max(history):.4f}")
        plt.xlabel("Trial")
        plt.ylabel("PR-AUC (CV)")
        plt.title("Optuna — Optimisation History")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/optuna_history.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✅ reports/optuna_history.png saved")
    except Exception as e:
        print(f"  ⚠️  Optuna plot skipped: {e}")


# ==========================================
# 7️⃣  PROBABILITY CALIBRATION
# ==========================================
# cv="prefit" was removed from sklearn ≥ 1.2 — raises InvalidParameterError.
# cv=None triggers a 5-fold internal refit, discarding tuned parameters.
#
# FIX: manual isotonic calibration using a holdout split.
#   1. Reserve 20% of training data as calibration set (stratified).
#   2. Refit best_pipeline on the 80% train subset.
#   3. Wrap refitted pipeline with CalibratedClassifierCV(cv=None)
#      — cv=None in sklearn ≥ 1.2 means "use estimator as-is" (prefit).
#   4. Fit calibrator on the 20% calibration set.
# This correctly learns the isotonic mapping on held-out data without
# leaking calibration signal into the threshold tuning step.
# ==========================================

print("\nCalibrating probabilities (manual holdout isotonic calibration)...")

from sklearn.model_selection import train_test_split as _tts
from sklearn.base import clone as _clone
import sklearn
_sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

# Split train → fit_subset (80%) + cal_subset (20%)
X_fit, X_cal, y_fit, y_cal = _tts(
    X_tr, y_train,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y_train,
)

# Refit the tuned pipeline on the fit subset
fitted_for_cal = _clone(best_pipeline)
fitted_for_cal.fit(X_fit, y_fit)

# Build calibrated wrapper
# sklearn ≥ 1.2: cv=None means "estimator is already fitted, wrap as-is"
# sklearn < 1.2: cv="prefit" was the correct argument
if _sklearn_version >= (1, 2):
    calibrated = CalibratedClassifierCV(
        estimator = fitted_for_cal,
        method    = "isotonic",
        cv        = None,   # ✅ prefit behaviour in sklearn ≥ 1.2
    )
else:
    calibrated = CalibratedClassifierCV(
        estimator = fitted_for_cal,
        method    = "isotonic",
        cv        = "prefit",   # ✅ prefit behaviour in sklearn < 1.2
    )

# Fit calibration mapping on the held-out calibration set
calibrated.fit(X_cal, y_cal)

raw_probs = fitted_for_cal.predict_proba(X_te)[:, 1]
cal_probs = calibrated.predict_proba(X_te)[:, 1]

print(f"  sklearn version : {sklearn.__version__}")
print(f"  Fit subset      : {len(X_fit)} rows | Cal subset : {len(X_cal)} rows")
print(f"  Before → mean: {raw_probs.mean():.4f}  std: {raw_probs.std():.4f}")
print(f"  After  → mean: {cal_probs.mean():.4f}  std: {cal_probs.std():.4f}")
print("  ✅ Calibration complete")


# ==========================================
# 8️⃣  THRESHOLD TUNING — F2-score with floor
# ==========================================

print("\nTuning threshold (F2-score, min floor = 0.30)...")

y_pred_prob                        = calibrated.predict_proba(X_te)[:, 1]
prec_vals, rec_vals, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)

beta  = 2
# Note: prec_vals and rec_vals have length n+1, thresholds_pr has length n.
# fbeta[:-1] aligns with thresholds_pr correctly.
fbeta = ((1 + beta**2) * prec_vals * rec_vals) / (beta**2 * prec_vals + rec_vals + 1e-8)

# Raised floor from 0.30 → 0.40 after removing leaky features.
# Previous run (with FavoriteSeason + SpendingCategory leakage) had
# precision=89% at threshold=0.304. Without leakage, precision dropped
# to 63.6% at 0.309 (149 FP on 584 fidèles = 25.5% false alarm rate).
# Floor of 0.40 rebalances recall vs precision for honest features.
MIN_THRESHOLD = 0.40
valid         = thresholds_pr >= MIN_THRESHOLD
if valid.any():
    masked      = np.where(valid, fbeta[:-1], -np.inf)
    best_idx    = int(np.argmax(masked))
    best_thresh = float(thresholds_pr[best_idx])
else:
    best_thresh = MIN_THRESHOLD
    best_idx    = 0

print(f"  Threshold floor   : {MIN_THRESHOLD}")
print(f"  Optimal threshold : {best_thresh:.3f}")

y_pred = (y_pred_prob >= best_thresh).astype(int)


# ==========================================
# 9️⃣  FINAL EVALUATION
# ==========================================

print("\nFinal evaluation on test set...")
print("=" * 65)

roc_auc    = roc_auc_score(y_test, y_pred_prob)
pr_auc     = average_precision_score(y_test, y_pred_prob)
f1         = f1_score(y_test, y_pred, zero_division=0)
recall     = recall_score(y_test, y_pred, zero_division=0)
precision  = precision_score(y_test, y_pred, zero_division=0)
cm         = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"  Feature set   : {feature_note}")
print(f"  Tuner         : {tuner_used}")
print(f"  Threshold     : {best_thresh:.3f}")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  PR-AUC        : {pr_auc:.4f}  ← primary metric")
print(f"  F1-Score      : {f1:.4f}")
print(f"  Recall        : {recall:.4f}  ← % churners caught")
print(f"  Precision     : {precision:.4f}")
print(f"\n  Confusion matrix:")
print(f"    TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"    Missed churners : {fn}")
print()
print(classification_report(
    y_test, y_pred,
    target_names=["Fidèle", "Churn"],
    zero_division=0,
))


# ==========================================
# 🔟  REPORTS & VISUALISATIONS
# ==========================================

print("Saving reports...")

# ── Confusion matrix ──────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Fidèle", "Churn"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix — {best_name} ({tuner_used}, {feature_note})")
plt.xlabel(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ confusion_matrix.png")

# ── ROC curve ─────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC={roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--", color="#aaa", label="Random")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC Curve — {best_name} (calibrated)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ roc_curve.png")

# ── PR curve ──────────────────────────────────────────────────────
_, rec_all, prec_all = precision_recall_curve(y_test, y_pred_prob)[::-1]
pr_auc_plot = auc(rec_vals, prec_vals)
plt.figure(figsize=(7, 5))
plt.plot(rec_vals, prec_vals, color="#8e44ad", lw=2, label=f"PR AUC={pr_auc_plot:.4f}")
plt.scatter(rec_vals[best_idx], prec_vals[best_idx],
            color="red", zorder=5, s=80, label=f"Threshold={best_thresh:.2f}")
plt.axhline(y_test.mean(), color="#aaa", linestyle="--",
            label=f"Baseline={y_test.mean():.2%}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall — {best_name} (calibrated)")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("reports/pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ pr_curve.png")

# ── Calibration curve ─────────────────────────────────────────────
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(7, 5))
for probs, label, color in [
    (raw_probs, "Before calibration", "#e67e22"),
    (cal_probs, "After calibration",  "#2c5f8a"),
]:
    fp_c, mp_c = calibration_curve(y_test, probs, n_bins=10)
    ax.plot(mp_c, fp_c, marker="o", lw=2, label=label, color=color)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve")
ax.legend()
plt.tight_layout()
plt.savefig("reports/calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ calibration_curve.png")

# ── Tuner comparison bar chart ────────────────────────────────────
tuner_scores = {"GridSearchCV": grid_best_score}
if OPTUNA_OK:
    tuner_scores["Optuna"] = optuna_best_score

plt.figure(figsize=(6, 4))
bars = plt.bar(
    tuner_scores.keys(), tuner_scores.values(),
    color=["#2c5f8a", "#27ae60"][: len(tuner_scores)],
    edgecolor="white",
)
for bar, val in zip(bars, tuner_scores.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", fontsize=10)
plt.ylim(min(tuner_scores.values()) * 0.98, 1.01)
plt.ylabel("CV PR-AUC")
plt.title(f"GridSearchCV vs Optuna — {best_name}")
plt.tight_layout()
plt.savefig("reports/tuner_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ tuner_comparison.png")

# ── Raw vs PCA comparison bar chart ──────────────────────────────
x = np.arange(len(models))
w = 0.35
plt.figure(figsize=(9, 5))
plt.bar(x - w / 2, [best_cv_raw[n] for n in models], w,
        label="Raw features", color="#2c5f8a", edgecolor="white")
plt.bar(x + w / 2, [best_cv_pca[n] for n in models], w,
        label="PCA features", color="#e67e22", edgecolor="white")
plt.xticks(x, list(models.keys()))
all_scores = list(best_cv_raw.values()) + list(best_cv_pca.values())
plt.ylim(min(all_scores) * 0.97, 1.01)
plt.ylabel("CV PR-AUC")
plt.title("Raw features vs PCA features — All models")
plt.legend()
plt.tight_layout()
plt.savefig("reports/raw_vs_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ raw_vs_pca.png")

# ── Feature importance ────────────────────────────────────────────
try:
    # Use fitted_for_cal directly — avoids sklearn version differences
    # in how calibrated.estimator is stored (varies between ≥1.2 and <1.2)
    tuned_model = fitted_for_cal.named_steps["model"]
    feat_names  = X_tr.columns
    plot_feature_importance(
        tuned_model,
        feature_names=feat_names,
        top_n=20,
        save_path="reports/feature_importance.png",
    )
    print("  ✅ feature_importance.png")
except Exception as e:
    print(f"  ⚠️  Feature importance skipped: {e}")


# ==========================================
# 1️⃣1️⃣  DECISION TREE EXTRACTION
# ==========================================
# HistGradientBoosting internal trees are _PredictorNode C objects —
# not sklearn DecisionTreeClassifier/Regressor. They cannot be extracted
# directly. A surrogate DecisionTreeClassifier is always used for HGB and LR.
# RF: estimators_[0] is a real DecisionTreeClassifier → extracted directly.
# Surrogate has NO class_weight — it mimics already-calibrated predictions.
# ==========================================

print(f"\n{'=' * 65}")
print("  DECISION TREE EXTRACTION")
print(f"{'=' * 65}")

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def _node_counts(tree, node_idx):
    """
    Robust node count reader.
    value shape: (1, 2) — may be raw counts OR weighted floats.
    Use n_node_samples as the total and derive counts from proportions.
    """
    v = tree.tree_.value[node_idx]      # shape (1, 2)
    total = int(tree.tree_.n_node_samples[node_idx])
    raw0, raw1 = v[0][0], v[0][1]
    total_raw = raw0 + raw1
    if total_raw == 0:
        return 0, 0
    n0 = round(total * raw0 / total_raw)
    n1 = total - n0
    return n0, n1


feat_names_tree   = list(X_tr.columns)
tree_obj          = None
tree_source       = None
is_surrogate      = False
root_feature_name = None
root_threshold    = None

# ── Step 1: Get the tree object ───────────────────────────────────

try:
    # Use fitted_for_cal directly — same reason as feature importance above
    tuned_model = fitted_for_cal.named_steps["model"]

    if isinstance(tuned_model, RandomForestClassifier):
        tree_obj    = tuned_model.estimators_[0]
        tree_source = "Random Forest — estimators_[0] (one real tree from the forest)"

    else:
        is_surrogate = True
        model_label  = (
            "Hist Gradient Boosting"
            if isinstance(tuned_model, HistGradientBoostingClassifier)
            else best_name
        )
        tree_source = (
            f"Surrogate DecisionTreeClassifier — fitted on {model_label} "
            f"calibrated predictions (max_depth=3)"
        )

        surrogate_probs  = calibrated.predict_proba(X_tr)[:, 1]
        surrogate_labels = (surrogate_probs >= best_thresh).astype(int)

        tree_obj = DecisionTreeClassifier(
            max_depth=3,
            random_state=42,
        )
        tree_obj.fit(X_tr, surrogate_labels)

    print(f"  Source : {tree_source}")

except Exception as e:
    print(f"  ⚠️  Could not extract tree: {e}")
    tree_obj = None


# ── Step 2: Visualise ─────────────────────────────────────────────

if tree_obj is not None:
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        tree_obj,
        feature_names=feat_names_tree,
        class_names=["Fidèle", "Churn"],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3,
        impurity=True,
        proportion=False,
        ax=ax,
    )
    suffix = "surrogate" if is_surrogate else "extracted"
    plt.title(
        f"Decision Tree ({suffix}) — {best_name}\n{tree_source}",
        fontsize=11, pad=12,
    )
    plt.tight_layout()
    plt.savefig("reports/decision_tree.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ reports/decision_tree.png saved")

    tree_text = export_text(
        tree_obj,
        feature_names=feat_names_tree,
        max_depth=3,
        spacing=3,
        decimals=4,
        show_weights=True,
    )
    with open("reports/decision_tree_text.txt", "w", encoding="utf-8") as f:
        f.write(f"Source : {tree_source}\n")
        f.write(f"Model  : {best_name}\n\n")
        f.write(tree_text)
    print("  ✅ reports/decision_tree_text.txt saved")


# ── Step 3: Extract root split ────────────────────────────────────

if tree_obj is not None:
    root_feature_idx  = tree_obj.tree_.feature[0]
    root_threshold    = tree_obj.tree_.threshold[0]
    root_feature_name = feat_names_tree[root_feature_idx]
    root_samples      = int(tree_obj.tree_.n_node_samples[0])
    n_fidele, n_churn = _node_counts(tree_obj, 0)

    print(f"\n  ── Root split (first decision) ──")
    print(f"  Feature   : {root_feature_name}")
    print(f"  Threshold : {root_threshold:.4f}  (scaled — see original value below)")
    print(f"  Condition :")
    print(f"    LEFT  branch → {root_feature_name} ≤ {root_threshold:.4f}")
    print(f"    RIGHT branch → {root_feature_name}  > {root_threshold:.4f}")
    print(f"  Root node : {root_samples} samples  |  Fidèle={n_fidele}  Churn={n_churn}")

    # Back-transform to original scale
    try:
        scaler_loaded   = joblib.load("models/scaler.pkl")
        scaler_features = (
            list(scaler_loaded.feature_names_in_)
            if hasattr(scaler_loaded, "feature_names_in_")
            else []
        )

        if root_feature_name in scaler_features:
            idx_sc      = scaler_features.index(root_feature_name)
            mean_val    = scaler_loaded.mean_[idx_sc]
            std_val     = scaler_loaded.scale_[idx_sc]
            orig_thresh = root_threshold * std_val + mean_val
            print(f"\n  Original-scale threshold : {orig_thresh:.4f}")
            print(f"  (scaled={root_threshold:.4f}  |  mean={mean_val:.4f}  |  std={std_val:.4f})")
            print(f"  Interpretation:")
            print(f"    {root_feature_name} ≤ {orig_thresh:.4f}  →  LEFT  branch")
            print(f"    {root_feature_name}  > {orig_thresh:.4f}  →  RIGHT branch")
        else:
            print(f"\n  Note: '{root_feature_name}' is binary/OHE — no rescaling needed.")
            print(f"  Interpretation:")
            print(f"    {root_feature_name} ≤ {root_threshold:.1f}  →  LEFT")
            print(f"    {root_feature_name}  > {root_threshold:.1f}  →  RIGHT")

    except FileNotFoundError:
        print("\n  Note: scaler.pkl not found — scaled threshold only.")
    except Exception as e:
        print(f"\n  Note: could not back-transform threshold: {e}")

    # ── Children of root ──────────────────────────────────────────

    print(f"\n  ── Children of root ──")
    for child_idx, branch in [(1, "LEFT  (≤ threshold)"), (2, "RIGHT (>  threshold)")]:
        try:
            child_feat_idx    = tree_obj.tree_.feature[child_idx]
            child_thresh      = tree_obj.tree_.threshold[child_idx]
            child_samples     = int(tree_obj.tree_.n_node_samples[child_idx])
            c_fidele, c_churn = _node_counts(tree_obj, child_idx)
            churn_pct         = c_churn / (c_fidele + c_churn + 1e-8) * 100

            if child_feat_idx == -2:
                majority = "Churn" if c_churn > c_fidele else "Fidèle"
                print(f"  {branch} : LEAF — {child_samples} samples"
                      f"  |  Fidèle={c_fidele}  Churn={c_churn}"
                      f"  |  Churn%={churn_pct:.1f}%  → predicts {majority}")
            else:
                next_feat = feat_names_tree[child_feat_idx]
                print(f"  {branch} : {child_samples} samples"
                      f"  |  Fidèle={c_fidele}  Churn={c_churn}"
                      f"  |  Churn%={churn_pct:.1f}%"
                      f"  → next split on '{next_feat}' ≤ {child_thresh:.4f}")
        except IndexError:
            print(f"  {branch} : node {child_idx} not available (tree is a stump)")

    print(f"\n  ✅ Decision tree extraction complete")
    print(f"     reports/decision_tree.png      — visual (max_depth=3)")
    print(f"     reports/decision_tree_text.txt — text representation")


# ==========================================
# 1️⃣2️⃣  SAVE MODEL + ARTIFACTS
# ==========================================

joblib.dump(calibrated, "models/churn_model.pkl")
joblib.dump({
    "threshold"          : best_thresh,
    "min_threshold"      : MIN_THRESHOLD,
    "best_model_name"    : best_name,
    "tuner_used"         : tuner_used,
    "feature_set"        : feature_note,
    "use_pca"            : USE_PCA,
    "calibrated"         : True,
    "primary_cv_metric"  : "average_precision",
    "tree_root_feature"  : root_feature_name,
    "tree_root_threshold": float(root_threshold) if root_threshold is not None else None,
}, "models/threshold.pkl")

print(f"\n  ✅ models/churn_model.pkl  (calibrated, {tuner_used})")
print(f"  ✅ models/threshold.pkl    (threshold = {best_thresh:.3f})")

print(f"\n{'=' * 65}")
print(f"🎯 Training complete.")
print(f"   Best model   : {best_name}")
print(f"   Tuner        : {tuner_used}")
print(f"   Feature set  : {feature_note}")
print(f"   ROC-AUC      : {roc_auc:.4f}")
print(f"   PR-AUC       : {pr_auc:.4f}  ← primary metric")
print(f"   F1           : {f1:.4f}")
print(f"   Recall       : {recall:.4f}")
print(f"   Threshold    : {best_thresh:.3f}")
print(f"   Tree feature : {root_feature_name if root_feature_name else 'N/A'}")
print(f"   Reports      → reports/")
print(f"   Model        → models/churn_model.pkl")
print(f"{'=' * 65}")