# ==========================================
# RETAIL ML PROJECT - TRAINING SCRIPT
# ==========================================

import pandas as pd
import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score, GridSearchCV

# Import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import plot_feature_importance, plot_correlation_heatmap, run_pca, plot_pca_2d


# ==========================================
# 1️⃣ LOAD PREPROCESSED DATA
# ==========================================

print("Loading preprocessed data...")

X_train = pd.read_csv("data/train_test/X_train.csv")
X_test  = pd.read_csv("data/train_test/X_test.csv")
y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()
y_test  = pd.read_csv("data/train_test/y_test.csv").squeeze()

print(f"  X_train shape      : {X_train.shape}")
print(f"  X_test  shape      : {X_test.shape}")
print(f"  Churn rate (train) : {y_train.mean():.2%}")
print(f"  Churn rate (test)  : {y_test.mean():.2%}")

assert X_train.isnull().sum().sum() == 0, "NaNs found in X_train!"
assert X_test.isnull().sum().sum()  == 0, "NaNs found in X_test!"
print("  ✅ No NaNs in training data.")


# ==========================================
# 2️⃣ CLASS IMBALANCE — SMOTE
# ==========================================
# SMOTE (Synthetic Minority Over-sampling
# Technique) creates synthetic samples of the
# minority class (Churn=1) to balance the dataset.
#
# Our churn rate is 33% — not severely imbalanced
# but we demonstrate SMOTE and compare results.
#
# IMPORTANT: SMOTE is applied ONLY on X_train.
# NEVER on X_test (would contaminate evaluation).
# ==========================================

print("\nHandling class imbalance with SMOTE...")
print(f"  Before SMOTE — Fidèle: {(y_train==0).sum()} | Churn: {(y_train==1).sum()}")

try:
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"  After  SMOTE — Fidèle: {(y_train_smote==0).sum()} | Churn: {(y_train_smote==1).sum()}")
    print(f"  SMOTE added {len(X_train_smote) - len(X_train)} synthetic samples")

    # Convert back to DataFrame to keep column names
    X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
    y_train_smote = pd.Series(y_train_smote, name="Churn")
    SMOTE_AVAILABLE = True

except ImportError:
    print("  ⚠️  imbalanced-learn not installed.")
    print("      Run: pip install imbalanced-learn")
    print("      Continuing without SMOTE...")
    X_train_smote = X_train.copy()
    y_train_smote = y_train.copy()
    SMOTE_AVAILABLE = False

# --- Plot class distribution before/after SMOTE ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (X, y, title) in zip(axes, [
    (X_train,       y_train,       "Before SMOTE"),
    (X_train_smote, y_train_smote, "After SMOTE" if SMOTE_AVAILABLE else "No SMOTE"),
]):
    counts = y.value_counts()
    bars   = ax.bar(
        ["Fidèle (0)", "Churn (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=["#2c5f8a", "#c0392b"],
        edgecolor="white"
    )
    for bar, val in zip(bars, [counts.get(0,0), counts.get(1,0)]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 10,
                str(val), ha="center", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Number of customers")
    ax.set_ylim(0, max(counts) * 1.2)

plt.suptitle("Class Distribution — SMOTE Effect", fontsize=13)
plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/smote_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/smote_distribution.png saved")


# ==========================================
# 3️⃣ CORRELATION HEATMAP
# ==========================================

print("\nGenerating correlation heatmap...")
plot_correlation_heatmap(
    X_train,
    title="Training Data — Feature Correlation Matrix",
    save_path="reports/correlation_heatmap.png"
)


# ==========================================
# 4️⃣ PCA ANALYSIS
# ==========================================

print("\nRunning PCA analysis...")
X_train_pca, X_test_pca, pca, n_comp = run_pca(
    X_train, X_test,
    variance_threshold=0.95,
    save_path="reports/pca_variance.png"
)
plot_pca_2d(X_train_pca, y_train, save_path="reports/pca_2d.png")


# ==========================================
# 5️⃣ DEFINE MODELS
# ==========================================

print("\nDefining candidate models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    )
}


# ==========================================
# 6️⃣ CROSS-VALIDATION — With & Without SMOTE
# ==========================================

print("\nRunning cross-validation (5-fold ROC-AUC)...")
print("-" * 55)

cv_results = {}

for name, model in models.items():
    # Without SMOTE
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="roc_auc", n_jobs=-1
    )
    cv_results[name] = scores
    print(f"  {name}")
    print(f"    Without SMOTE → ROC-AUC : {scores.mean():.4f} ± {scores.std():.4f}")

    # With SMOTE (if available)
    if SMOTE_AVAILABLE:
        scores_smote = cross_val_score(
            model, X_train_smote, y_train_smote,
            cv=5, scoring="roc_auc", n_jobs=-1
        )
        print(f"    With    SMOTE → ROC-AUC : {scores_smote.mean():.4f} ± {scores_smote.std():.4f}")

best_name     = max(cv_results, key=lambda k: cv_results[k].mean())
best_cv_score = cv_results[best_name].mean()

if best_cv_score > 0.98:
    print(f"\n  ⚠️  ROC-AUC={best_cv_score:.4f} suspiciously high (synthetic data).")
else:
    print(f"\n  ✅ Best model : {best_name} (ROC-AUC={best_cv_score:.4f})")


# ==========================================
# 7️⃣ HYPERPARAMETER TUNING — GridSearchCV
# ==========================================

print(f"\nTuning hyperparameters (GridSearchCV) for: {best_name}...")

if best_name == "Random Forest":
    param_grid = {
        "n_estimators"    : [100, 200],
        "max_depth"       : [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    base_model = RandomForestClassifier(random_state=42, class_weight="balanced")

elif best_name == "Gradient Boosting":
    param_grid = {
        "n_estimators" : [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth"    : [3, 5]
    }
    base_model = GradientBoostingClassifier(random_state=42)

else:
    param_grid = {
        "C"     : [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    }
    base_model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )

grid_search = GridSearchCV(
    base_model, param_grid,
    cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"  Best params     : {grid_search.best_params_}")
print(f"  Best CV ROC-AUC : {grid_search.best_score_:.4f}")


# ==========================================
# 8️⃣ FINAL EVALUATION ON TEST SET
# ==========================================

print("\nEvaluating best model on test set...")
print("-" * 55)

y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
roc_auc     = roc_auc_score(y_test, y_pred_prob)

print(f"\n  ROC-AUC Score : {roc_auc:.4f}")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fidèle (0)", "Churn (1)"]))


# ==========================================
# 9️⃣ SAVE REPORTS & VISUALIZATIONS
# ==========================================

print("Saving visualizations to reports/...")

# Confusion Matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fidèle", "Churn"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix — {best_name}")
plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#c0392b", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="#aaa", linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {best_name}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ roc_curve.png")

# Feature Importance
plot_feature_importance(
    best_model, feature_names=X_train.columns,
    top_n=20, save_path="reports/feature_importance.png"
)

# Model Comparison Bar Chart
cv_means = {k: v.mean() for k, v in cv_results.items()}
cv_stds  = {k: v.std()  for k, v in cv_results.items()}

plt.figure(figsize=(8, 5))
bars = plt.bar(
    cv_means.keys(), cv_means.values(),
    yerr=cv_stds.values(), capsize=6,
    color=["#2c5f8a", "#e67e22", "#27ae60"], edgecolor="white"
)
plt.ylim(0, 1.1)
plt.ylabel("ROC-AUC (5-fold CV)")
plt.title("Model Comparison — Cross-Validation Scores")
for bar, val in zip(bars, cv_means.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{val:.4f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ model_comparison.png")


# ==========================================
# 🔟 SAVE TRAINED MODEL
# ==========================================

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/churn_model.pkl")
print("\n  ✅ models/churn_model.pkl saved")
print("\n🎯 Training complete. Reports saved in reports/")