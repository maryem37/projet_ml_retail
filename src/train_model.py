# ==========================================
# RETAIL ML PROJECT - TRAINING SCRIPT
# ==========================================

import pandas as pd
import numpy as np
import joblib
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


# ==========================================
# 2️⃣ SANITY CHECK — No NaNs allowed
# ==========================================

assert X_train.isnull().sum().sum() == 0, "❌ NaNs found in X_train!"
assert X_test.isnull().sum().sum()  == 0, "❌ NaNs found in X_test!"
print("  ✅ No NaNs in training data.")


# ==========================================
# 3️⃣ DEFINE MODELS TO COMPARE
# ==========================================

print("\nDefining candidate models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 4️⃣ CROSS-VALIDATION COMPARISON
# ==========================================

print("\nRunning cross-validation (5-fold) on all models...")
print("-" * 55)

cv_results = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    cv_results[name] = scores
    print(f"  {name}")
    print(f"    ROC-AUC : {scores.mean():.4f} ± {scores.std():.4f}")

# Pick best model by mean CV score
best_name = max(cv_results, key=lambda k: cv_results[k].mean())
best_cv_score = cv_results[best_name].mean()

# Warn if suspiciously perfect
if best_cv_score > 0.98:
    print(f"\n  ⚠️  WARNING: ROC-AUC={best_cv_score:.4f} is suspiciously high.")
    print("     Check for remaining leaky features in your dataset.")
else:
    print(f"\n  ✅ Best model : {best_name} (ROC-AUC={best_cv_score:.4f})")


# ==========================================
# 5️⃣ HYPERPARAMETER TUNING (Best Model)
# ==========================================

print(f"\nTuning hyperparameters for: {best_name}...")

if best_name == "Random Forest":
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    base_model = RandomForestClassifier(random_state=42, class_weight="balanced")

elif best_name == "Gradient Boosting":
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    }
    base_model = GradientBoostingClassifier(random_state=42)

else:  # Logistic Regression
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    }
    base_model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"  Best params     : {grid_search.best_params_}")
print(f"  Best CV ROC-AUC : {grid_search.best_score_:.4f}")


# ==========================================
# 6️⃣ FINAL EVALUATION ON TEST SET
# ==========================================

print("\nEvaluating best model on test set...")
print("-" * 55)

y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"\n  ROC-AUC Score : {roc_auc:.4f}")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fidèle (0)", "Churn (1)"]))


# ==========================================
# 7️⃣ SAVE REPORTS & VISUALIZATIONS
# ==========================================

print("Saving visualizations to reports/...")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fidèle", "Churn"]
)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix — {best_name}")
plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ confusion_matrix.png saved")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {best_name}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ roc_curve.png saved")

# --- Feature Importance (tree-based models only) ---
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(
        best_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index,hue=importances.index, palette="viridis", legend=False)
    plt.title(f"Top 20 Feature Importances — {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ feature_importance.png saved")

# --- CV Score Comparison Bar Chart ---
cv_means  = {k: v.mean() for k, v in cv_results.items()}
cv_stds   = {k: v.std()  for k, v in cv_results.items()}

plt.figure(figsize=(8, 5))
bars = plt.bar(
    cv_means.keys(),
    cv_means.values(),
    yerr=cv_stds.values(),
    capsize=6,
    color=["#4C72B0", "#DD8452", "#55A868"]
)
plt.ylim(0, 1.05)
plt.ylabel("ROC-AUC (5-fold CV)")
plt.title("Model Comparison — Cross-Validation Scores")
for bar, val in zip(bars, cv_means.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.4f}",
        ha="center", va="bottom", fontsize=10
    )
plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ model_comparison.png saved")


# ==========================================
# 8️⃣ SAVE THE TRAINED MODEL
# ==========================================

print("\nSaving trained model...")
joblib.dump(best_model, "models/churn_model.pkl")
print("  ✅ models/churn_model.pkl saved")

print("\n🎯 Training complete. Next step → predict.py or Flask app.")