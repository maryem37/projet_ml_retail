# ==========================================
# RETAIL ML PROJECT - REGRESSION SCRIPT
# ==========================================
# Task: Predict how much a customer will spend
#       (MonetaryTotal) — a regression problem.
#
# This complements the classification task
# (predicting Churn) by estimating customer
# lifetime value instead of churn probability.
#
# Models compared:
#   - Linear Regression (baseline)
#   - Ridge Regression  (L2 regularization)
#   - Random Forest Regressor
#   - Gradient Boosting Regressor
#
# Metrics used:
#   - MAE  (Mean Absolute Error)     → average £ error
#   - RMSE (Root Mean Squared Error) → penalizes big errors
#   - R²   (R-squared)               → % variance explained
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV


# ==========================================
# 0️⃣ SETUP
# ==========================================

os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

print("Loading data for regression...")

# Load raw data to get original MonetaryTotal
# (preprocessing applied log transform — we want
# the original scale for interpretable predictions)
df_raw = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

# Load preprocessed features (already cleaned, encoded, scaled)
X_train = pd.read_csv("data/train_test/X_train.csv")
X_test  = pd.read_csv("data/train_test/X_test.csv")
y_churn_train = pd.read_csv("data/train_test/y_train.csv").squeeze()
y_churn_test  = pd.read_csv("data/train_test/y_test.csv").squeeze()

print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")


# ==========================================
# 2️⃣ BUILD REGRESSION TARGET
# ==========================================
# Target: MonetaryTotal (customer total spend in £)
# We rebuild it from raw data aligned to train/test split.
# ==========================================

print("\nBuilding regression target (MonetaryTotal)...")

# Use the same random_state=42 split to get matching indices
from sklearn.model_selection import train_test_split

df_raw_clean = df_raw.dropna(subset=["MonetaryTotal"]).reset_index(drop=True)

# Align raw target with preprocessed features using index positions
# We split the raw target the same way preprocessing.py did
y_monetary = df_raw_clean["MonetaryTotal"].values[:len(X_train) + len(X_test)]

# Split 80/20 matching original split
split_idx   = len(X_train)
y_reg_train = pd.Series(
    np.sign(y_monetary[:split_idx]) * np.log1p(np.abs(y_monetary[:split_idx])),
    name="MonetaryTotal_log"
)
y_reg_test = pd.Series(
    np.sign(y_monetary[split_idx:split_idx + len(X_test)]) * np.log1p(np.abs(y_monetary[split_idx:split_idx + len(X_test)])),
    name="MonetaryTotal_log"
)
# Handle any remaining length mismatch gracefully
min_train = min(len(X_train), len(y_reg_train))
min_test  = min(len(X_test),  len(y_reg_test))

X_train_reg = X_train.iloc[:min_train]
X_test_reg  = X_test.iloc[:min_test]
y_reg_train = y_reg_train.iloc[:min_train]
y_reg_test  = y_reg_test.iloc[:min_test]

print(f"  Regression train set : {X_train_reg.shape[0]} samples")
print(f"  Regression test set  : {X_test_reg.shape[0]} samples")
print(f"  Target range         : £{y_reg_train.min():.0f} → £{y_reg_train.max():.0f}")
print(f"  Target mean          : £{y_reg_train.mean():.2f}")
print(f"  Target std           : £{y_reg_train.std():.2f}")


# ==========================================
# 3️⃣ VISUALIZE TARGET DISTRIBUTION
# ==========================================

print("\nPlotting target distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Raw distribution
axes[0].hist(y_reg_train, bins=50, color="#2c5f8a", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("MonetaryTotal (£)")
axes[0].set_ylabel("Count")
axes[0].set_title("MonetaryTotal Distribution (Raw)")
axes[0].axvline(y_reg_train.mean(), color="#c0392b", linestyle="--",
                label=f"Mean: £{y_reg_train.mean():.0f}")
axes[0].legend()

# Log-transformed distribution
log_target = np.sign(y_reg_train) * np.log1p(np.abs(y_reg_train))
axes[1].hist(log_target, bins=50, color="#27ae60", edgecolor="white", alpha=0.85)
axes[1].set_xlabel("log1p(MonetaryTotal)")
axes[1].set_ylabel("Count")
axes[1].set_title("MonetaryTotal Distribution (Log-transformed)")

plt.suptitle("Regression Target — MonetaryTotal Analysis", fontsize=13)
plt.tight_layout()
plt.savefig("reports/regression_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/regression_target_distribution.png saved")


# ==========================================
# 4️⃣ DEFINE REGRESSION MODELS
# ==========================================

print("\nDefining regression models...")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression" : Ridge(alpha=1.0),
    "Random Forest"    : RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=RANDOM_STATE
    ),
}


# ==========================================
# 5️⃣ CROSS-VALIDATION COMPARISON
# ==========================================
# Use negative MAE as scoring (sklearn convention:
# higher = better, so negative MAE is maximized)
# ==========================================

print("\nRunning cross-validation (5-fold, scoring=neg_MAE)...")
print("-" * 55)

cv_results = {}

for name, model in models.items():
    scores_mae = cross_val_score(
        model, X_train_reg, y_reg_train,
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    scores_r2 = cross_val_score(
        model, X_train_reg, y_reg_train,
        cv=5, scoring="r2", n_jobs=-1
    )

    cv_results[name] = {
        "mae": -scores_mae.mean(),
        "r2" :  scores_r2.mean(),
    }

    print(f"  {name}")
    print(f"    MAE : £{-scores_mae.mean():.2f} ± £{scores_mae.std():.2f}")
    print(f"    R²  : {scores_r2.mean():.4f} ± {scores_r2.std():.4f}")

# Best model by R²
best_name = max(cv_results, key=lambda k: cv_results[k]["r2"])
print(f"\n  ✅ Best model : {best_name} (R²={cv_results[best_name]['r2']:.4f})")


# ==========================================
# 6️⃣ HYPERPARAMETER TUNING — GridSearchCV
# ==========================================

print(f"\nTuning hyperparameters for: {best_name}...")

if best_name == "Random Forest":
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth"   : [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

elif best_name == "Gradient Boosting":
    param_grid = {
        "n_estimators" : [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth"    : [3, 5],
    }
    base_model = GradientBoostingRegressor(random_state=RANDOM_STATE)

elif best_name == "Ridge Regression":
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    base_model = Ridge()

else:  # Linear Regression has no hyperparameters
    print("  Linear Regression has no hyperparameters to tune.")
    base_model = LinearRegression()
    param_grid = {}

if param_grid:
    grid_search = GridSearchCV(
        base_model, param_grid,
        cv=5, scoring="r2", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_reg, y_reg_train)
    best_model = grid_search.best_estimator_
    print(f"  Best params : {grid_search.best_params_}")
    print(f"  Best CV R²  : {grid_search.best_score_:.4f}")
else:
    best_model = base_model
    best_model.fit(X_train_reg, y_reg_train)


# ==========================================
# 7️⃣ FINAL EVALUATION ON TEST SET
# ==========================================

print("\nEvaluating best regression model on test set...")
print("-" * 55)

y_pred = best_model.predict(X_test_reg)

mae  = mean_absolute_error(y_reg_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
r2   = r2_score(y_reg_test, y_pred)

print(f"\n  MAE  (Mean Absolute Error)     : £{mae:.2f}")
print(f"  RMSE (Root Mean Squared Error) : £{rmse:.2f}")
print(f"  R²   (Explained Variance)      : {r2:.4f}")
print(f"\n  Interpretation:")
print(f"    On average, predictions are off by £{mae:.0f}")
print(f"    The model explains {r2*100:.1f}% of spend variance")


# ==========================================
# 8️⃣ VISUALIZATIONS
# ==========================================

print("\nSaving regression visualizations...")

# --- Actual vs Predicted scatter ---
plt.figure(figsize=(8, 6))
plt.scatter(y_reg_test, y_pred, alpha=0.35, s=15,
            color="#2c5f8a", edgecolors="none")

# Perfect prediction line
min_val = min(y_reg_test.min(), y_pred.min())
max_val = max(y_reg_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val],
         color="#c0392b", linestyle="--", linewidth=1.5, label="Perfect prediction")

plt.xlabel("Actual MonetaryTotal (£)", fontsize=11)
plt.ylabel("Predicted MonetaryTotal (£)", fontsize=11)
plt.title(f"Actual vs Predicted — {best_name}\nMAE=£{mae:.0f} | RMSE=£{rmse:.0f} | R²={r2:.3f}",
          fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("reports/regression_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/regression_actual_vs_predicted.png")

# --- Residuals plot ---
residuals = y_reg_test.values - y_pred

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Residuals vs predicted
axes[0].scatter(y_pred, residuals, alpha=0.35, s=15,
                color="#e67e22", edgecolors="none")
axes[0].axhline(0, color="#c0392b", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Predicted MonetaryTotal (£)")
axes[0].set_ylabel("Residual (Actual − Predicted)")
axes[0].set_title("Residuals vs Predicted")

# Residuals distribution
axes[1].hist(residuals, bins=50, color="#2c5f8a", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="#c0392b", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Residual (£)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Residual Distribution (mean={residuals.mean():.1f})")

plt.suptitle(f"Residual Analysis — {best_name}", fontsize=13)
plt.tight_layout()
plt.savefig("reports/regression_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/regression_residuals.png")

# --- Model comparison bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

names  = list(cv_results.keys())
maes   = [cv_results[n]["mae"] for n in names]
r2s    = [cv_results[n]["r2"]  for n in names]
colors = ["#2c5f8a", "#27ae60", "#e67e22", "#c0392b"]

# MAE bars
bars = axes[0].bar(names, maes, color=colors, edgecolor="white")
for bar, val in zip(bars, maes):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"£{val:.0f}", ha="center", fontsize=9)
axes[0].set_ylabel("MAE (£) — lower is better")
axes[0].set_title("Cross-Validation MAE Comparison")
axes[0].tick_params(axis="x", rotation=15)

# R² bars
bars2 = axes[1].bar(names, r2s, color=colors, edgecolor="white")
for bar, val in zip(bars2, r2s):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", fontsize=9)
axes[1].set_ylabel("R² — higher is better")
axes[1].set_title("Cross-Validation R² Comparison")
axes[1].set_ylim(0, 1.1)
axes[1].tick_params(axis="x", rotation=15)

plt.suptitle("Regression Model Comparison", fontsize=13)
plt.tight_layout()
plt.savefig("reports/regression_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/regression_model_comparison.png")

# --- Feature importance (if tree-based) ---
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(
        best_model.feature_importances_,
        index=X_train_reg.columns
    ).sort_values(ascending=True).tail(20)

    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh", color="#2c5f8a", edgecolor="white")
    plt.xlabel("Importance")
    plt.title(f"Top 20 Feature Importances — {best_name} (Regression)")
    plt.tight_layout()
    plt.savefig("reports/regression_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ reports/regression_feature_importance.png")


# ==========================================
# 9️⃣ SAVE MODEL
# ==========================================

joblib.dump(best_model, "models/regression_model.pkl")
print("\n  ✅ models/regression_model.pkl saved")


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "="*55)
print("  REGRESSION COMPLETE")
print("="*55)
print(f"  Task        : Predict MonetaryTotal (£)")
print(f"  Best model  : {best_name}")
print(f"  MAE         : £{mae:.2f}")
print(f"  RMSE        : £{rmse:.2f}")
print(f"  R²          : {r2:.4f}")
print(f"\n  Reports:")
print("    reports/regression_target_distribution.png")
print("    reports/regression_actual_vs_predicted.png")
print("    reports/regression_residuals.png")
print("    reports/regression_model_comparison.png")
print("    models/regression_model.pkl")
print("="*55)