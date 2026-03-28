# ==========================================
# RETAIL ML PROJECT - OPTUNA TUNING SCRIPT
# ==========================================
# Optuna is smarter than GridSearchCV:
# Instead of testing ALL combinations blindly,
# Optuna uses results from previous trials to
# GUIDE the search toward better parameters.
#
# This is called "Bayesian Optimization" and
# is much faster than GridSearch for large
# hyperparameter spaces.
#
# GridSearch: tries 12 combinations → 60 fits
# Optuna:     tries 50 trials but skips bad
#             regions → finds better params faster
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️  Optuna not installed. Run: pip install optuna")
    OPTUNA_AVAILABLE = False


# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

print("Loading preprocessed data...")

X_train = pd.read_csv("data/train_test/X_train.csv")
X_test  = pd.read_csv("data/train_test/X_test.csv")
y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()
y_test  = pd.read_csv("data/train_test/y_test.csv").squeeze()

print(f"  X_train : {X_train.shape} | X_test : {X_test.shape}")


if not OPTUNA_AVAILABLE:
    print("\n❌ Optuna not available. Install with: pip install optuna")
    exit()


# ==========================================
# 2️⃣ DEFINE OBJECTIVE FUNCTIONS
# ==========================================
# Each objective function:
#   - receives a "trial" object from Optuna
#   - asks Optuna to suggest hyperparameters
#   - trains and evaluates the model
#   - returns the score to maximize
# ==========================================

def objective_rf(trial):
    """Objective for Random Forest."""
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 50, 300),
        "max_depth"       : trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features"    : trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    model  = RandomForestClassifier(**params, random_state=42, class_weight="balanced")
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def objective_gb(trial):
    """Objective for Gradient Boosting."""
    params = {
        "n_estimators" : trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth"    : trial.suggest_int("max_depth", 2, 8),
        "subsample"    : trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }
    model  = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def objective_lr(trial):
    """Objective for Logistic Regression."""
    params = {
        "C"      : trial.suggest_float("C", 0.001, 100, log=True),
        "solver" : trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        "max_iter": 1000,
    }
    model  = LogisticRegression(**params, random_state=42, class_weight="balanced")
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ==========================================
# 3️⃣ RUN OPTUNA STUDIES
# ==========================================

N_TRIALS = 30  # Number of trials per model

objectives = {
    "Random Forest"      : objective_rf,
    "Gradient Boosting"  : objective_gb,
    "Logistic Regression": objective_lr,
}

study_results = {}

for model_name, objective in objectives.items():
    print(f"\nOptimizing {model_name} ({N_TRIALS} trials)...")

    study = optuna.create_study(
        direction="maximize",           # Maximize ROC-AUC
        study_name=model_name,
        sampler=optuna.samplers.TPESampler(seed=42)  # TPE = smart Bayesian sampler
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_trial = study.best_trial
    print(f"  ✅ Best ROC-AUC : {best_trial.value:.4f}")
    print(f"  Best params     : {best_trial.params}")

    study_results[model_name] = {
        "study"     : study,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
    }


# ==========================================
# 4️⃣ COMPARE OPTUNA vs GRIDSEARCH
# ==========================================

print("\n" + "="*55)
print("  OPTUNA RESULTS SUMMARY")
print("="*55)

best_overall_name  = max(study_results, key=lambda k: study_results[k]["best_value"])
best_overall_score = study_results[best_overall_name]["best_value"]

for name, res in study_results.items():
    marker = " ← BEST" if name == best_overall_name else ""
    print(f"  {name:<25} ROC-AUC = {res['best_value']:.4f}{marker}")

print(f"\n  🏆 Best model : {best_overall_name}")
print(f"  🏆 Best score : {best_overall_score:.4f}")


# ==========================================
# 5️⃣ TRAIN BEST MODEL WITH OPTUNA PARAMS
# ==========================================

print(f"\nTraining final model with Optuna best params...")

best_params = study_results[best_overall_name]["best_params"]

if best_overall_name == "Random Forest":
    final_model = RandomForestClassifier(
        **best_params, random_state=42, class_weight="balanced"
    )
elif best_overall_name == "Gradient Boosting":
    final_model = GradientBoostingClassifier(**best_params, random_state=42)
else:
    final_model = LogisticRegression(
        **best_params, random_state=42, class_weight="balanced"
    )

final_model.fit(X_train, y_train)

y_pred_prob = final_model.predict_proba(X_test)[:, 1]
test_roc    = roc_auc_score(y_test, y_pred_prob)
print(f"  Test ROC-AUC (Optuna model) : {test_roc:.4f}")


# ==========================================
# 6️⃣ VISUALIZE OPTUNA OPTIMIZATION HISTORY
# ==========================================

print("\nSaving Optuna visualizations...")
os.makedirs("reports", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, res) in zip(axes, study_results.items()):
    study  = res["study"]
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    best_so_far = [max(values[:i+1]) for i in range(len(values))]

    ax.plot(range(1, len(values)+1), values,
            color="#aaa", linewidth=0.8, alpha=0.7, label="Trial score")
    ax.plot(range(1, len(best_so_far)+1), best_so_far,
            color="#2c5f8a", linewidth=2, label="Best so far")
    ax.axhline(y=res["best_value"], color="#c0392b",
               linestyle="--", linewidth=1.2,
               label=f"Best: {res['best_value']:.4f}")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize=8)
    ax.set_ylim(max(0, min(values) - 0.05), 1.05)

plt.suptitle("Optuna Optimization History — All Models", fontsize=13)
plt.tight_layout()
plt.savefig("reports/optuna_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/optuna_history.png saved")

# --- Parameter importance (best model only) ---
try:
    best_study = study_results[best_overall_name]["study"]
    importances = optuna.importance.get_param_importances(best_study)

    plt.figure(figsize=(8, 4))
    params_names  = list(importances.keys())
    params_values = list(importances.values())
    plt.barh(params_names, params_values, color="#2c5f8a", edgecolor="white")
    plt.xlabel("Importance")
    plt.title(f"Hyperparameter Importance — {best_overall_name}")
    plt.tight_layout()
    plt.savefig("reports/optuna_param_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ reports/optuna_param_importance.png saved")
except Exception as e:
    print(f"  ⚠️  Could not plot param importance: {e}")


# ==========================================
# 7️⃣ SAVE OPTUNA MODEL
# ==========================================

os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/churn_model_optuna.pkl")
print("  ✅ models/churn_model_optuna.pkl saved")

print("\n" + "="*55)
print("  OPTUNA TUNING COMPLETE")
print("="*55)
print(f"  Best model  : {best_overall_name}")
print(f"  Best params : {best_params}")
print(f"  Test ROC-AUC: {test_roc:.4f}")
print("\n  Reports:")
print("    reports/optuna_history.png")
print("    reports/optuna_param_importance.png")
print("    models/churn_model_optuna.pkl")
print("="*55)