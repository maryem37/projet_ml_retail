# ==========================================
# RETAIL ML PROJECT - LEAKAGE DIAGNOSTIC
# ==========================================
# Run this script to identify which features
# are leaking the Churn target and causing
# perfect scores (ROC-AUC = 1.0000).
#
# Usage:
#   python src/find_leaky_features.py
# ==========================================

import pandas as pd
import numpy as np

print("=" * 60)
print("  LEAKAGE DIAGNOSTIC")
print("=" * 60)

X_train = pd.read_csv("data/train_test/X_train.csv")
y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()

print(f"  X_train shape : {X_train.shape}")
print(f"  Churn rate    : {y_train.mean():.2%}\n")

# ── 1. Correlation with target ─────────────────────────
print("─" * 60)
print("  STEP 1 — Pearson correlation with Churn target")
print("─" * 60)

correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)

print("\n  Top 20 most correlated features:")
print(correlations.head(20).to_string())

leaky_high   = correlations[correlations > 0.70]
leaky_medium = correlations[(correlations > 0.40) & (correlations <= 0.70)]

print(f"\n  🔴 HIGH correlation > 0.70 (almost certainly leaky): {len(leaky_high)}")
for col, val in leaky_high.items():
    print(f"      {col:<45} {val:.4f}")

print(f"\n  🟠 MEDIUM correlation 0.40–0.70 (investigate):       {len(leaky_medium)}")
for col, val in leaky_medium.items():
    print(f"      {col:<45} {val:.4f}")


# ── 2. OHE column check — AccountStatus & CustomerType ─
print("\n" + "─" * 60)
print("  STEP 2 — Known semantic leakers (OHE columns)")
print("─" * 60)

# These are the most likely culprits in retail churn datasets
known_suspects = [
    # AccountStatus — Closed/Suspended accounts ARE churned
    "AccountStatus_Closed",
    "AccountStatus_Suspended",
    "AccountStatus_Pending",
    # CustomerType — "Perdu" (lost) directly means churned
    "CustomerType_Perdu",
    "CustomerType_Régulier",
    "CustomerType_Hyperactif",
    "CustomerType_Occasionnel",
    "CustomerType_Nouveau",
]

print("\n  Checking known suspect columns...")
found_suspects = []
for col in known_suspects:
    if col in X_train.columns:
        corr = abs(X_train[col].corr(y_train))
        flag = "🔴 LEAKY" if corr > 0.40 else "🟢 ok"
        print(f"    {flag}  {col:<45} corr={corr:.4f}")
        if corr > 0.40:
            found_suspects.append(col)
    else:
        # Try partial match for OHE variations
        matches = [c for c in X_train.columns if col.split("_")[0] in c]
        if matches:
            for m in matches[:3]:
                corr = abs(X_train[m].corr(y_train))
                flag = "🔴 LEAKY" if corr > 0.40 else "🟢 ok"
                print(f"    {flag}  {m:<45} corr={corr:.4f}")


# ── 3. Quick model test — feature importances ──────────
print("\n" + "─" * 60)
print("  STEP 3 — Random Forest feature importances")
print("─" * 60)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(
    rf.feature_importances_, index=X_train.columns
).sort_values(ascending=False)

print("\n  Top 20 most important features (RF):")
print(importances.head(20).to_string())

top_feature     = importances.index[0]
top_importance  = importances.iloc[0]

if top_importance > 0.30:
    print(f"\n  🔴 WARNING: '{top_feature}' has importance {top_importance:.4f}")
    print(f"     A single feature explaining >30% is a strong leakage signal.")


# ── 4. Recommended drops ───────────────────────────────
print("\n" + "─" * 60)
print("  STEP 4 — Recommended additional drops")
print("─" * 60)

to_drop = set()

# High correlation leakers
for col in leaky_high.index:
    to_drop.add(col)

# Known semantic leakers found
for col in found_suspects:
    # Find the base column name (before OHE suffix)
    base = col.rsplit("_", 1)[0]
    # Add all OHE variants of this column
    variants = [c for c in X_train.columns if c.startswith(base + "_") or c == base]
    for v in variants:
        to_drop.add(v)

# Top RF importance — if single feature dominates
if top_importance > 0.30:
    base = top_feature.rsplit("_", 1)[0]
    variants = [c for c in X_train.columns if c.startswith(base + "_") or c == base]
    for v in variants:
        to_drop.add(v)

print("\n  Add these to columns_to_drop in preprocessing.py:")
print()
for col in sorted(to_drop):
    print(f'    "{col}",')

print()
print("  Then re-run:")
print("    python src/preprocessing.py")
print("    python src/train_model.py")
print()
print("  Expected scores after fixing leakage:")
print("    ROC-AUC : 0.75 – 0.90")
print("    F1      : 0.60 – 0.78")
print("    Recall  : 0.65 – 0.82")
print("=" * 60)