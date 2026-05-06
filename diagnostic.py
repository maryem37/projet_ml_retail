# ==========================================
# DIAGNOSTIC SCRIPT v2 — fixed qcut crash
# ==========================================

import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

print("=" * 60)
print("  DIAGNOSTIC 1 — CustomerTenureDays semantics")
print("=" * 60)

print("\nFirst 10 rows:")
cols = ["CustomerTenureDays", "Churn", "Recency", "FirstPurchaseDaysAgo"]
cols = [c for c in cols if c in df.columns]
print(df[cols].head(10).to_string())

print("\nCorrelations with Churn:")
for col in ["CustomerTenureDays", "Recency", "FirstPurchaseDaysAgo"]:
    if col in df.columns:
        corr = df[col].corr(df["Churn"])
        print(f"  {col:<25} corr={corr:+.4f}")

print("\nChurn rate by CustomerTenureDays quartile:")
try:
    # NOTE: duplicates='drop' may produce < 4 bins when values repeat.
    # If we pass 4 fixed labels, pandas raises:
    #   ValueError: Bin labels must be one fewer than the number of bin edges
    tenure_bins = pd.qcut(df["CustomerTenureDays"], q=4, duplicates="drop")
    n_bins = len(tenure_bins.cat.categories)

    # Create readable labels that match the actual number of bins.
    labels = []
    for i, interval in enumerate(tenure_bins.cat.categories):
        q = i + 1
        if i == 0:
            labels.append(f"Q{q}(low) {interval}")
        elif i == n_bins - 1:
            labels.append(f"Q{q}(high) {interval}")
        else:
            labels.append(f"Q{q} {interval}")

    df["tenure_q"] = tenure_bins.cat.rename_categories(labels)
    print(df.groupby("tenure_q", observed=True)["Churn"].agg(["mean", "count"]).round(3))
except Exception as e:
    print(f"  ⚠️  Could not compute tenure quantiles: {e}")
    print("  Showing churn rate overall and by a simple threshold instead...")
    overall = df["Churn"].agg(["mean", "count"]).round(3)
    print(pd.DataFrame([overall], index=["overall"]))
    if "CustomerTenureDays" in df.columns:
        med = df["CustomerTenureDays"].median()
        df["tenure_split"] = np.where(df["CustomerTenureDays"] <= med, f"<= median({med:.0f})", f"> median({med:.0f})")
        print(df.groupby("tenure_split")["Churn"].agg(["mean", "count"]).round(3))

print("\nBasic stats CustomerTenureDays:")
print(df["CustomerTenureDays"].describe().round(2))
print(f"  Zeros : {(df['CustomerTenureDays'] == 0).sum()}")

if "Recency" in df.columns:
    print(f"\nCorrelation CustomerTenureDays vs Recency : {df['CustomerTenureDays'].corr(df['Recency']):.4f}")
    max_recency = df["Recency"].max()
    df["tenure_derived"] = max_recency - df["Recency"]
    match = (df["tenure_derived"] == df["CustomerTenureDays"]).mean()
    print(f"  max(Recency) - Recency == CustomerTenureDays : {match:.2%}")
    print(f"  (if ~100% -> CustomerTenureDays IS derived from Recency -> LEAKAGE)")

print("\n")
print("=" * 60)
print("  DIAGNOSTIC 2 — FavoriteSeason leakage check")
print("=" * 60)

print("\nFavoriteSeason value counts:")
print(df["FavoriteSeason"].value_counts())

print("\nChurn rate per FavoriteSeason:")
fs_churn = df.groupby("FavoriteSeason")["Churn"].agg(["mean", "count"]).round(3)
fs_churn.columns = ["ChurnRate", "Count"]
print(fs_churn.sort_values("ChurnRate", ascending=False))

if "PreferredMonth" in df.columns:
    print("\nChurn rate per PreferredMonth:")
    pm_churn = df.groupby("PreferredMonth")["Churn"].agg(["mean", "count"]).round(3)
    pm_churn.columns = ["ChurnRate", "Count"]
    print(pm_churn)

    print("\nCross-tab FavoriteSeason vs PreferredMonth:")
    ct = pd.crosstab(df["FavoriteSeason"], df["PreferredMonth"])
    print(ct)

    print("\n  Is FavoriteSeason derived from PreferredMonth?")
    def month_to_season(m):
        if m in [12, 1, 2]:  return "Hiver"
        if m in [3, 4, 5]:   return "Printemps"
        if m in [6, 7, 8]:   return "Été"
        if m in [9, 10, 11]: return "Automne"
        return "Unknown"

    df["season_from_month"] = df["PreferredMonth"].apply(month_to_season)
    match = (df["season_from_month"] == df["FavoriteSeason"]).mean()
    print(f"  Match rate : {match:.2%}")
    print(f"  (> 80% -> FavoriteSeason IS PreferredMonth in disguise -> LEAKAGE)")
else:
    print("\n  PreferredMonth not in raw data")
    if "Recency" in df.columns:
        print("\nMean Recency + CustomerTenureDays per FavoriteSeason:")
        print(df.groupby("FavoriteSeason")[["Churn", "Recency", "CustomerTenureDays"]].mean().round(3))