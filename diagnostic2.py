# ==========================================
# DIAGNOSTIC 2 — SpendingCategory leakage
# ==========================================
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

print("=" * 65)
print("  SpendingCategory — is it derived from MonetaryTotal?")
print("=" * 65)

print("\nSpendingCategory value counts:")
print(df["SpendingCategory"].value_counts())

print("\nMonetaryTotal stats per SpendingCategory:")
print(df.groupby("SpendingCategory")["MonetaryTotal"].describe().round(0)[
    ["min","25%","50%","75%","max","mean"]
])

print("\nChurn rate per SpendingCategory:")
sc = df.groupby("SpendingCategory")["Churn"].agg(["mean","count"]).round(3)
sc.columns = ["ChurnRate","Count"]
print(sc.sort_values("ChurnRate", ascending=False))

# Key test: can SpendingCategory be perfectly predicted from MonetaryTotal?
print("\n  Threshold detection (sorted by MonetaryTotal median):")
medians = df.groupby("SpendingCategory")["MonetaryTotal"].median().sort_values()
print(medians)

# Check if ranges are non-overlapping
print("\n  MonetaryTotal ranges per category:")
for cat in df["SpendingCategory"].unique():
    sub = df[df["SpendingCategory"] == cat]["MonetaryTotal"]
    print(f"  {cat:<12} min={sub.min():>10.0f}  max={sub.max():>10.0f}  median={sub.median():>10.0f}")

# Perfect derivation test
print("\n  Overlap check between categories (sorted by median):")
cats_sorted = medians.index.tolist()
for i in range(len(cats_sorted)-1):
    cat_a = cats_sorted[i]
    cat_b = cats_sorted[i+1]
    max_a = df[df["SpendingCategory"]==cat_a]["MonetaryTotal"].max()
    min_b = df[df["SpendingCategory"]==cat_b]["MonetaryTotal"].min()
    overlap = max_a >= min_b
    print(f"  {cat_a} max={max_a:.0f}  vs  {cat_b} min={min_b:.0f}  → overlap={'YES ← ranges overlap' if overlap else 'NO  ← clean cut'}")

print("\n")
print("=" * 65)
print("  AvgQuantityPerTransaction — leakage check")
print("=" * 65)
if "AvgQuantityPerTransaction" in df.columns:
    print("\nCorrelation with MonetaryTotal:", df["AvgQuantityPerTransaction"].corr(df["MonetaryTotal"]))
    print("Correlation with Churn        :", df["AvgQuantityPerTransaction"].corr(df["Churn"]))
    print("\nBasic stats:")
    print(df["AvgQuantityPerTransaction"].describe().round(2))