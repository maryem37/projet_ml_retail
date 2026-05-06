# ==========================================
# RETAIL ML PROJECT - PREPROCESSING SCRIPT
# ==========================================
# FIXES vs previous version:
#   ✅ RegistrationDate → parsed → RegYear/RegMonth/RegDay/RegWeekday extracted
#      then dropped (not just dropped raw)
#   ✅ LastLoginIP → IsPrivateIP flag + IP class (A/B/C) extracted
#      then dropped (not just dropped raw)
#   ✅ Country → Target Encoding (smoothed, k=10) — high cardinality 37+ values
#      fit on train only, applied to test (no leakage)
#   ✅ PCA applied → X_train_pca / X_test_pca saved separately
#      both raw-scaled and PCA versions saved so training.py can use either
#   ✅ FirstPurchaseDaysAgo → DROPPED (constant=374 for all rows, pure noise)
#   ✅ SatisfactionScore → DROPPED (non-significant, p=0.365, corr=+0.014)
#   ✅ Target encoding smoothed (k=10) to prevent rare-country leakage
#      (Bahrain/Canada/Brazil etc. had n=1 → churn=1.0 without smoothing)
#   ✅ RegistrationDate parser fixed (dayfirst=False) to suppress UserWarning
#      caused by mixed US/UK/ISO date formats in raw data
#   ✅ Target encoding groupby hardened (.values strips index) → index-safe
#
# FIXES in this version (v3):
#   ✅ Log-transforms extended to all high-skew columns detected in EDA
#      (|skew| > 1): CancelledTransactions, SupportTicketsCount, ReturnRatio,
#      AvgDaysBetweenPurchases, ZeroPriceCount, UniqueCountries,
#      AvgProductsPerTransaction, MonetaryAvg, WeekendPurchaseRatio,
#      MonetaryMax — in addition to the original 3 (MonetaryTotal,
#      Frequency, AvgBasketValue)
#   ✅ RegYear added back — customer cohort signal (year of registration)
#      has predictive value in a multi-year snapshot dataset
#   ✅ AvgBasketValue computed BEFORE log-transforms to ensure consistency
#      (was already the case but now explicitly ordered and commented)
#   ✅ DisengagementScore formula hardened — ReturnRatio and
#      CancelledTransactions are each standardised before combination
#      instead of using the arbitrary /10 divisor
#   ✅ WeekendPreference OHE coverage confirmed — verified it is object dtype
#      so pd.get_dummies handles it correctly
#   ✅ IsPrivateIP and IPClass value/signal noted in comments —
#      these are low-signal features; flag for potential drop post-training
#      if feature importance is negligible
# ==========================================

import pandas as pd
import numpy as np
import joblib
import ipaddress

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.decomposition    import PCA


# ==========================================
# 1️⃣ LOAD RAW DATA
# ==========================================

print("Loading raw dataset...")
df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
print(f"  Raw shape : {df.shape}")


# ==========================================
# 2️⃣ PARSE RegistrationDate → features
# ==========================================
# FIX v3: RegYear is now KEPT — year of registration encodes customer cohort
# (customers registered in 2009 vs 2015 have very different tenure profiles).
# In a multi-year snapshot dataset, RegYear carries genuine predictive signal.
# format="mixed" + dayfirst=False handles all three formats cleanly:
#   UK  "12/09/2009", ISO "2010-10-04", US "10/18/2010"
# Failures become NaT (coerce) → filled with median post-split.
# ==========================================

print("Parsing RegistrationDate...")

if "RegistrationDate" in df.columns:
    df["RegistrationDate"] = pd.to_datetime(
        df["RegistrationDate"],
        format="mixed",
        dayfirst=False,
        errors="coerce"
    )

    df["RegYear"]    = df["RegistrationDate"].dt.year
    df["RegMonth"]   = df["RegistrationDate"].dt.month
    df["RegDay"]     = df["RegistrationDate"].dt.day
    df["RegWeekday"] = df["RegistrationDate"].dt.weekday   # 0=Mon, 6=Sun

    n_nat = df["RegistrationDate"].isna().sum()
    if n_nat > 0:
        print(f"  ⚠️  {n_nat} unparseable dates → NaT (will be imputed post-split)")

    print(f"  ✅ Extracted: RegYear, RegMonth, RegDay, RegWeekday")

    df = df.drop(columns=["RegistrationDate"])
    print(f"  ✅ RegistrationDate dropped after extraction")


# ==========================================
# 3️⃣ PARSE LastLoginIP → features
# ==========================================
# Features extracted:
#   IsPrivateIP  — 1 if RFC1918 private address (10.x, 192.168.x, 172.16-31.x)
#   IPClass      — Class A (1-126), B (128-191), C (192-223), Other
# Note: these are low-signal features in a retail context.
# Retain for now; drop post-training if feature importance is negligible.
# ==========================================

print("Engineering LastLoginIP features...")

def _is_private_ip(ip_str) -> int:
    """Returns 1 if IP is RFC1918 private, 0 otherwise, -1 if unparseable."""
    try:
        return int(ipaddress.ip_address(str(ip_str)).is_private)
    except Exception:
        return -1

def _ip_class(ip_str) -> int:
    """
    Returns IP class as integer:
      1 = Class A (1-126),  2 = Class B (128-191),
      3 = Class C (192-223), 0 = Other/Unknown
    """
    try:
        first_octet = int(str(ip_str).split(".")[0])
        if 1 <= first_octet <= 126:
            return 1
        elif 128 <= first_octet <= 191:
            return 2
        elif 192 <= first_octet <= 223:
            return 3
        return 0
    except Exception:
        return 0

if "LastLoginIP" in df.columns:
    df["IsPrivateIP"] = df["LastLoginIP"].apply(_is_private_ip)
    df["IPClass"]     = df["LastLoginIP"].apply(_ip_class)

    print(f"  ✅ IsPrivateIP: {df['IsPrivateIP'].value_counts().to_dict()}")
    print(f"  ✅ IPClass: {df['IPClass'].value_counts().sort_index().to_dict()}")

    df = df.drop(columns=["LastLoginIP"])
    print(f"  ✅ LastLoginIP dropped after extraction")


# ==========================================
# 4️⃣ FEATURE SELECTION (SURGICAL DROPS)
# ==========================================

print("\nDropping redundant / useless / leaky columns...")

columns_to_drop = [
    # --- Constant / redundant / ID ---
    "NewsletterSubscribed",
    "UniqueInvoices",
    "TotalTransactions",
    "UniqueDescriptions",
    "NegativeQuantityCount",
    "CustomerID",
    # --- Leaky: encode Churn directly ---
    "ChurnRiskCategory",
    "RFMSegment",
    "LoyaltyLevel",
    "CustomerType",           # "Perdu" = churned in French
    # --- Leaky: circular (Churn defined from Recency) ---
    "Recency",                # corr=0.859 with Churn
    # --- Constant across all rows: FirstPurchaseDaysAgo=374 for every customer ---
    "FirstPurchaseDaysAgo",
    # --- Non-significant: p=0.365, corr=+0.014 ---
    "SatisfactionScore",
    # --- Multicollinearity & Redundancy Drops ---
    "AvgLinesPerInvoice",    # r=0.96 with AvgProductsPerTransaction
    "TotalQuantity",         # r=0.92 with MonetaryTotal
    "MonetaryStd",           # Highly entangled cluster
    "MonetaryMin",           # Highly entangled cluster
    "MinQuantity",           # Highly entangled cluster
    "MaxQuantity",           # Highly entangled cluster (Keeping MonetaryMax)
    # PreferredMonth: temporal leakage — encodes time-since-last-activity
    # rather than genuine behaviour. Surrogate tree confirmed: PreferredMonth
    # ≤ 8.5 → 91.4% churn on RIGHT branch (purchased mainly after August
    # = churned by cutoff snapshot date).
    "PreferredMonth",
    # FavoriteSeason: CONFIRMED leakage (diagnostic 2026-05-06).
    # 100% match with season derived from PreferredMonth:
    #   Automne=mois 9/10/11 -> 1.9% churn (active customers near snapshot)
    #   Hiver=mois 12/1/2    -> 51.8% churn
    #   Printemps=mois 3/4/5 -> 61.3% churn
    #   Ete=mois 6/7/8       -> 50.4% churn
    # FavoriteSeason is a bijective recode of PreferredMonth.
    # Both encode snapshot date of last purchase, not genuine preference.
    # Was responsible for 58% of model feature importance -> artificial signal.
    "FavoriteSeason",
    # SpendingCategory: CONFIRMED leakage (diagnostic 2026-05-06).
    # Perfect clean-cut discretisation of MonetaryTotal with zero overlap:
    #   Low    : MonetaryTotal < £100
    #   Medium : £100 <= MonetaryTotal < £500
    #   High   : £500 <= MonetaryTotal < £2001
    #   VIP    : MonetaryTotal >= £2001
    # Responsible for 70% of regression feature importance -> pure leakage.
    # Also correlates directly with Churn (Low=66%, Medium=51%, High=25%, VIP=7%)
    # because MonetaryTotal itself is a strong churn predictor.
    # Both classification and regression models must not use SpendingCategory.
    "SpendingCategory",
    # BasketSizeCategory: same risk — likely derived from AvgBasketValue or
    # MonetaryTotal/Frequency. Drop to be safe; it encodes monetary behaviour
    # already captured by the raw numeric columns.
    "BasketSizeCategory",
]

df = df.drop(columns=columns_to_drop, errors="ignore")
print(f"  Shape after drops : {df.shape}")


# ==========================================
# 5️⃣ SENTINEL VALUE CLEANUP
# ==========================================

print("Replacing sentinel values with NaN...")

if "SupportTicketsCount" in df.columns:
    df["SupportTicketsCount"] = df["SupportTicketsCount"].replace([-1, 999], np.nan)
    print("  ✅ SupportTicketsCount sentinels → NaN")

# IsPrivateIP -1 (unparseable) → NaN
if "IsPrivateIP" in df.columns:
    df["IsPrivateIP"] = df["IsPrivateIP"].replace(-1, np.nan)


# ==========================================
# 6️⃣ ORDINAL ENCODING
# ==========================================

print("Applying ordinal encoding...")

# SpendingCategory and BasketSizeCategory REMOVED from ordinal encoding —
# both dropped as leaky (confirmed 2026-05-06 diagnostic).
# SpendingCategory is a clean-cut discretisation of MonetaryTotal.
# BasketSizeCategory encodes AvgBasketValue or MonetaryTotal/Frequency.

if "AgeCategory" in df.columns:
    df["AgeCategory"] = df["AgeCategory"].map(
        {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5}
    )   # "Inconnu" → NaN, imputed post-split

if "PreferredTimeOfDay" in df.columns:
    df["PreferredTimeOfDay"] = df["PreferredTimeOfDay"].map(
        {"Nuit": 0, "Matin": 1, "Midi": 2, "Après-midi": 3, "Soir": 4}
    )

print("  ✅ Ordinal encoding complete")


# ==========================================
# 7️⃣ ONE-HOT ENCODING (Non-ordinal, excluding Country)
# ==========================================
# Country excluded here — gets smoothed Target Encoding post-split (step 10)
# because it has 37+ unique values with many rare categories.
# WeekendPreference is confirmed object dtype → handled correctly by get_dummies.
# ==========================================

print("Applying one-hot encoding (Country excluded → Target Encoding)...")

country_col = None
if "Country" in df.columns:
    country_col = df["Country"].copy()
    df = df.drop(columns=["Country"])

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if "Churn" in categorical_cols:
    categorical_cols.remove("Churn")

if categorical_cols:
    print(f"  OHE columns: {categorical_cols}")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
else:
    print("  No remaining categorical columns to OHE.")

if country_col is not None:
    df["Country"] = country_col

print(f"  Shape after OHE (before target encoding): {df.shape}")


# ==========================================
# 8️⃣ SPLIT FEATURES / TARGET
# ==========================================

print("\nSplitting target variable...")

if "Churn" not in df.columns:
    raise ValueError("Target column 'Churn' not found.")

X = df.drop("Churn", axis=1)
y = df["Churn"]

print(f"  Features  : {X.shape[1]} columns")
print(f"  Churn rate: {y.mean():.2%}")


# ==========================================
# 9️⃣ TRAIN / TEST SPLIT — EARLY (80/20 stratified)
# ==========================================

print("Performing train/test split (80/20 stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y,
)

print(f"  X_train : {X_train.shape} | X_test : {X_test.shape}")
print(f"  Churn rate — train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")


# ==========================================
# 🔟 TARGET ENCODING — Country (post-split, smoothed)
# ==========================================
# FIX: Hardened groupby — uses .values to strip index, preventing
# silent bugs when indices are non-contiguous or reset elsewhere.
# Smoothing (k=10) prevents rare-country memorization.
# Formula: smoothed = (n * country_mean + k * global_mean) / (n + k)
# ==========================================

print("Applying Target Encoding to Country (fit on train only, smoothed k=10)...")

if "Country" in X_train.columns:
    _train_df = pd.DataFrame({
        "Country": X_train["Country"].values,
        "Churn":   y_train.values
    })
    global_churn_rate = y_train.mean()

    country_stats = _train_df.groupby("Country")["Churn"].agg(["mean", "count"])

    k = 10
    country_stats["smoothed"] = (
        (country_stats["count"] * country_stats["mean"] + k * global_churn_rate)
        / (country_stats["count"] + k)
    )

    country_churn_map = country_stats["smoothed"]

    print(f"  Top 5 country churn rates after smoothing (k={k}):")
    for c, r in country_churn_map.sort_values(ascending=False).head(5).items():
        n = int(country_stats.loc[c, "count"])
        print(f"    {c:<25} {r:.3f}  (n={n})")

    X_train["Country_TargetEnc"] = (
        X_train["Country"].map(country_churn_map).fillna(global_churn_rate)
    )
    X_test["Country_TargetEnc"] = (
        X_test["Country"].map(country_churn_map).fillna(global_churn_rate)
    )

    X_train = X_train.drop(columns=["Country"])
    X_test  = X_test.drop(columns=["Country"])

    print(f"  ✅ Country → Country_TargetEnc (smoothed, k={k})")
    print(f"     Global churn fallback for unknown countries: {global_churn_rate:.3f}")

    country_encoding = {
        "country_churn_map": country_churn_map.to_dict(),
        "global_churn_rate": global_churn_rate,
        "smoothing_k":       k,
    }
else:
    country_encoding = {}
    print("  Country column not found — skipping target encoding")


# ==========================================
# 1️⃣1️⃣ MISSING VALUES (Post-split — no leakage)
# ==========================================

print("Handling missing values (fit on train only)...")

# Age
if "Age" in X_train.columns:
    X_train["Age_IsMissing"] = X_train["Age"].isna().astype(int)
    X_test["Age_IsMissing"]  = X_test["Age"].isna().astype(int)
    age_median = X_train["Age"].median()
    X_train["Age"] = X_train["Age"].fillna(age_median)
    X_test["Age"]  = X_test["Age"].fillna(age_median)
    print(f"  Age median (train)          : {age_median}")
else:
    age_median = None

# AgeCategory
if "AgeCategory" in X_train.columns:
    age_cat_median = X_train["AgeCategory"].median()
    X_train["AgeCategory"] = X_train["AgeCategory"].fillna(age_cat_median)
    X_test["AgeCategory"]  = X_test["AgeCategory"].fillna(age_cat_median)
    print(f"  AgeCategory median (train)  : {age_cat_median}")
else:
    age_cat_median = None

# AvgDaysBetweenPurchases — 0 is semantically correct (single-purchase customers)
if "AvgDaysBetweenPurchases" in X_train.columns:
    X_train["AvgDaysBetweenPurchases"] = X_train["AvgDaysBetweenPurchases"].fillna(0)
    X_test["AvgDaysBetweenPurchases"]  = X_test["AvgDaysBetweenPurchases"].fillna(0)

# SupportTicketsCount
if "SupportTicketsCount" in X_train.columns:
    support_median = X_train["SupportTicketsCount"].median()
    X_train["SupportTicketsCount"] = X_train["SupportTicketsCount"].fillna(support_median)
    X_test["SupportTicketsCount"]  = X_test["SupportTicketsCount"].fillna(support_median)
    print(f"  SupportTicketsCount median  : {support_median}")
else:
    support_median = None

# RegYear / RegMonth / RegDay / RegWeekday — impute with train median
for col in ["RegYear", "RegMonth", "RegDay", "RegWeekday"]:
    if col in X_train.columns:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col]  = X_test[col].fillna(med)

# IsPrivateIP
if "IsPrivateIP" in X_train.columns:
    ip_mode = X_train["IsPrivateIP"].mode()[0]
    X_train["IsPrivateIP"] = X_train["IsPrivateIP"].fillna(ip_mode)
    X_test["IsPrivateIP"]  = X_test["IsPrivateIP"].fillna(ip_mode)


# ==========================================
# 1️⃣2️⃣ FEATURE ENGINEERING (Post-split)
# ==========================================
# ORDER MATTERS:
#   AvgBasketValue must be computed on raw MonetaryTotal BEFORE
#   log-transforms are applied (step 13). This ensures the ratio
#   is computed on the original scale, then log-transformed consistently.
# ==========================================

print("Engineering new features...")

# AvgBasketValue — computed BEFORE log-transforms (see note above)
if "MonetaryTotal" in X_train.columns and "Frequency" in X_train.columns:
    X_train["AvgBasketValue"] = X_train["MonetaryTotal"] / (X_train["Frequency"] + 1)
    X_test["AvgBasketValue"]  = X_test["MonetaryTotal"]  / (X_test["Frequency"]  + 1)

# EngagementScore (Recency-free)
if "Frequency" in X_train.columns and "CustomerTenureDays" in X_train.columns:
    X_train["EngagementScore"] = X_train["Frequency"] / (X_train["CustomerTenureDays"] + 1)
    X_test["EngagementScore"]  = X_test["Frequency"]  / (X_test["CustomerTenureDays"]  + 1)

# DisengagementScore — FIX v3: normalise each component before combining
# Previous formula used arbitrary /10 divisor for CancelledTransactions,
# which caused skewed CancelledTransactions to dominate the score.
# Fix: use z-score normalisation of each component on train statistics.
if "ReturnRatio" in X_train.columns and "CancelledTransactions" in X_train.columns:
    rr_mean  = X_train["ReturnRatio"].mean()
    rr_std   = X_train["ReturnRatio"].std() + 1e-8
    ct_mean  = X_train["CancelledTransactions"].mean()
    ct_std   = X_train["CancelledTransactions"].std() + 1e-8

    X_train["DisengagementScore"] = (
        (X_train["ReturnRatio"] - rr_mean) / rr_std
        + (X_train["CancelledTransactions"] - ct_mean) / ct_std
    )
    X_test["DisengagementScore"] = (
        (X_test["ReturnRatio"] - rr_mean) / rr_std
        + (X_test["CancelledTransactions"] - ct_mean) / ct_std
    )

# RevenueIndex (global_mean from train only)
if "MonetaryTotal" in X_train.columns:
    global_mean = X_train["MonetaryTotal"].mean()
    X_train["RevenueIndex"] = X_train["MonetaryTotal"] / (global_mean + 1)
    X_test["RevenueIndex"]  = X_test["MonetaryTotal"]  / (global_mean + 1)
    print(f"  RevenueIndex global_mean (train) : {global_mean:.4f}")
else:
    global_mean = None

print(f"  Shape after feature engineering — train: {X_train.shape} | test: {X_test.shape}")


# ==========================================
# 1️⃣3️⃣ LOG TRANSFORMATIONS
# ==========================================
# FIX v3: Extended to ALL high-skew columns detected in EDA (|skew| > 1).
# Previous version only log-transformed 3 columns, leaving 20+ skewed
# features that StandardScaler cannot correct.
# sign * log1p(|x|) handles negative values safely (e.g. MonetaryMin).
# ==========================================

print("Applying log transformations...")

log_cols = [
    # Original 3
    "MonetaryTotal",
    "Frequency",
    "AvgBasketValue",
    # FIX v3: Added from EDA high-skew list
    "CancelledTransactions",
    "SupportTicketsCount",
    "ReturnRatio",
    "AvgDaysBetweenPurchases",
    "ZeroPriceCount",
    "UniqueCountries",
    "AvgProductsPerTransaction",
    "MonetaryAvg",
    "WeekendPurchaseRatio",
    "MonetaryMax",
    "EngagementScore",
    "RevenueIndex",
    # AvgQuantityPerTransaction: max=12540 vs Q75=14 (317x above 3xIQR fence).
    # Outliers dominate tree splits → 27% artificial regression importance
    # despite corr=0.031 with MonetaryTotal. Log-transform neutralises this.
    "AvgQuantityPerTransaction",
]

for col in log_cols:
    if col in X_train.columns:
        X_train[col] = np.sign(X_train[col]) * np.log1p(np.abs(X_train[col]))
        X_test[col]  = np.sign(X_test[col])  * np.log1p(np.abs(X_test[col]))
        print(f"  ✅ log1p → {col}")
    else:
        print(f"  ⚠️  {col} not found — skipped")


# ==========================================
# 1️⃣4️⃣ NaN SAFETY NET
# ==========================================

print("NaN safety net...")

numeric_cols  = X_train.select_dtypes(include=["int64", "float64"]).columns
train_medians = X_train[numeric_cols].median()

nan_count = X_train.isnull().sum().sum() + X_test.isnull().sum().sum()
if nan_count > 0:
    print(f"  ⚠️  {nan_count} NaNs remaining — filling with train medians")
    X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
    X_test[numeric_cols]  = X_test[numeric_cols].fillna(train_medians)
    bool_cols = X_train.select_dtypes(include=["bool"]).columns
    X_train[bool_cols] = X_train[bool_cols].fillna(False)
    X_test[bool_cols]  = X_test[bool_cols].fillna(False)
else:
    print("  ✅ No remaining NaNs.")


# ==========================================
# 1️⃣5️⃣ SCALING NUMERIC FEATURES
# ==========================================

print("Scaling numeric features (StandardScaler)...")

X_train = X_train.astype({c: int for c in X_train.select_dtypes("bool").columns})
X_test  = X_test.astype({c: int for c in X_test.select_dtypes("bool").columns})

numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

print(f"  Scaled {len(numeric_cols)} numeric columns")
print(f"  Final shape — train: {X_train.shape} | test: {X_test.shape}")


# ==========================================
# 1️⃣6️⃣ PCA — DIMENSIONALITY REDUCTION
# ==========================================
# Produces a second dataset variant: X_train_pca / X_test_pca
# training.py will compare models on both raw-scaled and PCA-reduced features.
# PCA fit on X_train ONLY → transform applied to X_test.
# SMOTE / class_weight handled in training.py.
# ==========================================

print("\nRunning PCA (95% variance threshold)...")

pca_numeric = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

pca = PCA(n_components=0.95, random_state=42)
X_train_pca_arr = pca.fit_transform(X_train[pca_numeric])
X_test_pca_arr  = pca.transform(X_test[pca_numeric])

n_components = pca.n_components_
variance_ret = pca.explained_variance_ratio_.sum()

print(f"  Components selected : {n_components}  (from {len(pca_numeric)} numeric features)")
print(f"  Variance retained   : {variance_ret:.2%}")

pca_cols    = [f"PC{i+1}" for i in range(n_components)]
X_train_pca = pd.DataFrame(X_train_pca_arr, columns=pca_cols, index=X_train.index)
X_test_pca  = pd.DataFrame(X_test_pca_arr,  columns=pca_cols, index=X_test.index)

print(f"  X_train_pca : {X_train_pca.shape}")
print(f"  X_test_pca  : {X_test_pca.shape}")

# PCA plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os
    os.makedirs("reports", exist_ok=True)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar)+1), cumvar, marker="o", markersize=3,
            color="#2c5f8a", linewidth=1.5)
    ax.axhline(0.95, color="#c0392b", linestyle="--", linewidth=1, label="95% threshold")
    ax.axvline(n_components, color="#e67e22", linestyle="--", linewidth=1,
               label=f"{n_components} components")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA — Cumulative Explained Variance")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("reports/pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ PCA variance plot → reports/pca_variance.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0: "#2c5f8a", 1: "#c0392b"}
    labels = {0: "Fidèle", 1: "Churn"}
    for cls in [0, 1]:
        mask = y_train == cls
        ax.scatter(
            X_train_pca.loc[mask, "PC1"],
            X_train_pca.loc[mask, "PC2"],
            c=colors[cls], label=labels[cls],
            alpha=0.4, s=10, edgecolors="none"
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D — Churn vs Fidèle")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/pca_2d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ PCA 2D scatter   → reports/pca_2d.png")
except Exception as e:
    print(f"  ⚠️  PCA plots skipped: {e}")


# ==========================================
# ⛔ NO SMOTE HERE — lives in training.py (class_weight="balanced")
# ==========================================

print(f"\n  Class distribution (NO SMOTE):")
print(f"    Train → Fidèle: {(y_train==0).sum()} | Churn: {(y_train==1).sum()} ({y_train.mean():.2%})")
print(f"    Test  → Fidèle: {(y_test==0).sum()}  | Churn: {(y_test==1).sum()}  ({y_test.mean():.2%})")


# ==========================================
# 1️⃣7️⃣ SAVE ALL DATASETS + ARTIFACTS
# ==========================================

print("\nSaving datasets and artifacts...")

import os
os.makedirs("data/train_test", exist_ok=True)
os.makedirs("data/processed",  exist_ok=True)
os.makedirs("models",          exist_ok=True)

# Raw scaled datasets (for training without PCA)
X_train.to_csv("data/train_test/X_train.csv",         index=False)
X_test.to_csv("data/train_test/X_test.csv",           index=False)
y_train.to_csv("data/train_test/y_train.csv",         index=False)
y_test.to_csv("data/train_test/y_test.csv",           index=False)

# PCA-reduced datasets (for training with PCA)
X_train_pca.to_csv("data/train_test/X_train_pca.csv", index=False)
X_test_pca.to_csv("data/train_test/X_test_pca.csv",   index=False)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save PCA object
joblib.dump(pca, "models/pca.pkl")

# Save all stats needed for production inference
imputation_stats = {
    "age_median"          : age_median,
    "age_cat_median"      : age_cat_median,
    "support_median"      : support_median,
    "satisfaction_median" : None,          # SatisfactionScore dropped — not needed
    "global_mean_monetary": global_mean,
    "train_medians"       : train_medians.to_dict(),
    "country_encoding"    : country_encoding,
    "pca_numeric_cols"    : pca_numeric,
    "pca_n_components"    : n_components,
    # DisengagementScore normalisation stats (needed for inference)
    "disengagement_rr_mean" : rr_mean  if "ReturnRatio" in X_train.columns else None,
    "disengagement_rr_std"  : rr_std   if "ReturnRatio" in X_train.columns else None,
    "disengagement_ct_mean" : ct_mean  if "CancelledTransactions" in X_train.columns else None,
    "disengagement_ct_std"  : ct_std   if "CancelledTransactions" in X_train.columns else None,
}
joblib.dump(imputation_stats, "models/imputation_stats.pkl")

print("✅ Preprocessing complete.")
print(f"   X_train     : {X_train.shape}     → data/train_test/X_train.csv")
print(f"   X_train_pca : {X_train_pca.shape} → data/train_test/X_train_pca.csv")
print(f"   X_test      : {X_test.shape}      → data/train_test/X_test.csv")
print(f"   X_test_pca  : {X_test_pca.shape}  → data/train_test/X_test_pca.csv")
print(f"   Churn rate train : {y_train.mean():.2%}  |  test : {y_test.mean():.2%}")
print("   Artifacts → models/scaler.pkl | models/pca.pkl | models/imputation_stats.pkl")