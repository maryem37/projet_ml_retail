# ==========================================
# RETAIL ML PROJECT - PREPROCESSING SCRIPT
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1️⃣ LOAD RAW DATA
# ==========================================

print("Loading raw dataset...")
df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
print(f"  Raw shape : {df.shape}")


# ==========================================
# 2️⃣ FEATURE SELECTION (SURGICAL DROPS)
# ==========================================

print("Dropping redundant / useless / leaky columns...")

columns_to_drop = [
    "NewsletterSubscribed",      # Constant feature
    "UniqueInvoices",            # Perfect correlation
    "TotalTransactions",         # Perfect correlation
    "UniqueDescriptions",        # Redundant with UniqueProducts
    "NegativeQuantityCount",     # Redundant with CancelledTransactions
    "RegistrationDate",          # High cardinality / raw date
    "LastLoginIP",               # High cardinality
    "CustomerID",                # ID column
    # --- Leaky columns (derived from Churn target) ---
    "ChurnRiskCategory",         # Directly encodes churn risk → leakage
    "RFMSegment",                # Built from RFM, correlates perfectly → leakage
    "LoyaltyLevel",              # May encode churn indirectly → leakage
]

df = df.drop(columns=columns_to_drop, errors="ignore")
print(f"  Shape after drops : {df.shape}")


# ==========================================
# 3️⃣ MISSING VALUES HANDLING
# ==========================================

print("Handling missing values...")

# ---- AGE (30% missing) ----
if "Age" in df.columns:
    df["Age_IsMissing"] = df["Age"].isna().astype(int)
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

# ---- AvgDaysBetweenPurchases (One-timers have NaN) ----
if "AvgDaysBetweenPurchases" in df.columns:
    df["AvgDaysBetweenPurchases"] = df["AvgDaysBetweenPurchases"].fillna(0)

# ---- SupportTickets: aberrant values (-1, 999) → NaN → median ----
if "SupportTickets" in df.columns:
    df["SupportTickets"] = df["SupportTickets"].replace([-1, 999], np.nan)
    df["SupportTickets"] = df["SupportTickets"].fillna(df["SupportTickets"].median())

# ---- Satisfaction: aberrant values (-1, 99) → NaN → median ----
if "Satisfaction" in df.columns:
    df["Satisfaction"] = df["Satisfaction"].replace([-1, 99], np.nan)
    df["Satisfaction"] = df["Satisfaction"].fillna(df["Satisfaction"].median())


# ==========================================
# 4️⃣ LOG TRANSFORMATIONS (Reduce Skewness)
# ==========================================

print("Applying log transformations...")

# Safe log1p that handles negative values
if "MonetaryTotal" in df.columns:
    df["MonetaryTotal"] = np.sign(df["MonetaryTotal"]) * np.log1p(np.abs(df["MonetaryTotal"]))

if "Frequency" in df.columns:
    df["Frequency"] = np.log1p(df["Frequency"])

if "TotalQuantity" in df.columns:
    df["TotalQuantity"] = np.sign(df["TotalQuantity"]) * np.log1p(np.abs(df["TotalQuantity"]))


# ==========================================
# 5️⃣ ORDINAL ENCODING
# ==========================================

print("Applying ordinal encoding...")

if "SpendingCategory" in df.columns:
    spending_map = {"Low": 0, "Medium": 1, "High": 2, "VIP": 3}
    df["SpendingCategory"] = df["SpendingCategory"].map(spending_map)

if "AgeCategory" in df.columns:
    age_cat_map = {
        "18-24": 0, "25-34": 1, "35-44": 2,
        "45-54": 3, "55-64": 4, "65+": 5, "Inconnu": 6
    }
    df["AgeCategory"] = df["AgeCategory"].map(age_cat_map)

if "BasketSize" in df.columns:
    basket_map = {"Petit": 0, "Moyen": 1, "Grand": 2, "Inconnu": 3}
    df["BasketSize"] = df["BasketSize"].map(basket_map)

if "PreferredTime" in df.columns:
    time_map = {
        "Nuit": 0, "Matin": 1, "Midi": 2,
        "Après-midi": 3, "Soir": 4
    }
    df["PreferredTime"] = df["PreferredTime"].map(time_map)


# ==========================================
# 6️⃣ ONE-HOT ENCODING (Non-ordinal)
# ==========================================

print("Applying one-hot encoding...")

categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()

if categorical_cols:
    print(f"  Encoding columns: {categorical_cols}")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
else:
    print("  No remaining categorical columns to encode.")


# ==========================================
# 7️⃣ SPLIT FEATURES / TARGET
# ==========================================

print("Splitting target variable...")

if "Churn" not in df.columns:
    raise ValueError("Target column 'Churn' not found in dataset.")

X = df.drop("Churn", axis=1)
y = df["Churn"]

print(f"  Features : {X.shape[1]} columns")
print(f"  Churn rate : {y.mean():.2%}")


# ==========================================
# 8️⃣ TRAIN / TEST SPLIT
# ==========================================

print("Performing train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"  X_train : {X_train.shape} | X_test : {X_test.shape}")


# ==========================================
# 9️⃣ NaN SAFETY NET (Before Scaling)
# ==========================================

print("Checking and filling any remaining NaN values...")

nan_cols = X_train.isnull().sum()
nan_cols = nan_cols[nan_cols > 0]

if len(nan_cols) > 0:
    print(f"  ⚠️  Remaining NaNs found:\n{nan_cols}")
else:
    print("  ✅ No remaining NaNs detected.")

# Numeric → fill with train median (no data leakage)
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
train_medians = X_train[numeric_cols].median()
X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
X_test[numeric_cols]  = X_test[numeric_cols].fillna(train_medians)

# Boolean / other → fill with 0
other_cols = X_train.columns.difference(numeric_cols)
X_train[other_cols] = X_train[other_cols].fillna(0)
X_test[other_cols]  = X_test[other_cols].fillna(0)


# ==========================================
# 🔟 SCALING NUMERIC FEATURES
# ==========================================

print("Scaling numeric features...")

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])


# ==========================================
# 1️⃣1️⃣ SAVE PROCESSED DATA
# ==========================================

print("Saving processed datasets...")

X_train.to_csv("data/train_test/X_train.csv", index=False)
X_test.to_csv("data/train_test/X_test.csv", index=False)
y_train.to_csv("data/train_test/y_train.csv", index=False)
y_test.to_csv("data/train_test/y_test.csv", index=False)

joblib.dump(scaler, "models/scaler.pkl")

print("✅ Preprocessing complete.")
print(f"   X_train : {X_train.shape}")
print(f"   X_test  : {X_test.shape}")
print("   Files saved in data/train_test/ and models/")