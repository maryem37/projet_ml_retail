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


# ==========================================
# 2️⃣ FEATURE SELECTION (SURGICAL DROPS)
# ==========================================

print("Dropping redundant / useless columns...")

columns_to_drop = [
    "NewsletterSubscribed",      # Constant feature
    "UniqueInvoices",            # Perfect correlation
    "TotalTransactions",         # Perfect correlation
    "UniqueDescriptions",        # Redundant with UniqueProducts
    "NegativeQuantityCount",     # Redundant with CancelledTransactions
    "RegistrationDate",          # High cardinality / raw date
    "LastLoginIP",               # High cardinality
    "CustomerID"                 # ID column
]

df = df.drop(columns=columns_to_drop, errors="ignore")


# ==========================================
# 3️⃣ MISSING VALUES HANDLING
# ==========================================

print("Handling missing values...")

# ---- AGE (30% missing) ----
if "Age" in df.columns:
    df["Age_IsMissing"] = df["Age"].isna().astype(int)
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

# ---- AvgDaysBetweenPurchases (One-timers) ----
if "AvgDaysBetweenPurchases" in df.columns:
    df["AvgDaysBetweenPurchases"] = df["AvgDaysBetweenPurchases"].fillna(0)


# ==========================================
# 4️⃣ LOG TRANSFORMATIONS (Reduce Skewness)
# ==========================================

print("Applying log transformations...")

if "MonetaryTotal" in df.columns:
    df["MonetaryTotal"] = np.log1p(df["MonetaryTotal"])

if "Frequency" in df.columns:
    df["Frequency"] = np.log1p(df["Frequency"])


# ==========================================
# 5️⃣ ORDINAL ENCODING
# ==========================================

print("Applying ordinal encoding...")

# Example: ChurnRiskCategory
if "ChurnRiskCategory" in df.columns:
    risk_map = {
        "Faible": 0,
        "Moyen": 1,
        "Elevé": 2,
        "Critique": 3
    }
    df["ChurnRiskCategory"] = df["ChurnRiskCategory"].map(risk_map)

# Example: LoyaltyLevel
if "LoyaltyLevel" in df.columns:
    loyalty_map = {
        "Nouveau": 0,
        "Jeune": 1,
        "Etabli": 2,
        "Ancien": 3,
        "Inconnu": 4
    }
    df["LoyaltyLevel"] = df["LoyaltyLevel"].map(loyalty_map)


# ==========================================
# 6️⃣ ONE-HOT ENCODING (Non-ordinal)
# ==========================================

print("Applying one-hot encoding...")

categorical_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# ==========================================
# 7️⃣ SPLIT FEATURES / TARGET
# ==========================================

print("Splitting target variable...")

if "Churn" not in df.columns:
    raise ValueError("Target column 'Churn' not found in dataset.")

X = df.drop("Churn", axis=1)
y = df["Churn"]


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


# ==========================================
# 9️⃣ SCALING NUMERIC FEATURES
# ==========================================

print("Scaling numeric features...")

numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


# ==========================================
# 🔟 SAVE PROCESSED DATA
# ==========================================

print("Saving processed datasets...")

X_train.to_csv("data/train_test/X_train.csv", index=False)
X_test.to_csv("data/train_test/X_test.csv", index=False)
y_train.to_csv("data/train_test/y_train.csv", index=False)
y_test.to_csv("data/train_test/y_test.csv", index=False)

joblib.dump(scaler, "models/scaler.pkl")

print("✅ Preprocessing complete.")
print("Files saved in data/train_test/ and models/")