# ==========================================
# RETAIL ML PROJECT - PREDICTION SCRIPT
# ==========================================

import pandas as pd
import numpy as np
import joblib


# ==========================================
# 1️⃣ LOAD SAVED MODEL & SCALER
# ==========================================

print("Loading model and scaler...")

model  = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("  ✅ churn_model.pkl loaded")
print("  ✅ scaler.pkl loaded")


# ==========================================
# 2️⃣ LOAD REFERENCE COLUMNS
# ==========================================

print("Loading reference columns from training data...")

X_train_ref      = pd.read_csv("data/train_test/X_train.csv", nrows=1)
expected_columns = X_train_ref.columns.tolist()

# These are the exact columns the scaler was fit on
# (scaler.feature_names_in_ stores them automatically)
scaler_columns = list(scaler.feature_names_in_)

print(f"  ✅ {len(expected_columns)} expected features loaded")
print(f"  ✅ Scaler trained on {len(scaler_columns)} numeric columns")


# ==========================================
# 3️⃣ PREDICTION FUNCTION
# ==========================================

def predict_churn(input_dict: dict) -> dict:
    """
    Takes a dictionary of raw customer features,
    preprocesses them, and returns a churn prediction.

    Parameters:
        input_dict (dict): Raw customer data as key-value pairs

    Returns:
        dict: {
            "prediction"  : 0 or 1,
            "label"       : "Fidèle" or "Churn",
            "probability" : float (probability of churn),
            "risk_level"  : "Faible" / "Moyen" / "Élevé" / "Critique"
        }
    """

    # --- Convert input to DataFrame ---
    df = pd.DataFrame([input_dict])

    # --- One-hot encode categorical columns ---
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Align columns to match training data exactly ---
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]

    # --- Scale ONLY the columns the scaler was trained on ---
    # Use only scaler_columns that exist in df (safe intersection)
    cols_to_scale = [c for c in scaler_columns if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # --- Predict ---
    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # --- Risk level ---
    if probability < 0.25:
        risk = "Faible"
    elif probability < 0.50:
        risk = "Moyen"
    elif probability < 0.75:
        risk = "Élevé"
    else:
        risk = "Critique"

    return {
        "prediction" : int(prediction),
        "label"      : "Churn" if prediction == 1 else "Fidèle",
        "probability": round(float(probability), 4),
        "risk_level" : risk
    }


# ==========================================
# 4️⃣ BATCH PREDICTION FUNCTION
# ==========================================

def predict_batch(csv_path: str, output_path: str = "reports/predictions.csv"):
    """
    Runs predictions on a full CSV file of customers.

    Parameters:
        csv_path    (str): Path to input CSV file
        output_path (str): Path to save results CSV
    """

    print(f"\nRunning batch prediction on: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"  Loaded {len(df_raw)} customers")

    results = []
    for _, row in df_raw.iterrows():
        result = predict_churn(row.to_dict())
        results.append(result)

    df_results = pd.DataFrame(results)

    if "CustomerID" in df_raw.columns:
        df_results.insert(0, "CustomerID", df_raw["CustomerID"].values)

    df_results.to_csv(output_path, index=False)
    print(f"  ✅ Predictions saved to {output_path}")
    print(f"\n  Churn summary:")
    print(f"    Fidèle : {(df_results['label'] == 'Fidèle').sum()}")
    print(f"    Churn  : {(df_results['label'] == 'Churn').sum()}")

    return df_results


# ==========================================
# 5️⃣ DEMO — Single Customer Prediction
# ==========================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  DEMO — Single Customer Prediction")
    print("="*55)

    sample_customer = {
        "Recency"                 : 15,
        "Frequency"               : 8,
        "MonetaryTotal"           : 450.0,
        "MonetaryAvg"             : 56.25,
        "MonetaryStd"             : 20.0,
        "MonetaryMin"             : 10.0,
        "MonetaryMax"             : 120.0,
        "TotalQuantity"           : 60,
        "AvgQtyPerTransaction"    : 7.5,
        "MinQuantity"             : 1,
        "MaxQuantity"             : 30,
        "CustomerTenureDays"      : 365,
        "FirstPurchaseDaysAgo"    : 400,
        "PreferredDayOfWeek"      : 2,
        "PreferredHour"           : 14,
        "PreferredMonth"          : 11,
        "WeekendRatio"            : 0.25,
        "AvgDaysBetweenPurchases" : 45,
        "UniqueProducts"          : 20,
        "AvgProductsPerTrans"     : 2.5,
        "UniqueCountries"         : 1,
        "CancelledTransactions"   : 0,
        "ReturnRatio"             : 0.0,
        "TotalTransactionLines"   : 80,
        "AvgLinesPerInvoice"      : 10.0,
        "Age"                     : 35,
        "Age_IsMissing"           : 0,
        "SupportTickets"          : 1,
        "Satisfaction"            : 4,
        "SpendingCategory"        : 2,
        "AgeCategory"             : 2,
        "BasketSizeCategory"      : 1,
        "PreferredTimeOfDay"      : 3,
        "CustomerType"            : "Régulier",
        "FavoriteSeason"          : "Hiver",
        "Region"                  : "UK",
        "WeekendPreference"       : "Semaine",
        "ProductDiversity"        : "Modéré",
        "Gender"                  : "M",
        "AccountStatus"           : "Active",
        "Country"                 : "United Kingdom",
    }

    result = predict_churn(sample_customer)

    print(f"\n  Customer Profile:")
    print(f"    Recency    : {sample_customer['Recency']} days since last purchase")
    print(f"    Frequency  : {sample_customer['Frequency']} orders")
    print(f"    Monetary   : £{sample_customer['MonetaryTotal']}")
    print(f"\n  Prediction Result:")
    print(f"    Label       : {result['label']}")
    print(f"    Probability : {result['probability']*100:.1f}% chance of churn")
    print(f"    Risk Level  : {result['risk_level']}")