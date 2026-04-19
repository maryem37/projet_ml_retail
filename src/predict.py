# ==========================================
# RETAIL ML PROJECT - PREDICTION SCRIPT
# ==========================================
# UPDATED to match preprocessing_v3.py:
#   ✅ RegistrationDate parsing → RegYear/RegMonth/RegDay/RegWeekday
#   ✅ LastLoginIP → IsPrivateIP + IPClass (not dropped raw)
#   ✅ Country → Target Encoding (not OHE)
#   ✅ PCA applied if training used PCA (auto-detected from threshold.pkl)
#   ✅ Full calibrated pipeline used for predict_proba
#   ✅ Tuned threshold applied
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os
import ipaddress

os.makedirs("reports", exist_ok=True)


# ==========================================
# 1️⃣ LOAD ARTIFACTS
# ==========================================

print("Loading model artifacts...")

pipeline         = joblib.load("models/churn_model.pkl")  # CalibratedClassifierCV
scaler           = joblib.load("models/scaler.pkl")
pca              = joblib.load("models/pca.pkl")
imputation_stats = joblib.load("models/imputation_stats.pkl")
threshold_data   = joblib.load("models/threshold.pkl")

THRESHOLD        = threshold_data["threshold"]
USE_PCA          = threshold_data.get("use_pca", False)
FEATURE_NOTE     = threshold_data.get("feature_set", "Raw scaled features")
TUNER_USED       = threshold_data.get("tuner_used", "GridSearchCV")

country_encoding  = imputation_stats.get("country_encoding", {})
country_churn_map = country_encoding.get("country_churn_map", {})
global_churn_rate = country_encoding.get("global_churn_rate", 0.33)
pca_numeric_cols  = imputation_stats.get("pca_numeric_cols", [])

print(f"  ✅ churn_model.pkl     loaded  (CalibratedClassifierCV)")
print(f"  ✅ scaler.pkl          loaded")
print(f"  ✅ pca.pkl             loaded")
print(f"  ✅ imputation_stats.pkl loaded")
print(f"  ✅ threshold.pkl       loaded  (threshold={THRESHOLD:.3f}, PCA={USE_PCA})")
print(f"     Trained with : {FEATURE_NOTE} | Tuner: {TUNER_USED}")


# ==========================================
# 2️⃣ LOAD REFERENCE COLUMNS
# ==========================================

print("Loading reference columns...")

ref_file    = "data/train_test/X_train_pca.csv" if USE_PCA else "data/train_test/X_train.csv"
X_ref       = pd.read_csv(ref_file, nrows=1)
expected_columns = X_ref.columns.tolist()
scaler_columns   = list(scaler.feature_names_in_)

print(f"  ✅ {len(expected_columns)} expected features loaded")
print(f"  ✅ Using: {'PCA features' if USE_PCA else 'Raw scaled features'}")


# ==========================================
# 3️⃣ PREPROCESSING HELPERS
# ==========================================

def _parse_registration_date(df: pd.DataFrame) -> pd.DataFrame:
    """Extract RegYear/RegMonth/RegDay/RegWeekday from RegistrationDate."""
    if "RegistrationDate" in df.columns:
        dates = pd.to_datetime(df["RegistrationDate"], dayfirst=True, errors="coerce")
        df["RegYear"]    = dates.dt.year.fillna(2010).astype(int)
        df["RegMonth"]   = dates.dt.month.fillna(6).astype(int)
        df["RegDay"]     = dates.dt.day.fillna(15).astype(int)
        df["RegWeekday"] = dates.dt.weekday.fillna(2).astype(int)
        df = df.drop(columns=["RegistrationDate"])
    return df


def _parse_last_login_ip(df: pd.DataFrame) -> pd.DataFrame:
    """Extract IsPrivateIP + IPClass from LastLoginIP."""
    def is_private(ip_str) -> int:
        try:
            return int(ipaddress.ip_address(str(ip_str)).is_private)
        except Exception:
            return 0

    def ip_class(ip_str) -> int:
        try:
            first = int(str(ip_str).split(".")[0])
            if 1 <= first <= 126:   return 1
            if 128 <= first <= 191: return 2
            if 192 <= first <= 223: return 3
            return 0
        except Exception:
            return 0

    if "LastLoginIP" in df.columns:
        df["IsPrivateIP"] = df["LastLoginIP"].apply(is_private)
        df["IPClass"]     = df["LastLoginIP"].apply(ip_class)
        df = df.drop(columns=["LastLoginIP"])
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using train statistics."""
    stats = imputation_stats

    if "Age" in df.columns:
        df["Age_IsMissing"] = df["Age"].isna().astype(int)
        df["Age"]           = df["Age"].fillna(stats.get("age_median", 49))

    if "AgeCategory" in df.columns:
        df["AgeCategory"] = df["AgeCategory"].fillna(stats.get("age_cat_median", 3))

    if "AvgDaysBetweenPurchases" in df.columns:
        df["AvgDaysBetweenPurchases"] = df["AvgDaysBetweenPurchases"].fillna(0)

    if "SupportTicketsCount" in df.columns:
        df["SupportTicketsCount"] = (
            df["SupportTicketsCount"].replace([-1, 999], np.nan)
            .fillna(stats.get("support_median", 2))
        )

   

    return df


def _ordinal_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encoding matching preprocessing_v3.py."""
    if "SpendingCategory" in df.columns and df["SpendingCategory"].dtype == object:
        df["SpendingCategory"] = df["SpendingCategory"].map(
            {"Low": 0, "Medium": 1, "High": 2, "VIP": 3}
        ).fillna(1)

    if "AgeCategory" in df.columns and df["AgeCategory"].dtype == object:
        df["AgeCategory"] = df["AgeCategory"].map(
            {"18-24":0,"25-34":1,"35-44":2,"45-54":3,"55-64":4,"65+":5}
        ).fillna(3)

    if "BasketSizeCategory" in df.columns and df["BasketSizeCategory"].dtype == object:
        df["BasketSizeCategory"] = df["BasketSizeCategory"].map(
            {"Petit":0,"Moyen":1,"Grand":2,"Inconnu":3}
        ).fillna(1)

    if "PreferredTimeOfDay" in df.columns and df["PreferredTimeOfDay"].dtype == object:
        df["PreferredTimeOfDay"] = df["PreferredTimeOfDay"].map(
            {"Nuit":0,"Matin":1,"Midi":2,"Après-midi":3,"Soir":4}
        ).fillna(2)

    return df


def _target_encode_country(df: pd.DataFrame) -> pd.DataFrame:
    """Apply target encoding to Country using train churn map."""
    if "Country" in df.columns:
        df["Country_TargetEnc"] = (
            df["Country"].map(country_churn_map).fillna(global_churn_rate)
        )
        df = df.drop(columns=["Country"])
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering matching preprocessing_v3.py."""
    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)

    if "Frequency" in df.columns and "CustomerTenureDays" in df.columns:
        df["EngagementScore"] = df["Frequency"] / (df["CustomerTenureDays"] + 1)

    if "ReturnRatio" in df.columns and "CancelledTransactions" in df.columns:
        df["DisengagementScore"] = df["ReturnRatio"] + (df["CancelledTransactions"] / 10)

    if "MonetaryTotal" in df.columns:
        gm = imputation_stats.get("global_mean_monetary", 1908.19)
        df["RevenueIndex"] = df["MonetaryTotal"] / (gm + 1)

    return df


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["MonetaryTotal","Frequency","AvgBasketValue"]:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    return df


def _preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full inference preprocessing pipeline.
    Mirrors preprocessing_v3.py exactly.
    """
    df = df.copy()

  # Drop leaky, redundant, and multicollinear columns if present
    columns_to_drop = [
        "Recency", "CustomerType", "ChurnRiskCategory", "RFMSegment", 
        "LoyaltyLevel", "CustomerID", "NewsletterSubscribed", "UniqueInvoices", 
        "TotalTransactions", "UniqueDescriptions", "NegativeQuantityCount", 
        "FirstPurchaseDaysAgo", "SatisfactionScore", "AvgLinesPerInvoice", 
        "TotalQuantity", "MonetaryStd", "MonetaryMin", "MinQuantity", "MaxQuantity"
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    # Parse date + IP before dropping
    df = _parse_registration_date(df)
    df = _parse_last_login_ip(df)

    # Sentinel cleanup
    if "SupportTicketsCount" in df.columns:
        df["SupportTicketsCount"] = df["SupportTicketsCount"].replace([-1,999], np.nan)
    if "SatisfactionScore" in df.columns:
        df["SatisfactionScore"] = df["SatisfactionScore"].replace([-1,99], np.nan)

    # Ordinal encode
    df = _ordinal_encode(df)

    # OHE (excluding Country)
    country_col = df.pop("Country") if "Country" in df.columns else None
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    if country_col is not None:
        df["Country"] = country_col

    # Target encode Country
    df = _target_encode_country(df)

    # Impute
    df = _impute(df)

    # Feature engineering
    df = _engineer_features(df)

    # Log transform
    df = _log_transform(df)

    # Align to expected columns (pre-PCA)
    base_cols = scaler_columns
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing)], axis=1)

    # Scale
    cols_to_scale = [c for c in scaler_columns if c in df.columns]
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # PCA if model was trained on PCA features
    if USE_PCA:
        pca_input = df_scaled[[c for c in pca_numeric_cols if c in df_scaled.columns]]
        pca_arr   = pca.transform(pca_input)
        pca_cols  = [f"PC{i+1}" for i in range(pca_arr.shape[1])]
        df_out    = pd.DataFrame(pca_arr, columns=pca_cols, index=df_scaled.index)
    else:
        df_out = df_scaled

    # Final column alignment
    missing_final = [c for c in expected_columns if c not in df_out.columns]
    if missing_final:
        df_out = pd.concat(
            [df_out, pd.DataFrame(0, index=df_out.index, columns=missing_final)], axis=1
        )
    df_out = df_out[expected_columns]

    return df_out


# ==========================================
# 4️⃣ SINGLE PREDICTION
# ==========================================

def predict_churn(input_dict: dict) -> dict:
    df  = pd.DataFrame([input_dict])
    df  = _preprocess_input(df)

    probability = float(pipeline.predict_proba(df)[0][1])
    prediction  = int(probability >= THRESHOLD)

    if probability < 0.25:   risk = "Faible"
    elif probability < 0.50: risk = "Moyen"
    elif probability < 0.75: risk = "Élevé"
    else:                    risk = "Critique"

    return {
        "prediction" : prediction,
        "label"      : "Churn" if prediction == 1 else "Fidèle",
        "probability": round(probability, 4),
        "risk_level" : risk,
        "threshold"  : round(THRESHOLD, 3),
    }


# ==========================================
# 5️⃣ BATCH PREDICTION (Vectorised)
# ==========================================

def predict_batch(csv_path: str, output_path: str = "reports/predictions.csv"):
    print(f"\nBatch prediction: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"  Loaded {len(df_raw)} customers")

    df_proc       = _preprocess_input(df_raw.copy())
    probabilities = pipeline.predict_proba(df_proc)[:, 1]
    predictions   = (probabilities >= THRESHOLD).astype(int)

    df_out = df_raw.copy()
    df_out["probability"]    = np.round(probabilities, 4)
    df_out["prediction"]     = predictions
    df_out["label"]          = np.where(predictions==1, "Churn", "Fidèle")
    df_out["risk_level"]     = pd.cut(probabilities,
        bins=[-np.inf,0.25,0.50,0.75,np.inf],
        labels=["Faible","Moyen","Élevé","Critique"])
    df_out["threshold_used"] = round(THRESHOLD, 3)

    df_out.to_csv(output_path, index=False)
    print(f"  ✅ Saved → {output_path}")
    print(f"  Fidèle: {(predictions==0).sum()} | Churn: {(predictions==1).sum()}")
    return df_out


# ==========================================
# 6️⃣ DEMO
# ==========================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  DEMO — Single Customer Prediction")
    print("="*55)

    sample_customer = {
        # Core RFM (no Recency — dropped as leaky)
        "Frequency"                : 8,
        "MonetaryTotal"            : 450.0,
        "MonetaryAvg"              : 56.25,
        "MonetaryStd"              : 20.0,
        "MonetaryMin"              : 10.0,
        "MonetaryMax"              : 120.0,
        "TotalQuantity"            : 60,
        "AvgQuantityPerTransaction": 7.5,
        "MinQuantity"              : 1,
        "MaxQuantity"              : 30,
        # Tenure / dates
        "CustomerTenureDays"       : 365,
        "FirstPurchaseDaysAgo"     : 400,
        "RegistrationDate"         : "15/06/2010",   # will be parsed
        # Behaviour
        "PreferredDayOfWeek"       : 2,
        "PreferredHour"            : 14,
        "PreferredMonth"           : 11,
        "WeekendPurchaseRatio"     : 0.25,
        "AvgDaysBetweenPurchases"  : 45,
        "UniqueProducts"           : 20,
        "AvgProductsPerTransaction": 2.5,
        "UniqueCountries"          : 1,
        "CancelledTransactions"    : 0,
        "ReturnRatio"              : 0.0,
        "AvgLinesPerInvoice"       : 10.0,
        # IP (will be parsed)
        "LastLoginIP"              : "85.244.30.10",
        # Demographics
        "Age"                      : 35,
        "SupportTicketsCount"      : 1,
        "SatisfactionScore"        : 4,
        # Categorical
        "SpendingCategory"         : "High",
        "AgeCategory"              : "35-44",
        "BasketSizeCategory"       : "Moyen",
        "PreferredTimeOfDay"       : "Après-midi",
        "FavoriteSeason"           : "Hiver",
        "Region"                   : "UK",
        "WeekendPreference"        : "Semaine",
        "ProductDiversity"         : "Modéré",
        "Gender"                   : "M",
        "AccountStatus"            : "Active",
        "Country"                  : "United Kingdom",
    }

    result = predict_churn(sample_customer)

    print(f"\n  Customer Profile:")
    print(f"    Frequency  : {sample_customer['Frequency']} orders")
    print(f"    Tenure     : {sample_customer['CustomerTenureDays']} days")
    print(f"    Monetary   : £{sample_customer['MonetaryTotal']}")
    print(f"    Country    : {sample_customer['Country']}")
    print(f"    RegDate    : {sample_customer['RegistrationDate']} (parsed automatically)")
    print(f"    IP         : {sample_customer['LastLoginIP']} (IsPrivate/Class extracted)")

    print(f"\n  Prediction Result:")
    print(f"    Label       : {result['label']}")
    print(f"    Probability : {result['probability']*100:.1f}% churn risk")
    print(f"    Risk Level  : {result['risk_level']}")
    print(f"    Threshold   : {result['threshold']}")
    print(f"    Feature set : {FEATURE_NOTE}")