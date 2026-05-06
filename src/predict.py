# ==========================================
# RETAIL ML PROJECT - PREDICTION SCRIPT
# ==========================================
# UPDATED to match preprocessing_v3.py:
#   ✅ RegistrationDate parsing → RegYear/RegMonth/RegDay/RegWeekday
#   ✅ LastLoginIP → IsPrivateIP + IPClass
#   ✅ Country → Target Encoding (not OHE)
#   ✅ PCA applied if training used PCA (auto-detected from threshold.pkl)
#   ✅ Full calibrated pipeline used for predict_proba
#   ✅ Tuned threshold applied
#   ✅ OHE replaced with hardcoded category dict — safe for single-row inference
#
# FIXES in this version (v2) — synchronized with preprocessing_v3.py:
#   ✅ RegYear now extracted in _parse_registration_date
#      (was silently skipped — missing column zero-filled → biased predictions)
#   ✅ DisengagementScore uses z-score normalisation with train stats
#      loaded from imputation_stats.pkl (replaces arbitrary /10 divisor)
#   ✅ _log_transform extended to all 15 high-skew columns matching
#      preprocessing_v3.py (was only 3 columns — silent feature mismatch)
#   ✅ AvgBasketValue computed BEFORE log-transforms (order preserved)
#   ✅ columns_to_drop updated: MultiCollinearity drops included
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os
import warnings
import ipaddress

os.makedirs("reports", exist_ok=True)


# ==========================================
# 1️⃣ LOAD ARTIFACTS
# ==========================================

print("Loading model artifacts...")

pipeline         = joblib.load("models/churn_model.pkl")   # CalibratedClassifierCV
scaler           = joblib.load("models/scaler.pkl")
pca              = joblib.load("models/pca.pkl")
imputation_stats = joblib.load("models/imputation_stats.pkl")
threshold_data   = joblib.load("models/threshold.pkl")

THRESHOLD    = threshold_data["threshold"]
USE_PCA      = threshold_data.get("use_pca", False)
FEATURE_NOTE = threshold_data.get("feature_set", "Raw scaled features")
TUNER_USED   = threshold_data.get("tuner_used", "GridSearchCV")

country_encoding  = imputation_stats.get("country_encoding", {})
country_churn_map = country_encoding.get("country_churn_map", {})
global_churn_rate = country_encoding.get("global_churn_rate", 0.33)
pca_numeric_cols  = imputation_stats.get("pca_numeric_cols", [])

# DisengagementScore normalisation stats — FIX v2
DISENG_RR_MEAN = imputation_stats.get("disengagement_rr_mean", 0.0)
DISENG_RR_STD  = imputation_stats.get("disengagement_rr_std",  1.0)
DISENG_CT_MEAN = imputation_stats.get("disengagement_ct_mean", 0.0)
DISENG_CT_STD  = imputation_stats.get("disengagement_ct_std",  1.0)

print(f"  ✅ churn_model.pkl      loaded  (CalibratedClassifierCV)")
print(f"  ✅ scaler.pkl           loaded")
print(f"  ✅ pca.pkl              loaded")
print(f"  ✅ imputation_stats.pkl loaded")
print(f"  ✅ threshold.pkl        loaded  (threshold={THRESHOLD:.3f}, PCA={USE_PCA})")
print(f"     Trained with : {FEATURE_NOTE} | Tuner: {TUNER_USED}")


# ==========================================
# 2️⃣ LOAD REFERENCE COLUMNS
# ==========================================

print("Loading reference columns...")

ref_file         = "data/train_test/X_train_pca.csv" if USE_PCA else "data/train_test/X_train.csv"
X_ref            = pd.read_csv(ref_file, nrows=1)
expected_columns = X_ref.columns.tolist()
scaler_columns   = list(scaler.feature_names_in_)

print(f"  ✅ {len(expected_columns)} expected features loaded")
print(f"  ✅ Using: {'PCA features' if USE_PCA else 'Raw scaled features'}")


# ==========================================
# 3️⃣ OHE CATEGORY MAP
# ==========================================
# Inference-safe OHE: hardcoded training categories instead of pd.get_dummies().
# pd.get_dummies() on a single row cannot infer all training categories —
# missing dummies are silently absent, then zero-filled → biased toward Fidèle.
#
# Rules (must match preprocessing_v3.py exactly):
#   - drop_first=True → first alphabetical category = reference (no column)
#   - Column name format: "{original_col}_{category_value}"
# ==========================================

OHE_CATEGORIES = {
    # FavoriteSeason REMOVED — confirmed leakage (bijective recode of PreferredMonth)
    # Diagnostic 2026-05-06: Match rate 100%, churn rates 1.9%/51%/61%/50% by season
    "Region"            : ["Europe", "Overseas", "UK"],
    "WeekendPreference" : ["Semaine", "Weekend"],
    "ProductDiversity"  : ["Diversifié", "Modéré", "Spécialisé"],
    "Gender"            : ["F", "M"],
    "AccountStatus"     : ["Active", "Inactive", "Suspended"],
}

# Columns dropped in preprocessing — silently ignored at inference
COLUMNS_TO_DROP = [
    "Recency", "CustomerType", "ChurnRiskCategory", "RFMSegment",
    "LoyaltyLevel", "CustomerID", "NewsletterSubscribed", "UniqueInvoices",
    "TotalTransactions", "UniqueDescriptions", "NegativeQuantityCount",
    "FirstPurchaseDaysAgo", "SatisfactionScore",
    "AvgLinesPerInvoice", "TotalQuantity", "MonetaryStd", "MonetaryMin",
    "MinQuantity", "MaxQuantity",
    "PreferredMonth",   # temporal leakage
    "FavoriteSeason",   # CONFIRMED leakage — bijective recode of PreferredMonth (match=100%)
    "SpendingCategory",  # CONFIRMED leakage — clean-cut discretisation of MonetaryTotal
    "BasketSizeCategory",# CONFIRMED leakage — encodes AvgBasketValue/MonetaryTotal
]

# Log-transform columns — FIX v2: matches preprocessing_v3.py exactly (15 cols)
LOG_COLS = [
    "MonetaryTotal",
    "Frequency",
    "AvgBasketValue",            # computed before log-transforms (step 5)
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
    # AvgQuantityPerTransaction: max=12540 vs Q75=14 (317x above fence)
    # Log-transform neutralises outlier dominance in tree splits.
    "AvgQuantityPerTransaction",
]


# ==========================================
# 4️⃣ PREPROCESSING HELPERS
# ==========================================

def _drop_leaky_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[c for c in COLUMNS_TO_DROP if c in df.columns],
        errors="ignore"
    )


def _parse_registration_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract RegYear / RegMonth / RegDay / RegWeekday from RegistrationDate.
    FIX v2: RegYear is now extracted — preprocessing_v3.py includes it as
    a feature (customer cohort signal). Omitting it caused silent zero-fill
    and biased predictions toward Fidèle.
    Fallback values = train medians (approximate).
    """
    if "RegistrationDate" in df.columns:
        dates = pd.to_datetime(
            df["RegistrationDate"],
            format="mixed",
            dayfirst=False,
            errors="coerce"
        )
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


def _ordinal_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encoding — must match preprocessing_v3.py exactly."""
    # SpendingCategory removed — confirmed leakage (clean-cut of MonetaryTotal)
    # BasketSizeCategory removed — confirmed leakage (encodes monetary behaviour)

    if "AgeCategory" in df.columns and df["AgeCategory"].dtype == object:
        df["AgeCategory"] = df["AgeCategory"].map(
            {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5}
        ).fillna(3)

    if "PreferredTimeOfDay" in df.columns and df["PreferredTimeOfDay"].dtype == object:
        df["PreferredTimeOfDay"] = df["PreferredTimeOfDay"].map(
            {"Nuit": 0, "Matin": 1, "Midi": 2, "Après-midi": 3, "Soir": 4}
        ).fillna(2)

    return df


def _apply_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inference-safe OHE using hardcoded training categories.
    drop_first=True: first alphabetical category = reference (no column created).
    """
    for col, categories in OHE_CATEGORIES.items():
        sorted_cats = sorted(categories)
        ref_cat     = sorted_cats[0]     # reference — no column
        non_ref     = sorted_cats[1:]    # these get a binary column each

        val = str(df[col].iloc[0]) if col in df.columns else ref_cat

        for cat in non_ref:
            df[f"{col}_{cat}"] = int(val == cat)

        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop any remaining unexpected string/category columns
    remaining_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if remaining_cat:
        df = df.drop(columns=remaining_cat)

    return df


def _target_encode_country(df: pd.DataFrame) -> pd.DataFrame:
    """Apply target encoding to Country using train churn map."""
    if "Country" in df.columns:
        df["Country_TargetEnc"] = (
            df["Country"].map(country_churn_map).fillna(global_churn_rate)
        )
        df = df.drop(columns=["Country"])
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using train statistics from imputation_stats.pkl."""
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
            df["SupportTicketsCount"]
            .replace([-1, 999], np.nan)
            .fillna(stats.get("support_median", 2))
        )

    for col in ["RegYear", "RegMonth", "RegDay", "RegWeekday"]:
        if col in df.columns:
            med = stats.get("train_medians", {}).get(col, None)
            if med is not None:
                df[col] = df[col].fillna(med)

    if "IsPrivateIP" in df.columns:
        df["IsPrivateIP"] = df["IsPrivateIP"].replace(-1, np.nan).fillna(0)

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering — must match preprocessing_v3.py exactly.
    ORDER MATTERS: AvgBasketValue is computed BEFORE log-transforms.

    FIX v2: DisengagementScore now uses z-score normalisation with train
    statistics loaded from imputation_stats.pkl. Previous formula used
    an arbitrary /10 divisor that caused CancelledTransactions to dominate.
    """
    # AvgBasketValue — computed BEFORE log-transforms (preprocessing order)
    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)
    else:
        missing = [c for c in ["MonetaryTotal", "Frequency"] if c not in df.columns]
        warnings.warn(
            f"Cannot compute AvgBasketValue — missing: {missing}. Defaulting to 0.0.",
            UserWarning, stacklevel=2,
        )
        df["AvgBasketValue"] = 0.0

    # EngagementScore
    if "Frequency" in df.columns and "CustomerTenureDays" in df.columns:
        df["EngagementScore"] = df["Frequency"] / (df["CustomerTenureDays"] + 1)

    # DisengagementScore — FIX v2: z-score normalisation with train stats
    if "ReturnRatio" in df.columns and "CancelledTransactions" in df.columns:
        df["DisengagementScore"] = (
            (df["ReturnRatio"]           - DISENG_RR_MEAN) / DISENG_RR_STD
            + (df["CancelledTransactions"] - DISENG_CT_MEAN) / DISENG_CT_STD
        )

    # RevenueIndex
    if "MonetaryTotal" in df.columns:
        gm = imputation_stats.get("global_mean_monetary", 1908.19)
        df["RevenueIndex"] = df["MonetaryTotal"] / (gm + 1)

    return df


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX v2: Extended from 3 → 15 columns to match preprocessing_v3.py.
    All columns in LOG_COLS are sign-preserved log1p transformed.
    Columns absent from df are silently skipped (guard via `if col in df.columns`).
    """
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    return df


def _preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full inference preprocessing pipeline.
    Mirrors preprocessing_v3.py step by step.
    """
    df = df.copy()

    # Step 1: drop leaky / redundant / constant columns
    df = _drop_leaky_cols(df)

    # Step 2: parse RegistrationDate → RegYear/RegMonth/RegDay/RegWeekday
    df = _parse_registration_date(df)

    # Step 3: parse LastLoginIP → IsPrivateIP + IPClass
    df = _parse_last_login_ip(df)

    # Step 4: sentinel cleanup
    if "SupportTicketsCount" in df.columns:
        df["SupportTicketsCount"] = df["SupportTicketsCount"].replace([-1, 999], np.nan)

    # Step 5: ordinal encoding
    df = _ordinal_encode(df)

    # Step 6: OHE (inference-safe, hardcoded categories)
    # Extract Country before OHE so it isn't dropped by _apply_ohe
    country_col = df.pop("Country") if "Country" in df.columns else None
    df = _apply_ohe(df)
    if country_col is not None:
        df["Country"] = country_col

    # Step 7: target encode Country
    df = _target_encode_country(df)

    # Step 8: impute missing values with train statistics
    df = _impute(df)

    # Step 9: feature engineering (AvgBasketValue BEFORE log-transforms)
    df = _engineer_features(df)

    # Step 10: log-transform all 15 high-skew columns
    df = _log_transform(df)

    # Step 11: align to scaler columns (fill missing with 0)
    missing_scaler = [c for c in scaler_columns if c not in df.columns]
    if missing_scaler:
        df = pd.concat(
            [df, pd.DataFrame(0, index=df.index, columns=missing_scaler)],
            axis=1
        )

    # Step 12: scale
    cols_to_scale = [c for c in scaler_columns if c in df.columns]
    df_scaled     = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Step 13: PCA if model was trained on PCA features
    if USE_PCA:
        pca_input = df_scaled[[c for c in pca_numeric_cols if c in df_scaled.columns]]
        pca_arr   = pca.transform(pca_input)
        pca_cols  = [f"PC{i+1}" for i in range(pca_arr.shape[1])]
        df_out    = pd.DataFrame(pca_arr, columns=pca_cols, index=df_scaled.index)
    else:
        df_out = df_scaled

    # Step 14: final column alignment to expected model input
    missing_final = [c for c in expected_columns if c not in df_out.columns]
    if missing_final:
        df_out = pd.concat(
            [df_out, pd.DataFrame(0, index=df_out.index, columns=missing_final)],
            axis=1
        )
    df_out = df_out[expected_columns]

    return df_out


# ==========================================
# 5️⃣ SINGLE PREDICTION
# ==========================================

def predict_churn(input_dict: dict) -> dict:
    """
    Predict churn probability for a single customer.
    Returns label, probability, risk level, and threshold used.
    """
    df          = pd.DataFrame([input_dict])
    df          = _preprocess_input(df)
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
# 6️⃣ BATCH PREDICTION (Vectorised)
# ==========================================

def predict_batch(csv_path: str, output_path: str = "reports/predictions.csv"):
    """
    Predict churn for all customers in a CSV file.
    Saves results with probability, label, risk level, and threshold used.
    """
    print(f"\nBatch prediction: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"  Loaded {len(df_raw)} customers")

    df_proc       = _preprocess_input(df_raw.copy())
    probabilities = pipeline.predict_proba(df_proc)[:, 1]
    predictions   = (probabilities >= THRESHOLD).astype(int)

    df_out = df_raw.copy()
    df_out["probability"]    = np.round(probabilities, 4)
    df_out["prediction"]     = predictions
    df_out["label"]          = np.where(predictions == 1, "Churn", "Fidèle")
    df_out["risk_level"]     = pd.cut(
        probabilities,
        bins   = [-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels = ["Faible", "Moyen", "Élevé", "Critique"]
    )
    df_out["threshold_used"] = round(THRESHOLD, 3)

    df_out.to_csv(output_path, index=False)
    print(f"  ✅ Saved → {output_path}")
    print(f"  Fidèle : {(predictions==0).sum()} | Churn : {(predictions==1).sum()}")
    return df_out


# ==========================================
# 7️⃣ DEMO
# ==========================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  DEMO — Single Customer Prediction")
    print("="*55)

    sample_customer = {
        # Core RFM (no Recency — dropped as leaky)
        "Frequency"                 : 8,
        "MonetaryTotal"             : 450.0,
        "MonetaryAvg"               : 56.25,
        "MonetaryMax"               : 120.0,
        "AvgQuantityPerTransaction" : 7.5,
        # Tenure / dates
        "CustomerTenureDays"        : 365,
        "RegistrationDate"          : "15/06/2010",   # → RegYear/Month/Day/Weekday
        # Behaviour
        "PreferredDayOfWeek"        : 2,
        "PreferredHour"             : 14,
        "WeekendPurchaseRatio"      : 0.25,
        "AvgDaysBetweenPurchases"   : 45,
        "UniqueProducts"            : 20,
        "AvgProductsPerTransaction" : 2.5,
        "UniqueCountries"           : 1,
        "ZeroPriceCount"            : 0,
        "CancelledTransactions"     : 0,
        "ReturnRatio"               : 0.0,
        # IP (parsed → IsPrivateIP + IPClass)
        "LastLoginIP"               : "85.244.30.10",
        # Demographics
        "Age"                       : 35,
        "SupportTicketsCount"       : 1,
        # Categorical — ordinal
        # "SpendingCategory" removed — confirmed leakage
        "AgeCategory"               : "35-44",
        # "BasketSizeCategory" removed — confirmed leakage
        "PreferredTimeOfDay"        : "Après-midi",
        # Categorical — OHE (hardcoded inference-safe)
        # "FavoriteSeason" removed — confirmed leakage, dropped in preprocessing
        "Region"                    : "UK",
        "WeekendPreference"         : "Semaine",
        "ProductDiversity"          : "Modéré",
        "Gender"                    : "M",
        "AccountStatus"             : "Active",
        # Country — target encoded
        "Country"                   : "United Kingdom",
    }

    result = predict_churn(sample_customer)

    print(f"\n  Customer Profile:")
    print(f"    Frequency  : {sample_customer['Frequency']} orders")
    print(f"    Tenure     : {sample_customer['CustomerTenureDays']} days")
    print(f"    Monetary   : £{sample_customer['MonetaryTotal']}")
    print(f"    Country    : {sample_customer['Country']}")
    print(f"    RegDate    : {sample_customer['RegistrationDate']}"
          f"  → RegYear/Month/Day/Weekday extracted")
    print(f"    IP         : {sample_customer['LastLoginIP']}"
          f"  → IsPrivateIP + IPClass extracted")

    print(f"\n  Prediction Result:")
    print(f"    Label       : {result['label']}")
    print(f"    Probability : {result['probability']*100:.1f}% churn risk")
    print(f"    Risk Level  : {result['risk_level']}")
    print(f"    Threshold   : {result['threshold']}")
    print(f"    Feature set : {FEATURE_NOTE}")
    print(f"    Tuner       : {TUNER_USED}")