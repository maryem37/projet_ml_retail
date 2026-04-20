# ==========================================
# RETAIL ML PROJECT - FLASK APPLICATION
# ==========================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
import ipaddress

app = Flask(__name__)

# ==========================================
# LOAD ARTIFACTS AT STARTUP
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(*parts):
    return os.path.join(BASE_DIR, *parts)

pipeline         = joblib.load(_path("models", "churn_model.pkl"))
scaler           = joblib.load(_path("models", "scaler.pkl"))
pca              = joblib.load(_path("models", "pca.pkl"))
imputation_stats = joblib.load(_path("models", "imputation_stats.pkl"))
threshold_data   = joblib.load(_path("models", "threshold.pkl"))

THRESHOLD        = threshold_data["threshold"]
USE_PCA          = threshold_data.get("use_pca", False)

country_encoding  = imputation_stats.get("country_encoding", {})
country_churn_map = country_encoding.get("country_churn_map", {})
global_churn_rate = country_encoding.get("global_churn_rate", 0.33)
pca_numeric_cols  = imputation_stats.get("pca_numeric_cols", [])

ref_file         = _path("data", "train_test", "X_train_pca.csv" if USE_PCA else "X_train.csv")
X_ref            = pd.read_csv(ref_file, nrows=1)
expected_columns = X_ref.columns.tolist()
scaler_columns   = list(scaler.feature_names_in_)

# Clustering
CLUSTER_AVAILABLE = False
kmeans            = None
CLUSTERING_FEATURES = [
    "Frequency", "MonetaryTotal", "CustomerTenureDays", "ReturnRatio",
    "CancelledTransactions", "UniqueProducts", "SupportTicketsCount",
    "EngagementScore", "DisengagementScore", "AvgBasketValue", "Country_TargetEnc",
]

try:
    kmeans = joblib.load(_path("models", "kmeans_model.pkl"))
    CLUSTER_AVAILABLE = True
    print("  ✅ kmeans_model.pkl loaded")
except FileNotFoundError:
    print("  ⚠️  kmeans_model.pkl not found — clustering disabled")

# Cluster labels matching the actual clustering.py output
CLUSTER_NAMES = {
    0: {"label": "Occasional",  "emoji": "🔵", "color": "#2c5f8a"},
    1: {"label": "At Risk",     "emoji": "🟠", "color": "#e67e22"},
    2: {"label": "Champions",   "emoji": "🟢", "color": "#27ae60"},
}

print(f"  ✅ Flask app loaded  |  threshold={THRESHOLD:.3f}  |  PCA={USE_PCA}")


# ==========================================
# PREPROCESSING — mirrors predict.py exactly
# ==========================================

OHE_CATEGORIES = {
    "FavoriteSeason"    : ["Automne", "Été", "Hiver", "Printemps"],
    "Region"            : ["Europe", "Overseas", "UK"],
    "WeekendPreference" : ["Semaine", "Weekend"],
    "ProductDiversity"  : ["Diversifié", "Modéré", "Spécialisé"],
    "Gender"            : ["F", "M"],
    "AccountStatus"     : ["Active", "Inactive", "Suspended"],
}


def _ordinal_encode(df):
    if "SpendingCategory" in df.columns and df["SpendingCategory"].dtype == object:
        df["SpendingCategory"] = df["SpendingCategory"].map(
            {"Low": 0, "Medium": 1, "High": 2, "VIP": 3}
        ).fillna(1)
    if "AgeCategory" in df.columns and df["AgeCategory"].dtype == object:
        df["AgeCategory"] = df["AgeCategory"].map(
            {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5}
        ).fillna(3)
    if "BasketSizeCategory" in df.columns and df["BasketSizeCategory"].dtype == object:
        df["BasketSizeCategory"] = df["BasketSizeCategory"].map(
            {"Petit": 0, "Moyen": 1, "Grand": 2, "Inconnu": 3}
        ).fillna(1)
    if "PreferredTimeOfDay" in df.columns and df["PreferredTimeOfDay"].dtype == object:
        df["PreferredTimeOfDay"] = df["PreferredTimeOfDay"].map(
            {"Nuit": 0, "Matin": 1, "Midi": 2, "Après-midi": 3, "Soir": 4}
        ).fillna(2)
    return df


def _apply_ohe(df):
    for col, categories in OHE_CATEGORIES.items():
        sorted_cats = sorted(categories)
        ref_cat     = sorted_cats[0]
        non_ref     = sorted_cats[1:]
        val = str(df[col].iloc[0]) if col in df.columns else ref_cat
        for cat in non_ref:
            df[f"{col}_{cat}"] = int(val == cat)
        if col in df.columns:
            df = df.drop(columns=[col])
    remaining = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if remaining:
        df = df.drop(columns=remaining)
    return df


def _target_encode_country(df):
    if "Country" in df.columns:
        df["Country_TargetEnc"] = (
            df["Country"].map(country_churn_map).fillna(global_churn_rate)
        )
        df = df.drop(columns=["Country"])
    return df


def _impute(df):
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


def _engineer_features(df):
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


def _log_transform(df):
    for col in ["MonetaryTotal", "Frequency", "AvgBasketValue"]:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    return df


def _preprocess_input(df):
    df = df.copy()

    columns_to_drop = [
        "Recency", "CustomerType", "ChurnRiskCategory", "RFMSegment",
        "LoyaltyLevel", "CustomerID", "NewsletterSubscribed", "UniqueInvoices",
        "TotalTransactions", "UniqueDescriptions", "NegativeQuantityCount",
        "FirstPurchaseDaysAgo", "SatisfactionScore", "AvgLinesPerInvoice",
        "TotalQuantity", "MonetaryStd", "MonetaryMin", "MinQuantity", "MaxQuantity",
        "PreferredMonth",
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    df = _ordinal_encode(df)

    country_col = df.pop("Country") if "Country" in df.columns else None
    df = _apply_ohe(df)
    if country_col is not None:
        df["Country"] = country_col

    df = _target_encode_country(df)
    df = _impute(df)
    df = _engineer_features(df)
    df = _log_transform(df)

    missing_scaler = [c for c in scaler_columns if c not in df.columns]
    if missing_scaler:
        df = pd.concat(
            [df, pd.DataFrame(0, index=df.index, columns=missing_scaler)], axis=1
        )

    cols_to_scale = [c for c in scaler_columns if c in df.columns]
    df_scaled     = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

    if USE_PCA:
        pca_input = df_scaled[[c for c in pca_numeric_cols if c in df_scaled.columns]]
        pca_arr   = pca.transform(pca_input)
        pca_cols  = [f"PC{i+1}" for i in range(pca_arr.shape[1])]
        df_out    = pd.DataFrame(pca_arr, columns=pca_cols, index=df_scaled.index)
    else:
        df_out = df_scaled

    missing_final = [c for c in expected_columns if c not in df_out.columns]
    if missing_final:
        df_out = pd.concat(
            [df_out, pd.DataFrame(0, index=df_out.index, columns=missing_final)], axis=1
        )
    df_out = df_out[expected_columns]
    return df_out, df_scaled   # return both: df_out for model, df_scaled for clustering


# ==========================================
# PREDICTION
# ==========================================

def predict_churn(input_dict):
    df_raw          = pd.DataFrame([input_dict])
    df_model, df_sc = _preprocess_input(df_raw)

    probability = float(pipeline.predict_proba(df_model)[0][1])
    prediction  = int(probability >= THRESHOLD)

    # Cluster on the 11 interpretable features (already scaled in df_sc)
    cluster_id   = None
    cluster_info = None
    if CLUSTER_AVAILABLE:
        cluster_cols = [c for c in CLUSTERING_FEATURES if c in df_sc.columns]
        X_cluster    = df_sc[cluster_cols].fillna(0)

        # Pad or trim to match kmeans expected feature count
        while X_cluster.shape[1] < kmeans.n_features_in_:
            X_cluster[f"_pad_{X_cluster.shape[1]}"] = 0
        X_cluster = X_cluster.iloc[:, :kmeans.n_features_in_]

        cluster_id   = int(kmeans.predict(X_cluster)[0])
        cluster_info = CLUSTER_NAMES.get(cluster_id, {
            "label": f"Segment {cluster_id}", "emoji": "⚪", "color": "#888"
        })

    if probability < 0.25:   risk = "Faible"
    elif probability < 0.50: risk = "Moyen"
    elif probability < 0.75: risk = "Élevé"
    else:                    risk = "Critique"

    return {
        "prediction"   : prediction,
        "label"        : "Churn" if prediction == 1 else "Fidèle",
        "probability"  : round(probability * 100, 1),
        "risk_level"   : risk,
        "threshold"    : round(THRESHOLD, 3),
        "cluster_id"   : cluster_id,
        "cluster_label": cluster_info["label"]  if cluster_info else None,
        "cluster_emoji": cluster_info["emoji"]  if cluster_info else None,
        "cluster_color": cluster_info["color"]  if cluster_info else None,
    }


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def index():
    return render_template("index.html", cluster_available=CLUSTER_AVAILABLE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        customer = {
            "Frequency"               : float(data.get("frequency", 1)),
            "MonetaryTotal"           : float(data.get("monetary", 0)),
            "MonetaryMax"             : float(data.get("monetary_max", 0)),
            "AvgQuantityPerTransaction": float(data.get("avg_qty", 1)),
            "CustomerTenureDays"      : float(data.get("tenure", 0)),
            "PreferredDayOfWeek"      : float(data.get("preferred_day", 2)),
            "PreferredHour"           : float(data.get("preferred_hour", 14)),
            "WeekendPurchaseRatio"    : float(data.get("weekend_ratio", 0.3)),
            "AvgDaysBetweenPurchases" : float(data.get("avg_days_between", 30)),
            "UniqueProducts"          : float(data.get("unique_products", 1)),
            "AvgProductsPerTransaction": float(data.get("avg_products", 2)),
            "UniqueCountries"         : float(data.get("unique_countries", 1)),
            "CancelledTransactions"   : float(data.get("cancelled", 0)),
            "ReturnRatio"             : float(data.get("return_ratio", 0)),
            "Age"                     : float(data.get("age", 35)),
            "SupportTicketsCount"     : float(data.get("support_tickets", 0)),
            "SpendingCategory"        : data.get("spending_cat", "Medium"),
            "AgeCategory"             : data.get("age_cat", "35-44"),
            "BasketSizeCategory"      : data.get("basket_size", "Moyen"),
            "PreferredTimeOfDay"      : data.get("preferred_time", "Après-midi"),
            "FavoriteSeason"          : data.get("season", "Hiver"),
            "Region"                  : data.get("region", "UK"),
            "WeekendPreference"       : data.get("weekend_pref", "Semaine"),
            "ProductDiversity"        : data.get("prod_diversity", "Modéré"),
            "Gender"                  : data.get("gender", "M"),
            "AccountStatus"           : data.get("account_status", "Active"),
            "Country"                 : data.get("country", "United Kingdom"),
            "RegMonth"                : int(data.get("reg_month", 6)),
            "RegDay"                  : int(data.get("reg_day", 15)),
            "RegWeekday"              : int(data.get("reg_weekday", 2)),
            "IsPrivateIP"             : int(data.get("is_private_ip", 0)),
            "IPClass"                 : int(data.get("ip_class", 1)),
        }

        result = predict_churn(customer)
        return jsonify({"success": True, "result": result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)