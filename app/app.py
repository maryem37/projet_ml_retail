# ==========================================
# RETAIL ML PROJECT - FLASK APPLICATION
# ==========================================
# FIXES:
#   ✅ /debug endpoint — shows exactly what values reach the model
#   ✅ Console prints received payload on every /predict call
#   ✅ avg_qty → AvgQuantityPerTransaction (was defaulting to 1)
#   ✅ avg_products → AvgProductsPerTransaction (separate feature)
#   ✅ unique_countries read from payload
#   ✅ Cluster feature alignment uses exact CLUSTERING_FEATURES order
#   ✅ Full traceback printed on errors
# ==========================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


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

THRESHOLD       = threshold_data["threshold"]
USE_PCA         = threshold_data.get("use_pca", False)

country_encoding  = imputation_stats.get("country_encoding", {})
country_churn_map = country_encoding.get("country_churn_map", {})
global_churn_rate = country_encoding.get("global_churn_rate", 0.33)
pca_numeric_cols  = imputation_stats.get("pca_numeric_cols", [])

ref_file         = _path("data", "train_test", "X_train_pca.csv" if USE_PCA else "X_train.csv")
X_ref            = pd.read_csv(ref_file, nrows=1)
expected_columns = X_ref.columns.tolist()
scaler_columns   = list(scaler.feature_names_in_)

CLUSTERING_FEATURES = [
    "Frequency", "MonetaryTotal", "CustomerTenureDays", "ReturnRatio",
    "CancelledTransactions", "UniqueProducts", "SupportTicketsCount",
    "EngagementScore", "DisengagementScore", "AvgBasketValue", "Country_TargetEnc",
]

CLUSTER_AVAILABLE = False
kmeans            = None

try:
    kmeans = joblib.load(_path("models", "kmeans_model.pkl"))
    CLUSTER_AVAILABLE = True
    print(f"  ✅ kmeans_model.pkl loaded  (expects {kmeans.n_features_in_} features)")
except FileNotFoundError:
    print("  ⚠️  kmeans_model.pkl not found — clustering disabled")

CLUSTER_NAMES = {
    0: {"label": "Occasional",  "emoji": "🔵", "color": "#2c5f8a"},
    1: {"label": "At Risk",     "emoji": "🟠", "color": "#e67e22"},
    2: {"label": "Champions",   "emoji": "🟢", "color": "#27ae60"},
}

print(f"  ✅ Flask app loaded  |  threshold={THRESHOLD:.3f}  |  PCA={USE_PCA}")
print(f"  ✅ GBM root split: CustomerTenureDays ≤ 189.5 days")
print(f"     → tenure < 190 → higher churn branch (52.9% churn rate)")
print(f"     → tenure > 190 → lower  churn branch (42.1% churn rate)")


# ==========================================
# OHE CATEGORY MAP
# ==========================================

OHE_CATEGORIES = {
    "FavoriteSeason"    : ["Automne", "Été", "Hiver", "Printemps"],
    "Region"            : ["Europe", "Overseas", "UK"],
    "WeekendPreference" : ["Semaine", "Weekend"],
    "ProductDiversity"  : ["Diversifié", "Modéré", "Spécialisé"],
    "Gender"            : ["F", "M"],
    "AccountStatus"     : ["Active", "Inactive", "Suspended"],
}


# ==========================================
# PREPROCESSING HELPERS
# ==========================================

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
        non_ref     = sorted_cats[1:]
        val = str(df[col].iloc[0]) if col in df.columns else sorted_cats[0]
        for cat in non_ref:
            df[f"{col}_{cat}"] = int(val == cat)
        if col in df.columns:
            df = df.drop(columns=[col])
    leftover = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if leftover:
        df = df.drop(columns=leftover)
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


def _build_customer_dict(data: dict) -> dict:
    return {
        "CustomerTenureDays"       : float(data.get("tenure",           365)),
        "Frequency"                : float(data.get("frequency",        1)),
        "AvgQuantityPerTransaction": float(data.get("avg_qty",          5)),
        "AvgProductsPerTransaction": float(data.get("avg_products",     2)),
        "UniqueProducts"           : float(data.get("unique_products",  1)),
        "MonetaryTotal"            : float(data.get("monetary",         0)),
        "MonetaryMax"              : float(data.get("monetary_max",     0)),
        "AvgDaysBetweenPurchases"  : float(data.get("avg_days_between", 30)),
        "UniqueCountries"          : float(data.get("unique_countries", 1)),
        "CancelledTransactions"    : float(data.get("cancelled",        0)),
        "ReturnRatio"              : float(data.get("return_ratio",     0)),
        "SupportTicketsCount"      : float(data.get("support_tickets",  0)),
        "PreferredDayOfWeek"       : float(data.get("preferred_day",    2)),
        "PreferredHour"            : float(data.get("preferred_hour",   14)),
        "WeekendPurchaseRatio"     : float(data.get("weekend_ratio",    0.3)),
        "Age"                      : float(data.get("age",              35)),
        "SpendingCategory"         : data.get("spending_cat",    "Medium"),
        "AgeCategory"              : data.get("age_cat",         "35-44"),
        "BasketSizeCategory"       : data.get("basket_size",     "Moyen"),
        "PreferredTimeOfDay"       : data.get("preferred_time",  "Après-midi"),
        "FavoriteSeason"           : data.get("season",          "Automne"),
        "Region"                   : data.get("region",          "UK"),
        "WeekendPreference"        : data.get("weekend_pref",    "Semaine"),
        "ProductDiversity"         : data.get("prod_diversity",  "Modéré"),
        "Gender"                   : data.get("gender",          "M"),
        "AccountStatus"            : data.get("account_status",  "Active"),
        "Country"                  : data.get("country",         "United Kingdom"),
        "RegMonth"                 : int(data.get("reg_month",   6)),
        "RegDay"                   : int(data.get("reg_day",     15)),
        "RegWeekday"               : int(data.get("reg_weekday", 2)),
        "IsPrivateIP"              : int(data.get("is_private_ip", 0)),
        "IPClass"                  : int(data.get("ip_class",      1)),
    }


def _preprocess_input(df):
    df = df.copy()
    cols_to_drop = [
        "Recency", "CustomerType", "ChurnRiskCategory", "RFMSegment",
        "LoyaltyLevel", "CustomerID", "NewsletterSubscribed", "UniqueInvoices",
        "TotalTransactions", "UniqueDescriptions", "NegativeQuantityCount",
        "FirstPurchaseDaysAgo", "SatisfactionScore", "AvgLinesPerInvoice",
        "TotalQuantity", "MonetaryStd", "MonetaryMin", "MinQuantity", "MaxQuantity",
        "PreferredMonth",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
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
    df_scaled = df.copy()
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
    return df_out, df_scaled


def predict_churn(customer: dict) -> dict:
    df_raw              = pd.DataFrame([customer])
    df_model, df_scaled = _preprocess_input(df_raw)
    probability         = float(pipeline.predict_proba(df_model)[0][1])
    prediction          = int(probability >= THRESHOLD)

    cluster_id   = None
    cluster_info = None
    if CLUSTER_AVAILABLE:
        cluster_data = {col: (df_scaled[col].values[0] if col in df_scaled.columns else 0.0)
                        for col in CLUSTERING_FEATURES}
        X_cluster = pd.DataFrame([cluster_data])[CLUSTERING_FEATURES]
        n_exp = kmeans.n_features_in_
        if X_cluster.shape[1] < n_exp:
            for i in range(n_exp - X_cluster.shape[1]):
                X_cluster[f"_pad_{i}"] = 0.0
        elif X_cluster.shape[1] > n_exp:
            X_cluster = X_cluster.iloc[:, :n_exp]
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


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"success": False, "error": "Invalid or empty JSON body"}), 400

        # Always print to console so you can verify values are arriving correctly
        print(f"\n[PREDICT] spending={data.get('spending_cat')}  "
              f"freq={data.get('frequency')}  avg_qty={data.get('avg_qty')}  "
              f"tenure={data.get('tenure')}  season={data.get('season')}")

        customer = _build_customer_dict(data)
        result   = predict_churn(customer)

        print(f"         → prob={result['probability']}%  label={result['label']}")
        return jsonify({"success": True, "result": result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/debug", methods=["POST", "OPTIONS"])
def debug():
    """
    Debug endpoint — paste into your browser console to verify values:

    fetch('http://127.0.0.1:5000/debug', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
            spending_cat:'Low', frequency:1, avg_qty:1,
            tenure:50, season:'Hiver'
        })
    }).then(r=>r.json()).then(d=>console.table(d))
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data     = request.get_json(force=True, silent=True) or {}
        customer = _build_customer_dict(data)

        df_raw              = pd.DataFrame([customer])
        df_model, df_scaled = _preprocess_input(df_raw)
        probability         = float(pipeline.predict_proba(df_model)[0][1])

        key_cols = [
            "CustomerTenureDays", "Frequency", "SpendingCategory",
            "AvgQuantityPerTransaction", "UniqueProducts", "MonetaryTotal",
            "EngagementScore", "DisengagementScore", "AvgBasketValue",
            "FavoriteSeason_Hiver", "FavoriteSeason_Printemps",
        ]
        scaled_vals = {
            col: round(float(df_model[col].values[0]), 4)
            for col in key_cols if col in df_model.columns
        }

        return jsonify({
            "step1_received": {k: data.get(k) for k in
                ["spending_cat","frequency","avg_qty","tenure","season","unique_products"]},
            "step2_customer_dict": {
                "SpendingCategory"         : customer["SpendingCategory"],
                "Frequency"                : customer["Frequency"],
                "AvgQuantityPerTransaction": customer["AvgQuantityPerTransaction"],
                "CustomerTenureDays"       : customer["CustomerTenureDays"],
                "FavoriteSeason"           : customer["FavoriteSeason"],
            },
            "step3_scaled_model_input": scaled_vals,
            "result": {
                "probability_pct": round(probability * 100, 2),
                "prediction"     : "Churn" if probability >= THRESHOLD else "Fidèle",
                "threshold"      : THRESHOLD,
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)