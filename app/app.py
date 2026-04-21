# ==========================================
# RETAIL ML PROJECT - FLASK APPLICATION
# ==========================================
# ENDPOINTS:
#   GET  /                   → main UI
#   POST /predict            → churn prediction
#   POST /predict_revenue    → revenue regression (NEW)
#   POST /debug              → pipeline trace
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
# LOAD ALL ARTIFACTS AT STARTUP
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(*parts):
    return os.path.join(BASE_DIR, *parts)


# Shared preprocessing artifacts
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

# Regression artifact
REGRESSION_AVAILABLE = False
reg_pipeline         = None
reg_leaky_cols       = ["MonetaryTotal", "AvgBasketValue", "RevenueIndex", "MonetaryAvg", "MonetaryMax"]
reg_cap_value        = 0.0
reg_metrics          = {}
reg_model_name       = "Unknown"

try:
    reg_artifact   = joblib.load(_path("models", "regression_model.pkl"))
    reg_pipeline   = reg_artifact["pipeline"]
    reg_leaky_cols = reg_artifact.get("leaky_cols_dropped", reg_leaky_cols)
    reg_cap_value  = float(reg_artifact.get("outlier_cap_value", 0))
    reg_metrics    = reg_artifact.get("metrics", {})
    reg_model_name = reg_artifact.get("best_model_name", "Regression Model")
    REGRESSION_AVAILABLE = True
    print(f"  ✅ regression_model.pkl loaded  ({reg_model_name})")
    print(f"     Cap=£{reg_cap_value:,.0f}  MedAE=£{reg_metrics.get('medae_capped',0):,.0f}")
except FileNotFoundError:
    print("  ⚠️  regression_model.pkl not found — run src/regression.py first")

# Clustering artifact
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
    print(f"  ✅ kmeans_model.pkl loaded  ({kmeans.n_features_in_} features)")
except FileNotFoundError:
    print("  ⚠️  kmeans_model.pkl not found")

CLUSTER_NAMES = {
    0: {"label": "Occasional", "emoji": "🔵", "color": "#2c5f8a"},
    1: {"label": "At Risk",    "emoji": "🟠", "color": "#e67e22"},
    2: {"label": "Champions",  "emoji": "🟢", "color": "#27ae60"},
}

print(f"  ✅ Flask loaded | threshold={THRESHOLD:.3f} | PCA={USE_PCA} | "
      f"Regression={'ON' if REGRESSION_AVAILABLE else 'OFF'}")


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
# SHARED PREPROCESSING
# ==========================================

def _ordinal_encode(df):
    if "SpendingCategory" in df.columns and df["SpendingCategory"].dtype == object:
        df["SpendingCategory"] = df["SpendingCategory"].map(
            {"Low": 0, "Medium": 1, "High": 2, "VIP": 3}).fillna(1)
    if "AgeCategory" in df.columns and df["AgeCategory"].dtype == object:
        df["AgeCategory"] = df["AgeCategory"].map(
            {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5}).fillna(3)
    if "BasketSizeCategory" in df.columns and df["BasketSizeCategory"].dtype == object:
        df["BasketSizeCategory"] = df["BasketSizeCategory"].map(
            {"Petit": 0, "Moyen": 1, "Grand": 2, "Inconnu": 3}).fillna(1)
    if "PreferredTimeOfDay" in df.columns and df["PreferredTimeOfDay"].dtype == object:
        df["PreferredTimeOfDay"] = df["PreferredTimeOfDay"].map(
            {"Nuit": 0, "Matin": 1, "Midi": 2, "Après-midi": 3, "Soir": 4}).fillna(2)
    return df


def _apply_ohe(df):
    for col, categories in OHE_CATEGORIES.items():
        sorted_cats = sorted(categories)
        non_ref     = sorted_cats[1:]
        v = str(df[col].iloc[0]) if col in df.columns else sorted_cats[0]
        for cat in non_ref:
            df[f"{col}_{cat}"] = int(v == cat)
        if col in df.columns:
            df = df.drop(columns=[col])
    for c in df.select_dtypes(include=["object", "category"]).columns:
        df = df.drop(columns=[c])
    return df


def _target_encode_country(df):
    if "Country" in df.columns:
        df["Country_TargetEnc"] = df["Country"].map(country_churn_map).fillna(global_churn_rate)
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
            .fillna(stats.get("support_median", 2)))
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
        "Recency", "CustomerType", "ChurnRiskCategory", "RFMSegment", "LoyaltyLevel",
        "CustomerID", "NewsletterSubscribed", "UniqueInvoices", "TotalTransactions",
        "UniqueDescriptions", "NegativeQuantityCount", "FirstPurchaseDaysAgo",
        "SatisfactionScore", "AvgLinesPerInvoice", "TotalQuantity", "MonetaryStd",
        "MonetaryMin", "MinQuantity", "MaxQuantity", "PreferredMonth",
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

    missing = [c for c in scaler_columns if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing)], axis=1)

    df_scaled = df.copy()
    df_scaled[[c for c in scaler_columns if c in df.columns]] = \
        scaler.transform(df[[c for c in scaler_columns if c in df.columns]])

    if USE_PCA:
        pca_in  = df_scaled[[c for c in pca_numeric_cols if c in df_scaled.columns]]
        pca_arr = pca.transform(pca_in)
        df_out  = pd.DataFrame(pca_arr,
                               columns=[f"PC{i+1}" for i in range(pca_arr.shape[1])],
                               index=df_scaled.index)
    else:
        df_out = df_scaled

    missing2 = [c for c in expected_columns if c not in df_out.columns]
    if missing2:
        df_out = pd.concat([df_out, pd.DataFrame(0, index=df_out.index, columns=missing2)], axis=1)
    return df_out[expected_columns], df_scaled


# ==========================================
# CHURN PREDICTION
# ==========================================

def run_churn(customer: dict) -> dict:
    df_raw, df_scaled = _preprocess_input(pd.DataFrame([customer]))
    prob       = float(pipeline.predict_proba(df_raw)[0][1])
    prediction = int(prob >= THRESHOLD)

    cluster_id   = None
    cluster_info = None
    if CLUSTER_AVAILABLE:
        cd = {col: (df_scaled[col].values[0] if col in df_scaled.columns else 0.0)
              for col in CLUSTERING_FEATURES}
        Xc = pd.DataFrame([cd])[CLUSTERING_FEATURES]
        ne = kmeans.n_features_in_
        if Xc.shape[1] < ne:
            for i in range(ne - Xc.shape[1]):
                Xc[f"_p{i}"] = 0.0
        elif Xc.shape[1] > ne:
            Xc = Xc.iloc[:, :ne]
        cluster_id   = int(kmeans.predict(Xc)[0])
        cluster_info = CLUSTER_NAMES.get(cluster_id,
                                         {"label": f"Segment {cluster_id}", "emoji": "⚪", "color": "#888"})

    risk = ("Faible" if prob < 0.25 else "Moyen" if prob < 0.50
            else "Élevé" if prob < 0.75 else "Critique")

    return {
        "prediction"   : prediction,
        "label"        : "Churn" if prediction == 1 else "Fidèle",
        "probability"  : round(prob * 100, 1),
        "risk_level"   : risk,
        "threshold"    : round(THRESHOLD, 3),
        "cluster_id"   : cluster_id,
        "cluster_label": cluster_info["label"]  if cluster_info else None,
        "cluster_emoji": cluster_info["emoji"]  if cluster_info else None,
        "cluster_color": cluster_info["color"]  if cluster_info else None,
    }


# ==========================================
# REVENUE PREDICTION
# ==========================================

def run_revenue(customer: dict) -> dict:
    _, df_scaled = _preprocess_input(pd.DataFrame([customer]))

    # Drop leaky cols that were excluded during regression training
    X_reg = df_scaled.drop(
        columns=[c for c in reg_leaky_cols if c in df_scaled.columns], errors="ignore"
    )

    # Align to regression model's expected feature set
    inner = reg_pipeline.named_steps["model"]
    if hasattr(inner, "feature_names_in_"):
        reg_cols = list(inner.feature_names_in_)
        for c in reg_cols:
            if c not in X_reg.columns:
                X_reg[c] = 0.0
        X_reg = X_reg[reg_cols]
    elif hasattr(inner, "n_features_in_"):
        n = inner.n_features_in_
        if X_reg.shape[1] > n:
            X_reg = X_reg.iloc[:, :n]
        elif X_reg.shape[1] < n:
            for i in range(n - X_reg.shape[1]):
                X_reg[f"_pad_{i}"] = 0.0

    log_pred      = float(reg_pipeline.predict(X_reg)[0])
    predicted_gbp = float(np.expm1(max(log_pred, 0)))
    is_high       = predicted_gbp > reg_cap_value

    if predicted_gbp < 200:
        tier, tier_color = "Low",        "#6b8aaa"
    elif predicted_gbp < 800:
        tier, tier_color = "Medium",     "#1a56a0"
    elif predicted_gbp < 2000:
        tier, tier_color = "High",       "#27ae60"
    elif predicted_gbp < reg_cap_value:
        tier, tier_color = "Premium",    "#c97a1a"
    else:
        tier, tier_color = "Enterprise", "#c0392b"

    return {
        "predicted_revenue"    : round(predicted_gbp, 2),
        "predicted_revenue_fmt": f"£{predicted_gbp:,.0f}",
        "tier"                 : tier,
        "tier_color"           : tier_color,
        "is_high_spender"      : bool(is_high),
        "cap_value"            : round(reg_cap_value, 0),
        "cap_value_fmt"        : f"£{reg_cap_value:,.0f}",
        "model_name"           : reg_model_name,
        "r2"                   : round(reg_metrics.get("r2_capped",    0), 4),
        "medae"                : round(reg_metrics.get("medae_capped", 0), 0),
        "mae"                  : round(reg_metrics.get("mae_capped",   0), 0),
    }


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def index():
    return render_template("index.html",
                           cluster_available=CLUSTER_AVAILABLE,
                           regression_available=REGRESSION_AVAILABLE)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON"}), 400
        print(f"\n[CHURN] tenure={data.get('tenure')}  season={data.get('season')}  "
              f"spending={data.get('spending_cat')}  freq={data.get('frequency')}")
        result = run_churn(_build_customer_dict(data))
        print(f"        → {result['probability']}%  [{result['label']}]")
        return jsonify({"success": True, "result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict_revenue", methods=["POST", "OPTIONS"])
def predict_revenue_route():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not REGRESSION_AVAILABLE:
        return jsonify({"success": False,
                        "error": "Regression model not loaded. Run: python src/regression.py"}), 503
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON"}), 400
        print(f"\n[REVENUE] tenure={data.get('tenure')}  freq={data.get('frequency')}  "
              f"spending={data.get('spending_cat')}")
        result = run_revenue(_build_customer_dict(data))
        print(f"          → {result['predicted_revenue_fmt']}  [{result['tier']}]")
        return jsonify({"success": True, "result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/debug", methods=["POST", "OPTIONS"])
def debug():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data     = request.get_json(force=True, silent=True) or {}
        customer = _build_customer_dict(data)
        df_m, df_s = _preprocess_input(pd.DataFrame([customer]))
        churn_prob = float(pipeline.predict_proba(df_m)[0][1])
        rev_result = run_revenue(customer) if REGRESSION_AVAILABLE else None
        key_cols   = ["CustomerTenureDays","Frequency","SpendingCategory",
                      "AvgQuantityPerTransaction","UniqueProducts","MonetaryTotal",
                      "EngagementScore","FavoriteSeason_Hiver","FavoriteSeason_Printemps"]
        return jsonify({
            "received" : {k: data.get(k) for k in
                          ["spending_cat","frequency","avg_qty","tenure","season","monetary"]},
            "scaled"   : {c: round(float(df_m[c].values[0]),4)
                          for c in key_cols if c in df_m.columns},
            "churn"    : {"probability_pct": round(churn_prob*100,2),
                          "prediction": "Churn" if churn_prob >= THRESHOLD else "Fidèle"},
            "revenue"  : rev_result,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)