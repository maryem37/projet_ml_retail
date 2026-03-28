# ==========================================
# RETAIL ML PROJECT - FLASK APPLICATION
# ==========================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ==========================================
# LOAD MODELS AT STARTUP
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model  = joblib.load(os.path.join(BASE_DIR, "models", "churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# Load clustering model if available
CLUSTER_AVAILABLE = False
kmeans = None
try:
    kmeans = joblib.load(os.path.join(BASE_DIR, "models", "kmeans_model.pkl"))
    CLUSTER_AVAILABLE = True
    print("  ✅ kmeans_model.pkl loaded")
except FileNotFoundError:
    print("  ⚠️  kmeans_model.pkl not found — clustering disabled")

X_train_ref      = pd.read_csv(os.path.join(BASE_DIR, "data", "train_test", "X_train.csv"), nrows=1)
expected_columns = X_train_ref.columns.tolist()
scaler_columns   = list(scaler.feature_names_in_)

# Cluster names based on our analysis
CLUSTER_NAMES = {
    0: {"label": "At Risk",      "emoji": "🟠", "color": "#e67e22"},
    1: {"label": "Active Loyal", "emoji": "🟢", "color": "#27ae60"},
    2: {"label": "Lost",         "emoji": "🔴", "color": "#c0392b"},
    3: {"label": "Occasional",   "emoji": "🔵", "color": "#2c5f8a"},
}


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_churn(input_dict):
    df = pd.DataFrame([input_dict])

    categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]

    cols_to_scale = [c for c in scaler_columns if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # Cluster prediction
    cluster_id   = None
    cluster_info = None
    if CLUSTER_AVAILABLE:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        X_num        = df[numeric_cols].fillna(0)

        # Align to kmeans feature count
        if X_num.shape[1] >= kmeans.n_features_in_:
            X_num = X_num.iloc[:, :kmeans.n_features_in_]
        else:
            for _ in range(kmeans.n_features_in_ - X_num.shape[1]):
                X_num[f"pad_{_}"] = 0

        cluster_id   = int(kmeans.predict(X_num)[0])
        cluster_info = CLUSTER_NAMES.get(cluster_id, {
            "label": f"Cluster {cluster_id}", "emoji": "⚪", "color": "#888"
        })

    # Risk level
    if probability < 0.25:
        risk = "Faible"
    elif probability < 0.50:
        risk = "Moyen"
    elif probability < 0.75:
        risk = "Élevé"
    else:
        risk = "Critique"

    return {
        "prediction"   : int(prediction),
        "label"        : "Churn" if prediction == 1 else "Fidèle",
        "probability"  : round(float(probability) * 100, 1),
        "risk_level"   : risk,
        "cluster_id"   : cluster_id,
        "cluster_label": cluster_info["label"] if cluster_info else None,
        "cluster_emoji": cluster_info["emoji"] if cluster_info else None,
        "cluster_color": cluster_info["color"] if cluster_info else None,
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
            "Recency"                 : float(data.get("recency", 0)),
            "Frequency"               : float(data.get("frequency", 1)),
            "MonetaryTotal"           : float(data.get("monetary", 0)),
            "MonetaryAvg"             : float(data.get("monetary_avg", 0)),
            "MonetaryStd"             : float(data.get("monetary_std", 0)),
            "MonetaryMin"             : float(data.get("monetary_min", 0)),
            "MonetaryMax"             : float(data.get("monetary_max", 0)),
            "TotalQuantity"           : float(data.get("total_quantity", 0)),
            "AvgQtyPerTransaction"    : float(data.get("avg_qty", 1)),
            "MinQuantity"             : float(data.get("min_qty", 1)),
            "MaxQuantity"             : float(data.get("max_qty", 1)),
            "CustomerTenureDays"      : float(data.get("tenure", 0)),
            "FirstPurchaseDaysAgo"    : float(data.get("first_purchase", 0)),
            "PreferredDayOfWeek"      : float(data.get("preferred_day", 0)),
            "PreferredHour"           : float(data.get("preferred_hour", 12)),
            "PreferredMonth"          : float(data.get("preferred_month", 6)),
            "WeekendRatio"            : float(data.get("weekend_ratio", 0.3)),
            "AvgDaysBetweenPurchases" : float(data.get("avg_days_between", 30)),
            "UniqueProducts"          : float(data.get("unique_products", 1)),
            "AvgProductsPerTrans"     : float(data.get("avg_products", 1)),
            "UniqueCountries"         : float(data.get("unique_countries", 1)),
            "CancelledTransactions"   : float(data.get("cancelled", 0)),
            "ReturnRatio"             : float(data.get("return_ratio", 0)),
            "TotalTransactionLines"   : float(data.get("total_lines", 1)),
            "AvgLinesPerInvoice"      : float(data.get("avg_lines", 1)),
            "Age"                     : float(data.get("age", 35)),
            "Age_IsMissing"           : 0,
            "SupportTickets"          : float(data.get("support_tickets", 0)),
            "Satisfaction"            : float(data.get("satisfaction", 3)),
            "SpendingCategory"        : float(data.get("spending_cat", 1)),
            "AgeCategory"             : float(data.get("age_cat", 2)),
            "BasketSizeCategory"      : float(data.get("basket_size", 1)),
            "PreferredTimeOfDay"      : float(data.get("preferred_time", 2)),
            "CustomerType"            : data.get("customer_type", "Régulier"),
            "FavoriteSeason"          : data.get("season", "Hiver"),
            "Region"                  : data.get("region", "UK"),
            "WeekendPreference"       : data.get("weekend_pref", "Semaine"),
            "ProductDiversity"        : data.get("prod_diversity", "Modéré"),
            "Gender"                  : data.get("gender", "M"),
            "AccountStatus"           : data.get("account_status", "Active"),
            "Country"                 : data.get("country", "United Kingdom"),
        }

        result = predict_churn(customer)
        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)