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
# LOAD MODEL & SCALER AT STARTUP
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model  = joblib.load(os.path.join(BASE_DIR, "models", "churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

X_train_ref      = pd.read_csv(os.path.join(BASE_DIR, "data", "train_test", "X_train.csv"), nrows=1)
expected_columns = X_train_ref.columns.tolist()
scaler_columns   = list(scaler.feature_names_in_)


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

    if probability < 0.25:
        risk = "Faible"
        risk_en = "Low"
    elif probability < 0.50:
        risk = "Moyen"
        risk_en = "Medium"
    elif probability < 0.75:
        risk = "Élevé"
        risk_en = "High"
    else:
        risk = "Critique"
        risk_en = "Critical"

    return {
        "prediction" : int(prediction),
        "label"      : "Churn" if prediction == 1 else "Fidèle",
        "probability": round(float(probability) * 100, 1),
        "risk_level" : risk,
        "risk_en"    : risk_en,
    }


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


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