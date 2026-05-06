
# Retail ML Project — Customer Segmentation, Churn & Revenue Forecasting

End-to-end machine learning project for retail customer intelligence:

- **Preprocessing** (feature engineering, encoding, scaling, PCA)
- **Customer segmentation** (K-Means)
- **Churn prediction** (calibrated classifier + tuned decision threshold)
- **Revenue forecasting** (regression on MonetaryTotal with outlier handling)
- **Monitoring & drift detection** (Evidently when available, fallback otherwise)
- **Flask web app** for interactive churn + revenue predictions

> Note: This repository contains a ready-to-run pipeline driven by [main.py](main.py). Most scripts currently read/write from the project folders directly (e.g. `data/train_test`, `models`, `reports`).

## Table of contents

- [Quickstart](#quickstart)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Run the pipeline](#run-the-pipeline)
    - [Run specific steps](#run-specific-steps)
    - [Monitoring](#monitoring)
    - [Run the demo predictor](#run-the-demo-predictor)
- [Flask web app](#flask-web-app)
- [Artifacts (what gets saved)](#artifacts-what-gets-saved)
- [Configuration](#configuration)
- [Notebooks & reports](#notebooks--reports)
- [Troubleshooting](#troubleshooting)

## Quickstart

```bash
cd /path/to/projet_ml_retail

python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run the core pipeline (preprocessing + classification)
python main.py

# Launch the web app (requires trained artifacts in models/)
python app/app.py
```

Open: http://localhost:5000

## Project structure

```
.
├── app/                       # Flask app + HTML template
│   ├── app.py
│   └── templates/
│       └── index.html
├── config.yaml                # Central config (logging, defaults, metadata)
├── data/
│   ├── raw/                   # Input CSV
│   ├── processed/             # Optional outputs (e.g., cluster labels)
│   └── train_test/            # Saved train/test splits (raw + PCA)
├── logs/                      # Pipeline logs
├── models/                    # Serialized artifacts (joblib)
├── notebooks/                 # EDA notebooks
├── reports/                   # Plots, metrics, HTML monitoring reports
├── src/                       # Pipeline scripts
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── clustering.py
│   ├── regression.py
│   ├── predict.py
│   ├── monitoring.py
│   └── utils.py
├── main.py                    # Pipeline runner (step orchestrator)
└── requirements.txt
```

## Setup

### Requirements

- Python **3.10+** recommended
- Windows / macOS / Linux

### Install

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows cmd.exe
venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Input data

By default the preprocessing step reads:

- `data/raw/retail_customers_COMPLETE_CATEGORICAL.csv`

If you replace the dataset, keep the same feature columns (or update the scripts accordingly).

## Run the pipeline

The pipeline runner is [main.py](main.py). It can run everything or selected steps.

### Full run (recommended)

```bash
python main.py
```

Default step sequence:

1. Preprocessing → `src/preprocessing.py`
2. Clustering (optional) → `src/clustering.py`
3. Classification (churn) → `src/train_model.py`
4. Regression (optional) → `src/regression.py`
5. Tests (optional) → `pytest tests/`
6. Monitoring (optional) → `src/monitoring.py`
7. Predict demo (optional) → `src/predict.py`
8. Flask app (optional) → `app/app.py`

### Run specific steps

Use `--steps` with numeric step IDs:

```bash
# preprocessing + classification only
python main.py --steps 1,3

# clustering + churn training + revenue regression
python main.py --steps 2,3,4
```

Useful flags:

```bash
python main.py --no-flask          # skip Flask app step
python main.py --no-regression     # skip regression step
python main.py --monitor           # include monitoring step
python main.py --test              # include pytest step (requires a tests/ folder)
python main.py --skip-on-fail      # continue even if a required step fails
python main.py --mlflow            # use MLflow training script if present
python main.py --mlflow-ui         # open MLflow UI after the pipeline
```

### Monitoring

```bash
python src/monitoring.py
```

Monitoring tries to use **Evidently** to generate HTML reports. If Evidently is not installed, the script falls back to a simplified KS-test drift report.

Optional install:

```bash
python -m pip install evidently
```

To simulate “production” drift, place:

- `data/production/new_customers.csv`
- `data/production/new_customers_labels.csv`

### Run the demo predictor

After training (and generating artifacts in `models/`), you can run:

```bash
python src/predict.py
```

This script loads `models/churn_model.pkl` + preprocessing artifacts, applies the tuned threshold from `models/threshold.pkl`, and prints prediction details.

## Flask web app

The app is in [app/app.py](app/app.py) and serves the UI in `app/templates/index.html`.

### Start the server

```bash
python app/app.py
```

Then open: http://localhost:5000

> The app expects trained artifacts in `models/`. Run `python main.py` first.

### API endpoints

- `GET /` → HTML UI
- `POST /predict` → churn prediction (probability + decision)
- `POST /predict_revenue` → revenue forecast (requires `models/regression_model.pkl`)
- `POST /debug` → returns a trace of the preprocessing/prediction path

Example request:

```bash
python run_part.py --steps clustering,predict
```

This allows you to control the execution order as needed.

    #### Options principales


    - `--no-flask` — Exécute le pipeline sans lancer Flask
    - `--monitor` — Monitoring uniquement
    - `--steps 1,3,4` — Exécute des étapes spécifiques (voir ci-dessous)
    - `--skip-on-fail` — Continue même si une étape échoue

    ### Application web

    Lancer l’application Flask (après le pipeline ou chargement des modèles) :
    ```bash
    python app/app.py
    ```

- Access the web interface at [http://localhost:5000](http://localhost:5000)
- Submit customer data for churn prediction and cluster assignment

## Requirements

See `requirements.txt` for the full list. Key packages include:
- pandas, numpy, scikit-learn, matplotlib, seaborn
- Flask, joblib, imbalanced-learn

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.