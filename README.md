
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
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d "{\"tenure\":365,\"frequency\":5,\"monetary\":350,\"spending_cat\":\"Medium\",\"season\":\"Automne\"}"
```

## Artifacts (what gets saved)

After a successful run, you should see the following artifacts:

### Classification (churn)

- `models/scaler.pkl` — StandardScaler fitted on training data
- `models/pca.pkl` — PCA transformer (used when PCA wins)
- `models/imputation_stats.pkl` — medians/means + country target encoding map
- `models/churn_model.pkl` — trained **CalibratedClassifierCV** pipeline
- `models/threshold.pkl` — tuned decision threshold + metadata (`use_pca`, etc.)

### Clustering

- `models/kmeans_model.pkl` — trained KMeans
- `reports/elbow_curve.png`, `reports/silhouette_scores.png`, etc.

### Regression (revenue)

- `models/regression_model.pkl` — dict artifact with `pipeline`, metrics and metadata
- `reports/regression_metrics.csv`, `reports/regression_target_distribution.png`, etc.

## Configuration

Project-wide settings are stored in [config.yaml](config.yaml).

Currently used by:

- `src/config_loader.py` for robust YAML loading (encoding fallbacks)
- `src/monitoring.py` for logging configuration (`logs/pipeline.log`, level/format)

Some scripts still use hard-coded paths (e.g. `data/train_test/*.csv`). If you want everything driven by `config.yaml`, the next step is to route file paths through `src/config_loader.py` across the pipeline.

## Notebooks & reports

- Notebooks live in `notebooks/` (EDA and exploration)
- Most scripts generate plots/CSVs into `reports/`
- Monitoring can generate HTML:
    - `reports/monitoring_report.html`
    - `reports/drift_report.html`

To open the notebooks:

```bash
python -m jupyter lab
```

## Troubleshooting

### “File not found” in `models/`

Run the pipeline first:

```bash
python main.py --steps 1,3
```

The Flask app and prediction scripts expect `models/churn_model.pkl`, `models/scaler.pkl`, `models/pca.pkl`, `models/imputation_stats.pkl`, and `models/threshold.pkl`.

### PowerShell cannot activate venv

If PowerShell blocks scripts, run once (as admin) and reopen PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Alternative (only for the current terminal session):

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

### Monitoring reports not generated

Install Evidently (optional):

```bash
python -m pip install evidently
```

### Pytest step fails / tests folder missing

This repo currently has no `tests/` folder. Only use the pytest step if you add tests.

---

## Notes

- This repo does not currently ship a `LICENSE` file; add one if you plan to redistribute.