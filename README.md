
# Retail ML Project

A machine learning pipeline for retail customer analysis, segmentation, and churn prediction, featuring data preprocessing, clustering, classification, monitoring, and a Flask web app for predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Web Application](#web-application)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This repository provides a full ML workflow for retail data, including:
- Data preprocessing and cleaning
- Customer segmentation using K-Means clustering
- Churn prediction with classification models
- Model monitoring and drift detection
- Interactive Flask web app for predictions

## Project Structure

- `data/` — Raw, processed, and train/test datasets
- `src/` — Source code for preprocessing, clustering, training, prediction, monitoring, and utilities
- `app/` — Flask web application and HTML templates
- `notebooks/` — Jupyter notebooks for EDA and reporting
- `models/` — Saved models and artifacts
- `logs/` — Log files
- `reports/` — Generated reports and summaries
- `requirements.txt` — Python dependencies
- `main.py` — Pipeline runner script

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/projet_ml_retail.git
    cd projet_ml_retail
    ```
2. **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Unix/Mac:
    source venv/bin/activate
    ```
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Run the Full Pipeline

```bash
python main.py
```



### Pipeline Steps

1. **Preprocessing:** Cleans and prepares data (`src/preprocessing.py`)
2. **Clustering:** Segments customers using K-Means (`src/clustering.py`)
3. **Classification:** Trains churn prediction model (`src/train_model.py`)
4. **Testing:** Runs tests (pytest)
5. **Monitoring:** Monitors model/data drift (`src/monitoring.py`)
6. **Prediction:** Runs batch predictions (`src/predict.py`)
7. **Web App:** Launches Flask app (`app/app.py`)

### Web Application

Start the Flask app (after running the pipeline or loading models):

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