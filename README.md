
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

    # Retail ML Project

    Un pipeline complet de machine learning pour l’analyse, la segmentation et la prédiction de churn des clients du secteur retail. Ce projet inclut le prétraitement des données, le clustering, la classification, le monitoring, ainsi qu’une application web Flask pour des prédictions interactives.

    ---

    ## Table des matières

    - [Présentation du projet](#présentation-du-projet)
    - [Structure du projet](#structure-du-projet)
    - [Installation](#installation)
    - [Utilisation](#utilisation)
      - [Exécution du pipeline](#exécution-du-pipeline)
      - [Application web](#application-web)
    - [Étapes du pipeline](#étapes-du-pipeline)
    - [Dépendances](#dépendances)
    - [Contribuer](#contribuer)
    - [Licence](#licence)

    ---

    ## Présentation du projet

    Ce dépôt propose un workflow ML complet pour la donnée retail, incluant :

    - Prétraitement et nettoyage des données
    - Segmentation client par clustering K-Means
    - Prédiction du churn par modèles de classification
    - Monitoring du modèle et détection de dérive
    - Application web Flask interactive pour la prédiction

    ---

    ## Structure du projet

    ```
    .
    ├── app/
    │   ├── app.py
    │   └── templates/
    │       └── index.html
    ├── config.yaml
    ├── data/
    │   ├── processed/
    │   ├── raw/
    │   └── train_test/
    ├── logs/
    ├── main.py
    ├── models/
    ├── notebooks/
    │   ├── 01_EDA.ipynb
    │   ├── EDA_Churn_Exploration.ipynb
    │   └── reports/
    ├── reports/
    ├── requirements.txt
    ├── src/
    │   ├── clustering.py
    │   ├── config_loader.py
    │   ├── find_leaky_features.py
    │   ├── monitoring.py
    │   ├── predict.py
    │   ├── preprocessing.py
    │   ├── train_model.py
    │   └── utils.py
    └── tools/
    ```

    ---

    ## Installation

    1. **Cloner le dépôt :**
        ```bash
        git clone https://github.com/yourusername/projet_ml_retail.git
        cd projet_ml_retail
        ```

    2. **Créer et activer un environnement virtuel (recommandé) :**
        ```bash
        python -m venv venv
        # Sous Windows :
        venv\Scripts\activate
        # Sous Unix/Mac :
        source venv/bin/activate
        ```

    3. **Installer les dépendances :**
        ```bash
        pip install -r requirements.txt
        ```

    ---

    ## Utilisation

    ### Exécution du pipeline

    Lancer le pipeline principal :
    ```bash
    python main.py
    ```

    #### Options principales

    - `--no-flask` — Exécute le pipeline sans lancer Flask
    - `--mlflow` — Active le tracking MLflow
    - `--test` — Lance les tests
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