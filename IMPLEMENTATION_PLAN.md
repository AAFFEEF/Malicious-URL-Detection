# Malicious URL Detection Project - Implementation Plan

## Project Overview
This project implements a Machine Learning pipeline for detecting malicious URLs, following a 6-phase development lifecycle.

## Project Structure
```
Malicious URL Detection/
├── data/
│   ├── raw/                # Raw datasets (PhishTank, etc.)
│   └── processed/          # Cleaned and feature-enriched data
├── models/                 # Saved models
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration parameters
│   ├── data_ingestion.py   # Phase 1: Ingestion & Validation
│   ├── feature_extraction.py # Phase 2: Lexical, Host, Content features
│   ├── feature_selection.py # Phase 3: GA/PSO Optimization mocks/wrappers
│   ├── model_training.py   # Phase 4: RF, XGBoost, Deep Learning
│   └── evaluation.py       # Phase 5: Metrics & Benchmarking
├── app.py                  # Phase 6: FastAPI Deployment
├── requirements.txt        # Dependencies
└── README.md
```

## Implementation Phases

### Phase 1: Data Ingestion and Validation
- **Goal**: Load raw data, normalize URLs, and validate schema.
- **Tech**: Pandas for data handling.
- **File**: `src/data_ingestion.py`

### Phase 2: Feature Engineering
- **Goal**: Extract Lexical, Host-based, and Content-based features.
- **Tech**: `urllib`, `tldextract`, `python-whois`.
- **File**: `src/feature_extraction.py`

### Phase 3: Feature Selection
- **Goal**: Optimize feature set.
- **Tech**: Scikit-learn (SelectKBest/RFE) and structure for Metaheuristics.
- **File**: `src/feature_selection.py`

### Phase 4: Model Development
- **Goal**: Train ML (RF/XGB) and DL (CNN/LSTM) models.
- **Tech**: Scikit-learn, TensorFlow/Keras.
- **File**: `src/model_training.py`

### Phase 5: Evaluation
- **Goal**: Measure Accuracy, Precision, Recall, F1, and TPR at low FPR.
- **Tech**: Scikit-learn metrics, Matplotlib (for plots).
- **File**: `src/evaluation.py`

### Phase 6: Deployment
- **Goal**: Real-time inference API.
- **Tech**: FastAPI, Uvicorn.
- **File**: `app.py`
