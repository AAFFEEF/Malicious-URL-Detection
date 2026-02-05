# Malicious URL Detection using Machine Learning

This project implements a complete pipeline for detecting malicious URLs, following a 6-phase development lifecycle.

## Project Architecture

The project is organized into `src/` modules corresponding to the phases:

1.  **Ingestion** (`src/data_ingestion.py`): Loads data (supports PhishTank/CSV) and normalizes it.
2.  **Feature Engineering** (`src/feature_extraction.py`): Extracts Lexical, Host-based, and Content-based features.
3.  **Feature Selection** (`src/feature_selection.py`): Optimizes the feature set using Statistical and Model-based methods.
4.  **Training** (`src/model_training.py`): Trains Random Forest / XGBoost models.
5.  **Evaluation** (`src/evaluation.py`): Calculates Accuracy, F1, and TPR at low FPR.
6.  **Deployment** (`app.py`): FastAPI service for real-time inference.

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline (Training)**:
    This script runs Phases 1 through 5, generates dummy data (if no file provided), trains the model, and saves it to `models/`.
    ```bash
    python -m src.run_pipeline
    ```

3.  **Start the API Services (Deployment)**:
    Once the model is trained, start the real-time API.
    ```bash
    uvicorn app:app --reload
    ```

4.  **Test the API**:
    Open your browser to `http://127.0.0.1:8000/docs` to test the `/predict` endpoint interactively.

## Folder Structure
- `data/`: Storage for datasets.
- `models/`: Where trained models (`.pkl`) are saved.
- `src/`: Source code for the ML pipeline.
