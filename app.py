from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import joblib
import pandas as pd
import uvicorn
import os
from src.feature_extraction import FeatureExtractor

# Initialize App
app = FastAPI(title="Malicious URL Detection API", version="1.0")

# CORS (Optional if serving from same origin, but good practice)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Artifacts
MODEL_PATH = "models/rf_model.pkl"
FEATURES_PATH = "models/feature_list.pkl"

model = None
feature_list = None

@app.on_event("startup")
def load_artifacts():
    global model, feature_list
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        feature_list = joblib.load(FEATURES_PATH)
        print("Model and features loaded successfully.")
    else:
        print("Model not found. Please run src/run_pipeline.py first.")

class URLRequest(BaseModel):
    url: str

    @validator('url')
    def validate_url(cls, v):
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        if '.' not in v:
            raise ValueError('Invalid URL format (must contain a domain)')
        if len(v) < 4:
            raise ValueError('URL is too short')
        return v

# Serve Frontend
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Mount static for assets if needed (not strictly needed for single file but good practice)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
def predict(request: URLRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    url = request.url
    
    # Create DataFrame for processing
    df = pd.DataFrame({"url": [url], "label": [0]}) # Dummy label
    
    # 1. Feature Engineering (Reuse the logic!)
    extractor = FeatureExtractor(df)
    df_features = extractor.run_all()
    
    # 2. Select columns matching training
    try:
        X_input = df_features[feature_list]
    except KeyError as e:
        # Handle case where features might be missing (though extractor should produce them all)
        # Or if training dropped some that are essentially generated (like 'url' column itself is excluded from feature_list)
        missing = [c for c in feature_list if c not in df_features.columns]
        if missing:
             raise HTTPException(status_code=500, detail=f"Feature extraction mistmatch. Missing: {missing}")
        X_input = df_features[feature_list]

    # 3. Predict
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]
    
    result = "MALICIOUS" if prediction == 1 else "LEGITIMATE"
    
    return {
        "url": url,
        "prediction": result,
        "malicious_probability": float(probability)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
