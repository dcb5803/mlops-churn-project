from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import numpy as np
import os

# Load MLflow Tracking and Model Registry details
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "Churn_RF_Model")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API")

# Placeholder for the loaded model
model = None

# Input data schema for the API
class ChurnFeatures(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float

@app.on_event("startup")
def load_model():
    """Load the latest Production-staged model from MLflow Registry."""
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # Load the model using the registered name and stage
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded model from {model_uri}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.get("/")
def home():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: ChurnFeatures):
    """Predict churn probability based on input features."""
    if model is None:
        return {"error": "Model not loaded"}, 500
    
    # Convert input to DataFrame for the model
    data = features.dict()
    input_df = pd.DataFrame([data])
    
    # Make prediction (returns array of 0 or 1)
    prediction = model.predict(input_df).tolist()
    
    return {"prediction": prediction[0]}
