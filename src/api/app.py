import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import json
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
import os

from prometheus_client import Counter, Histogram

PREDICTION_COUNTER = Counter(
    "ev_predictions_total",
    "Number of prediction calls",
    ["model_name"]
)

PREDICTION_LATENCY = Histogram(
    "ev_prediction_latency_seconds",
    "Time spent in prediction endpoint"
)


BASE_DIR = Path(__file__).resolve().parents[1]  # mlops/src
MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
FEATURE_COLUMNS = [
    "n_sessions_lag1",
    "avg_kwh_lag1",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "diff_lag1",
    "roll_mean_3h",
    "roll_mean_6h",
    "roll_mean_24h",
    "roll_std_24h",
    "roll_mean_168h",
    "hour_dow_mean",
]



# --------- Registry / Model Loading ---------

def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)
    

def load_production_model():
    registry = load_registry()
    prod = registry["production"]
    raw_path = prod["model_path"]
    
    # Robust way to get filename from ANY path string (Windows or Linux)
    # Split by \ first (Windows), then by / (Linux/Mac)
    model_filename = raw_path.split("\\")[-1].split("/")[-1]
    
    # Construct the clean path
    model_path_in_container = (MODELS_DIR / model_filename).resolve()
    
    print(f"Loading model from: {model_path_in_container}")
    
    model = joblib.load(model_path_in_container)
    return model, prod

model, prod_info = load_production_model()

# --------- FastAPI Setup ---------

app = FastAPI(
    title="EV Charging Demand API",
    version="0.1.0",
    description="Serve hourly total_kwh predictions using the production model."
)
# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

class FeatureVector(BaseModel):
    n_sessions_lag1: float
    avg_kwh_lag1: float
    hour_of_day: int
    day_of_week: int
    month: int
    is_weekend: int
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    month_sin: float
    month_cos: float
    lag_1: float
    lag_24: float
    lag_168: float
    diff_lag1: float
    roll_mean_3h: float
    roll_mean_6h: float
    roll_mean_24h: float
    roll_std_24h: float
    roll_mean_168h: float
    hour_dow_mean: float

    

class PredictRequest(BaseModel):
    instances: List[FeatureVector]

class PredictResponse(BaseModel):
    model_name: str
    predictions: List[float]

# --------- API Endpoints ---------
# we can add more endpoints later like available models, metrics , features etc ...
@app.get("/health")
def health():
    return {"status": "ok", "model_name": prod_info["model_name"]}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()
    df = pd.DataFrame([f.model_dump() for f in req.instances])
    df = df[FEATURE_COLUMNS]
    preds = model.predict(df)
    duration = time.time() - start

    PREDICTION_COUNTER.labels(model_name=prod_info["model_name"]).inc()
    PREDICTION_LATENCY.observe(duration)

    return PredictResponse(
        model_name=prod_info["model_name"],
        predictions=preds.tolist()
    )
