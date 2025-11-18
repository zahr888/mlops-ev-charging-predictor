from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import json
import joblib
import pandas as pd

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
    model_path = prod["model_path"]
    # Make model_path absolute if it’s not
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = (MODELS_DIR / model_path.name).resolve()
    model = joblib.load(model_path)
    return model, prod

model, prod_info = load_production_model()

# --------- FastAPI Setup ---------

app = FastAPI(
    title="EV Charging Demand API",
    version="0.1.0",
    description="Serve hourly total_kwh predictions using the production model."
)

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
    df = pd.DataFrame([f.model_dump() for f in req.instances])
    df = df[FEATURE_COLUMNS]   # enforce same columns & order as training
    preds = model.predict(df)
    return PredictResponse(
        model_name=prod_info["model_name"],
        predictions=preds.tolist()
    )
