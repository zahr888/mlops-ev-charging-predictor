# EV Charging Demand MLOps Pipeline

An end-to-end MLOps project for forecasting hourly EV charging demand. This repository implements a complete lifecycle: data engineering, model training, experiment tracking, registry management, real-time inference API, and batch processing using a modern stack.

## 🏗 Architecture Overview

*   **Data Pipeline:** Ingestion, cleaning, and feature engineering (lags, rolling stats, cyclical features).
*   **Training Pipeline:** Training multiple models (XGBoost, LightGBM, etc.), automatic evaluation, and artifact logging.
*   **Model Registry:** File-based registry that automatically promotes the best performing model to production.
*   **Serving (API):** Dockerized FastAPI service for real-time inference.
*   **Batch Processing:** Lambda-style handler for high-throughput offline predictions via S3.
*   **CI/CD:** GitHub Actions pipeline for automated testing and integration.
*   **Monitoring:** Prometheus & Grafana stack for tracking API metrics and model performance.

***

## 🚀 Quick Start

### Prerequisites
*   Docker & Docker Compose
*   Python 3.11+ (if running locally without Docker)

### Run the Full Stack (Recommended)
Start the API, Prometheus, Grafana, LocalStack (AWS emulation), and MLflow with one command:

```powershell
docker compose up --build -d
```

*   **API Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
*   **Grafana Dashboards:** [http://localhost:3000](http://localhost:3000) (Login: `admin` / `admin`)
*   **Prometheus:** [http://localhost:9090](http://localhost:9090)

***

## 🛠️ Pipeline Scripts (`src/pipeline/`)

The project uses a modular pipeline controlled by `run_pipeline.py`.

**1. Orchestrator (`run_pipeline.py`)**
Runs the full workflow: Feature Engineering → Training (All Models) → Evaluation → Registry Update.
```powershell
python src/pipeline/run_pipeline.py
```

**2. Individual Stages**
*   **`ingest.py`**: Loads raw CSVs, parses dates, saves as Parquet.
*   **`features.py`**: Generates lag/rolling features and aggregates to hourly level.
*   **`train.py`**: Trains models (Linear Regression, Decision Tree, XGBoost, LightGBM) and logs to MLflow.
*   **`eval.py`**: Generates metrics (MAE, RMSE, R²) and plots; saves results to `src/reports/`.
*   **`update_registry.py`**: Scans evaluation reports and updates `src/models/registry.json` with the best model.

***

## 🤖 Deployment & Inference

### 1. Real-Time API (FastAPI + Docker)
The API loads the production model defined in `registry.json` and serves predictions.

*   **Code:** `src/api/app.py`
*   **Container:** Dockerized using `python:3.11-slim`.
*   **Monitoring:** Instrumented with `prometheus-fastapi-instrumentator`.

**Test Prediction (PowerShell):**
```powershell
$body = @{
    instances = @(
        @{
            n_sessions_lag1 = 5; avg_kwh_lag1 = 12.3; hour_of_day = 10; day_of_week = 2; month = 5;
            is_weekend = 0; hour_sin = 0.5; hour_cos = 0.8; dow_sin = 0.3; dow_cos = 0.95;
            month_sin = 0.1; month_cos = 0.99; lag_1 = 20.0; lag_24 = 18.0; lag_168 = 22.0;
            diff_lag1 = 1.0; roll_mean_3h = 19.0; roll_mean_6h = 18.5; roll_mean_24h = 21.0;
            roll_std_24h = 3.0; roll_mean_168h = 20.5; hour_dow_mean = 19.5
        }
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body
```

### 2. Batch Inference (AWS Lambda Pattern)
Simulates a serverless workflow where uploading data to S3 triggers inference.

*   **Code:** `src/aws/lambda_infer.py`
*   **Trigger:** S3 Object Create event in `s3://ev-data/raw/`.
*   **Output:** Saves predictions to `s3://ev-data/predictions/`.

**Manual Test (LocalStack):**
```powershell
# Upload input file
awslocal s3 cp infer_input.parquet s3://ev-data/raw/infer_input.parquet

# Run handler manually (simulates Lambda trigger)
python src/aws/lambda_infer.py
```

***

## 📊 Monitoring & Observability

The system implements a full observability stack running in Docker:

1.  **Instrumentation:** The FastAPI app exposes custom metrics at `/metrics` (Request Rate, Prediction Count, Latency).
2.  **Prometheus:** Scrapes the API every 15 seconds.
3.  **Grafana:** Visualizes these metrics in real-time dashboards.

**To view metrics:**
1.  Generate traffic (run the API test loop).
2.  Open Grafana (`localhost:3000`).
3.  Query `rate(http_requests_total[1m])` or `ev_predictions_total`.

***

## 🔄 CI/CD (GitHub Actions)

The project includes a Continuous Integration pipeline (`.github/workflows/ci.yml`) that runs on every push/PR to `main`.

**Pipeline Steps:**
1.  **Environment Setup:** Sets up Python 3.11 on Ubuntu runners.
2.  **Dependency Installation:** Installs requirements + test dependencies (`httpx`, `pytest`).
3.  **Integration Testing:**
    *   Resolves python paths for the `src` module.
    *   Validates the Docker-compatible model loading logic (Windows paths on Linux runners).
    *   Runs `pytest` to verify the API health endpoint and model loading.

***

## 🔮 Future Roadmap

*   **Continuous Deployment (CD):** Automate deployment to AWS Lambda or ECS upon passing CI.
*   **Retraining Pipeline:** Implement a trigger to retrain models automatically when new data arrives or performance degrades (Drift Detection).
*   **Frontend:** Add a simple Streamlit or React dashboard for visualizing forecasts interactively.
*   **Feature Store:** Migration to a formal Feature Store (e.g., Feast) for managing training/inference consistency.