# Quick Start

- **Docker:** Start local services used by the project (e.g. LocalStack) with Docker Compose.

	- From the repository root (PowerShell):

		```powershell
		docker compose -f .\docker-compose.yml up -d
		```

**Pipeline Scripts Overview**

The project includes a modular ML pipeline for processing EV charging data, performing feature engineering, and training time-series forecasting models. Scripts are located in `src/pipeline/`:

1. **`ingest.py`** — Data ingestion & preprocessing
   - Loads raw CSV data (semicolon-separated, European format with comma decimals)
   - Parses datetime columns (`Start_plugin`, `End_plugout`)
   - Saves cleaned data as Parquet (compressed with Snappy)
   - Optionally uploads to LocalStack S3 (`s3://ev-data/parquets/ingest.parquet`)
   
   ```powershell
   python .\src\pipeline\ingest.py --csv ".\data\downloaded\Dataset 2_Hourly EV loads - Per user.csv" --output .\data\raw --upload
   ```

2. **`features.py`** — Feature engineering & cleaning
   - **Cleaning:** Standardizes column names, fixes missing `end_plugout`/`duration_hours` using median charging rate, corrects duration mismatches
   - **Aggregation:** Groups sessions by hour, computes `total_kwh`, session counts, and averages
   - **Feature engineering:**
     - Lag features: `lag_1`, `lag_24`, `lag_168` (1h, 1d, 1w)
     - Rolling statistics: 3h/6h/24h/168h rolling means & std
     - Cyclical encoding: sine/cosine transforms for hour/day/month
     - Temporal features: `hour_of_day`, `day_of_week`, `month`, `is_weekend`
     - Expanding mean per (hour, day-of-week) combination
   - Saves cleaned data to `data/clean/clean.parquet` and features to `data/features/features.parquet`
   - Uploads features to S3
   
   ```powershell
   python .\src\pipeline\features.py --input "s3://ev-data/parquets/ingest.parquet"
   # or local: --input ".\data\raw\ingest.parquet"
   ```

3. **`train.py`** — Model training with MLflow logging
   - Loads engineered features from Parquet (local or S3)
   - Trains one of four models: Linear Regression (`lr`), Decision Tree (`dt`), XGBoost (`xgb`), LightGBM (`lgb`)
   - Splits data using last 30 days (720 hours) as test set
   - Logs metrics (MAE, RMSE) and model artifacts to MLflow
   - Saves model locally (`src/models/`) and uploads to S3 (`s3://ev-data/artifacts/model/`)
   
   ```powershell
   # Train LightGBM model
   python .\src\pipeline\train.py --input ".\data\features\features.parquet" --model lgb --mlflow_uri http://localhost:5000
   
   # Train XGBoost
   python .\src\pipeline\train.py --input "s3://ev-data/parquets/features.parquet" --model xgb
   ```

4. **`eval.py`** — Model evaluation & reporting
   - Loads trained `.joblib` model and test data (local or S3)
   - Generates predictions and computes metrics: MAE, RMSE, R²
   - Creates visualizations:
     - **Predictions vs Actuals** plot (time series comparison)
     - **Residuals plot** (scatter plot of residuals vs predictions)
   - Saves evaluation report as `evaluation_report.txt` in `src/reports/<model_name>/`
   - Logs metrics to MLflow experiment `evaluations`
   
   ```powershell
   python .\src\pipeline\eval.py --model ".\src\models\xgb_model_20251117_2050.joblib" --test-data ".\data\features\features.parquet"
   
   # Or from S3:
   python .\src\pipeline\eval.py --model ".\src\models\lgb_model_20251117_2050.joblib" --test-data "s3://ev-data/parquets/features.parquet"
   ```

5. **`run_pipeline.py`** — End-to-end pipeline orchestrator
   - Orchestrates the complete ML workflow from feature engineering to model evaluation
   - Reads configuration from `config.yaml` (data paths, models to train)
   - **Step 1:** Runs feature engineering (`features.py`)
   - **Step 2:** Trains all configured models sequentially (`train.py` for each model)
   - **Step 3:** Evaluates all trained models (`eval.py` for each)
   - Handles UTF-8 encoding, provides progress logging, error handling
   - Returns summary: models trained/evaluated counts
   
   ```powershell
   cd .\src\pipeline
   python run_pipeline.py
   ```

---

## Deployment & Inference

### Real-Time API (FastAPI)

**`src/api/app.py`** — REST API for real-time predictions

- **Framework:** FastAPI with Uvicorn ASGI server
- **Endpoints:**
  - `GET /health` — Health check endpoint
  - `POST /predict` — Real-time prediction endpoint
- **Model Loading:** Loads best model from `src/models/registry.json` on startup
- **Request Format:** JSON with `instances` array containing feature dictionaries
- **Response:** JSON with `predictions` array

**Start the API server:**

```powershell
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Test the prediction endpoint:**

```powershell
$body = @{
    instances = @(
        @{
            n_sessions_lag1  = 5
            avg_kwh_lag1     = 12.3
            hour_of_day      = 10
            day_of_week      = 2
            month            = 5
            hour_sin         = 0.5
            hour_cos         = 0.8
            dow_sin          = 0.3
            dow_cos          = 0.95
            month_sin        = 0.1
            month_cos        = 0.99
            lag_1            = 20.0
            lag_24           = 18.0
            lag_168          = 22.0
            diff_lag1        = 1.0
            roll_mean_3h     = 19.0
            roll_mean_6h     = 18.5
            roll_mean_24h    = 21.0
            roll_std_24h     = 3.0
            roll_mean_168h   = 20.5
            hour_dow_mean    = 19.5
            is_weekend       = 0
        }
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body
```

**How it works:**
1. Uvicorn accepts HTTP requests on port 8000
2. FastAPI routes requests to appropriate endpoints (`/health`, `/predict`)
3. The `/predict` endpoint loads features, runs model inference, returns predictions

---

### Batch Inference (AWS Lambda)

**`src/aws/lambda_infer.py`** — Batch inference handler for AWS Lambda

- **Trigger:** S3 event (new Parquet file uploaded to `s3://ev-data/raw/`)
- **Process:**
  1. Reads input features from S3 Parquet file
  2. Loads trained model from `src/models/` (best model from registry)
  3. Generates predictions for entire batch
  4. Saves predictions as Parquet to `s3://ev-data/predictions/`
- **Testing:** Manual invocation simulates S3 event for local testing

**Upload input features to S3 (LocalStack):**

```powershell
awslocal s3 cp infer_input.parquet s3://ev-data/raw/infer_input.parquet
```

**Run batch inference locally (simulates Lambda):**

```powershell
python src/aws/lambda_infer.py
```

**Check output predictions:**

```powershell
awslocal s3 ls s3://ev-data/predictions/ --recursive
```

**Note:** In this iteration, Lambda logic is tested via manual invocation (simulated S3 event). For production, deploy to AWS Lambda with S3 trigger configuration.

---

## Note: MLflow → File/JSON Driven Pipeline

MLflow was used during initial experimentation for tracking metrics and model artifacts. Moving forward, the pipeline is transitioning to a **file/JSON driven approach**:
- Model metadata stored in `src/models/registry.json`
- Evaluation metrics saved as JSON files in `src/reports/<model_name>/metrics.json`
- This simplifies deployment and removes MLflow server dependency for production workflows