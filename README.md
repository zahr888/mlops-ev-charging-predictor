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

## Note: MLflow → File/JSON Driven Pipeline

MLflow was used during initial experimentation for tracking metrics and model artifacts. Moving forward, the pipeline is transitioning to a **file/JSON driven approach**:
- Model metadata stored in `src/models/registry.json`
- Evaluation metrics saved as JSON files in `src/reports/<model_name>/metrics.json`
- This simplifies deployment and removes MLflow server dependency for production workflows