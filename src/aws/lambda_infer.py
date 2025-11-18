import boto3
import joblib
import json
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime


# S3 configuration for LocalStack

s3 = boto3.client("s3", endpoint_url="http://localhost:4566",
                  aws_access_key_id="test", aws_secret_access_key="test")
BUCKET = "ev-data"

# Path to model registry 
BASE_DIR = Path(__file__).resolve().parents[2]  # mlops/
REGISTRY_PATH = BASE_DIR / "src/models/registry.json"

def handler(event, context=None):
    # 1. Figure out S3 key from event 
    record = event['Records'][0]
    input_key = record['s3']['object']['key']  # e.g. 'raw/infer_input.parquet'

    # 2. Read features Parquet from S3 
    obj = s3.get_object(Bucket=BUCKET, Key=input_key)
    df = pd.read_parquet(BytesIO(obj['Body'].read()))
    '''
    we can't use the pd.read_parquet("s3://...") methode here 
    because it requires additional dependencies and setup for s3 access (bad for portability)
    '''

    # 3. Load latest production model from registry (local path)
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    prod = reg['production']
    model_path = prod['model_path']
    model = joblib.load(model_path)

    # 4. Build DataFrame in expected order
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
    df_features = df[FEATURE_COLUMNS]

    # 5. Predict
    preds = model.predict(df_features)
    df['predicted_total_kwh'] = preds 

    # 6. Write predictions to S3
    now = datetime.now().strftime("%Y%m%d_%H%M")
    orig_base = Path(input_key).stem  # e.g., "your_uploaded_file"
    output_key = f"predictions/{now}_{orig_base}_pred.parquet"
    output_buf = BytesIO()
    df.to_parquet(output_buf, index=False)
    output_buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=output_key, Body=output_buf.getvalue())

    print(f"Predictions written to s3://{BUCKET}/{output_key}")

    return {"result_key": output_key, "predictions_written": len(df)}

# For local test:
if __name__ == '__main__':
    # Simulate an S3 event
    fake_event = {
                    "Records": [
                        {
                        "s3": {
                            "bucket": {"name": "ev-data"},
                            "object": {"key": "raw/infer_input.parquet"}
                            }
                        }
                    ]
                }
    handler(fake_event)
