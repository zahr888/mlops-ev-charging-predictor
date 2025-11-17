
import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from datetime import datetime
import boto3


# ----- Command-line args for flexibility -----
def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    parser.add_argument("--input", required=True, help="Path to features (local or s3)")
    parser.add_argument("--model", required=True, choices=["lr", "dt", "xgb", "lgb"], help="Which model to train")
    parser.add_argument("--output", default="C:\\Users\\GIGABYTE\\Documents\\ml\\mlops\\src\\models", help="Local model directory or S3 path")
    parser.add_argument("--mlflow_uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment", default="ev", help="MLflow experiment name")
    parser.add_argument("--bucket", default="ev-data", help="S3 bucket name for model artifacts")

    
    return parser.parse_args()

# ----- Main training logic -----
def main(args):
    print("connecting to data source...")
     # S3/LocalStack settings
    print("setting up mlflow...")
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    # Load features data
    print(f"Loading features from: {args.input}")
    if args.input.startswith("s3://"):
        df = pd.read_parquet(args.input, storage_options={ "client_kwargs": {"endpoint_url": "http://localhost:4566"},
    "key": "test",
    "secret": "test" }, engine='pyarrow')
    else:
        df = pd.read_parquet(args.input)

    # Split features/target
    X = df.drop(columns=['total_kwh'])
    y = df['total_kwh']

    # Train-test split (last 30 days as test)
    split_date = df.index[-24*30]
    X_train = X[df.index < split_date]
    X_test = X[df.index >= split_date]
    y_train = y[df.index < split_date]
    y_test = y[df.index >= split_date]

    # Model selection
    if args.model == "lr":
        model = LinearRegression()
    elif args.model == "dt":
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
    elif args.model == "xgb":
        model = XGBRegressor(n_estimators=100, verbosity=0, random_state=42, max_depth=5)
    elif args.model == "lgb":
        model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, random_state=42)

    # ----- MLflow run -----
    with mlflow.start_run(run_name=args.model):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        mlflow.log_param("model_type", args.model)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        # Save model artifact
        mlflow.sklearn.log_model(model, "model")
        # Save locally if requested
        now = datetime.now().strftime('%Y%m%d_%H%M')
        local_model_path = os.path.join(args.output, f"{args.model}_model_{now}.joblib")
        joblib.dump(model, local_model_path)
        print(f"Model saved locally: {local_model_path}")

        # Upload to S3 if output is an s3 path
        s3_model_key = f"artifacts/model/{args.model}_model_{now}.joblib"
        s3 = boto3.client('s3',
                        endpoint_url="http://localhost:4566",
                        aws_access_key_id="test",
                        aws_secret_access_key="test")
        try:
            s3.create_bucket(Bucket=args.bucket)
        except:
            pass
        s3.upload_file(local_model_path, args.bucket, s3_model_key)
        print(f"Model saved to s3://{args.bucket}/{s3_model_key}")

if __name__ == "__main__":
    try:
        args = parse_args()
        print("🚀 Starting training pipeline...")
        main(args)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()