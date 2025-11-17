import os
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import mlflow

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Command-line arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation Pipeline")
    parser.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--test-data", required=True, help="Path to test features (local or s3)")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "..", "reports"), help="Directory for saving reports/plots")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment", default="evaluations", help="MLflow experiment name")
    parser.add_argument("--run", default="evaluation", help="MLflow run name")
    return parser.parse_args()

# --- Main evaluation logic ---
def main(args):
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    # Load test data
    print(f"Loading test data from: {args.test_data}")
    if args.test_data.startswith("s3://"):
        df = pd.read_parquet(args.test_data, storage_options={
        "client_kwargs": {"endpoint_url": "http://localhost:4566"},
        "key": "test","secret": "test"}, engine="pyarrow")
    else:
        df = pd.read_parquet(args.test_data)

    X_test = df.drop(columns=['total_kwh'])
    y_test = df['total_kwh']

    print(f"Loading model from: {args.model}")
    model = joblib.load(args.model)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluation Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Create output directory
    model_name = os.path.splitext(os.path.basename(args.model))[0]  
    dynamic_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(dynamic_output_dir, exist_ok=True)
    args.output_dir = dynamic_output_dir

    # Generate and save visualizations
    # Plot 1: Predictions vs Actuals
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", marker='o', alpha=0.6)
    plt.plot(y_pred, label="Predicted", marker='x', alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel("totla kwh")
    plt.title("Predictions vs Actuals")
    plt.legend()
    plt.grid(alpha=0.3)
    pred_plot_path = os.path.join(args.output_dir, "predictions_vs_actuals.png")
    plt.savefig(pred_plot_path)
    plt.close()
    print(f"Saved plot: {pred_plot_path}")

    # Plot 2: Residuals
    residuals = y_test.values - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(alpha=0.3)
    residual_plot_path = os.path.join(args.output_dir, "residuals.png")
    plt.savefig(residual_plot_path)
    plt.close()
    print(f"Saved plot: {residual_plot_path}")

    # Save metrics to a text report
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model Evaluation Report\n")
        f.write(f"========================\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Data: {args.test_data}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"  MAE:  {mae:.4f}\n")
        f.write(f"  RMSE: {rmse:.4f}\n")
        f.write(f"  R²:   {r2:.4f}\n")
    print(f"Saved report: {report_path}")

    # Optional: Log metrics and artifacts to MLflow
    with mlflow.start_run(run_name=args.run):
        mlflow.log_metric("eval_mae", mae)
        mlflow.log_metric("eval_rmse", rmse)
        mlflow.log_metric("eval_r2", r2)

    print("Evaluation complete.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
