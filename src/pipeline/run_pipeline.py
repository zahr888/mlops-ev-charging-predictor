"""
run_pipeline.py
End-to-end ML pipeline orchestrator
"""
import sys
import os
import yaml
import subprocess
from pathlib import Path

# Disable MLflow emojis globally
os.environ['MLFLOW_TRACKING_PRINT_RUN_URL'] = 'false'

def load_config(config_path="config.yaml"):
    """Load pipeline configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")


    env = os.environ.copy()
    env.setdefault('PYTHONUTF8', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')


    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
    )

    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

    if result.stdout:
        print(result.stdout)

    return result


def main():
    print("=" * 60)
    print("[PIPELINE] Starting ML Pipeline")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Load configuration
    config = load_config()
    raw_path = config['data']['raw_path']
    features_path = config['data']['features_path']
    test_path = config['data']['test_path']
    model_output = config['paths']['model_output']
    models = config['models']
    
    print(f"\n[CONFIG] Loaded configuration:")
    print(f"  Raw: {raw_path}")
    print(f"  Features: {features_path}")
    print(f"  Models: {', '.join(models)}")
    
    # Step 1: Feature Engineering
    print("\n" + "=" * 60)
    print("[STEP 1/3] Feature Engineering")
    print("=" * 60)
    run_command(["python", "features.py", "--input", raw_path])
    
    # Step 2: Model Training
    print("\n" + "=" * 60)
    print("[STEP 2/3] Training Models")
    print("=" * 60)
    for i, model in enumerate(models, 1):
        print(f"\n[TRAIN {i}/{len(models)}] Training {model.upper()}...")
        run_command(["python", "train.py", "--input", features_path, "--model", model])
    
    # Step 3: Model Evaluation
    print("\n" + "=" * 60)
    print("[STEP 3/3] Evaluating Models")
    print("=" * 60)
    
    model_dir = Path(model_output)
    if not model_dir.is_absolute():
        model_dir = (script_dir / model_dir).resolve()

    evaluated = 0
    for model in models:
        pattern = f"{model}_model_*.joblib"
        model_files = sorted(model_dir.glob(pattern), key=os.path.getmtime, reverse=True)
        
        if model_files:
            latest_model = str(model_files[0])
            print(f"\n[EVAL] {model.upper()}: {Path(latest_model).name}")
            run_command(["python", "eval.py", "--model", latest_model, "--test-data", test_path])
            evaluated += 1
        else:
            print(f"[WARN] No model found for {model}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Pipeline completed!")
    print(f"  - {len(models)} models trained")
    print(f"  - {evaluated} models evaluated")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
