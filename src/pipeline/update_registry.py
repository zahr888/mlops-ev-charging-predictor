"""
update_registry.py
Selects the best model based on metrics.json files and updates models/registry.json
"""

import os
import json
from pathlib import Path

# BASE_DIR = project root (mlops/)
BASE_DIR = Path(__file__).resolve().parents[2]

REPORTS_DIR = BASE_DIR / "src" / "reports"
MODELS_DIR = BASE_DIR / "src" / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

def load_all_metrics():
    """Scan reports directory and load all metrics.json files."""
    metrics_list = []
    for metrics_file in REPORTS_DIR.glob("*/metrics.json"):
        with open(metrics_file) as f:
            metrics = json.load(f)
        metrics["metrics_path"] = str(metrics_file)
        metrics_list.append(metrics)
    return metrics_list

def select_best_model(metrics_list):
    """Choose best model based on lowest MAE (or RMSE)."""
    if not metrics_list:
        raise ValueError("No metrics.json files found; cannot update registry.")
    
    best = min(metrics_list, key=lambda m: m["mae"])
    return best

def build_registry_entry(best_metrics):
    """Build registry.json structure from best model metrics."""
    return {
        "production": {
            "model_name": Path(best_metrics["model_path"]).stem,
            "model_path": best_metrics["model_path"],
            "metrics": {
                "mae": best_metrics["mae"],
                "rmse": best_metrics["rmse"],
                "r2": best_metrics["r2"]
            },
            "test_data": best_metrics["test_data"],
            # optional: keep link to metrics.json
            "metrics_path": best_metrics["metrics_path"]
        }
    }

def main():
    metrics_list = load_all_metrics()
    best = select_best_model(metrics_list)
    registry = build_registry_entry(best)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[OK] Updated registry at: {REGISTRY_PATH}")
    print(f"[INFO] Production model: {registry['production']['model_name']}")
    print(f"[INFO] MAE: {registry['production']['metrics']['mae']}")
    print(f"[INFO] Path: {registry['production']['model_path']}")

if __name__ == "__main__":
    main()
