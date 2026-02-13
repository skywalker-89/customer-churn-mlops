"""
Regression Benchmark Runner — All Models (From-Scratch + Library)

Trains ALL regression models on the same data, evaluates them, logs
to MLflow, and produces a comparison table.

Usage:
    python src/regression/regression_benchmark.py

Requires Docker stack running (MinIO for data, MLflow for tracking).
"""

import sys
import os
import numpy as np
import pandas as pd
from io import BytesIO
import mlflow
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Allow imports from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models_scratch.linear_regression import LinearRegressionScratch
from src.models_scratch.multiple_regression import MultipleRegressionScratch
from src.models_scratch.polynomial_regression import PolynomialRegressionScratch
from src.models_scratch.ridge_regression import RidgeRegressionScratch


# ============================================================
# FROM-SCRATCH metric functions (reused from train_model.py)
# ============================================================
def train_test_split_np(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(np.floor(n * test_size))
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def mape_nonzero(y_true, y_pred):
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


# ============================================================
# Data loader
# ============================================================
def load_data(strategy="converting_only"):
    """Load training data from MinIO."""
    from minio import Minio

    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minio_password")
    bucket = os.getenv("SOURCE_BUCKET", "processed-data")

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
    response = client.get_object(bucket, "training_data.parquet")
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    response.release_conn()

    if strategy == "converting_only":
        df = df[df["revenue"] > 0].copy()
        print(f"   Strategy: converting_only → {len(df):,} sessions")

    # Drop target and dead features
    drop_cols = ["revenue", "engagement_depth"]  # engagement_depth is constant (std=0)
    if "is_ordered" in df.columns:
        drop_cols.append("is_ordered")

    X_df = df.drop(columns=drop_cols)
    y = df["revenue"].values.astype(np.float64)
    feature_names = list(X_df.columns)
    
    # Feature correlation analysis (for debugging)
    print(f"\n   📊 Feature Correlations with Revenue:")
    correlations = []
    for i, fname in enumerate(feature_names):
        corr = np.corrcoef(X_df[fname].values, y)[0, 1]
        correlations.append((fname, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for fname, corr in correlations[:5]:  # Top 5
        print(f"      {fname:<30} {corr:>7.4f}")
    
    # Feature Engineering: Create interaction features
    print(f"\n   🔧 Engineering interaction features...")
    X_array = X_df.values.astype(np.float64)
    
    # hour_of_day × is_weekend (time patterns differ on weekends)
    if 'hour_of_day' in feature_names and 'is_weekend' in feature_names:
        hour_idx = feature_names.index('hour_of_day')
        weekend_idx = feature_names.index('is_weekend')
        hour_weekend = X_array[:, hour_idx] * X_array[:, weekend_idx]
        X_array = np.column_stack([X_array, hour_weekend])
        feature_names.append('hour_x_weekend')
    
    # utm_source × device_type (source effectiveness varies by device)
    if 'utm_source_gsearch' in feature_names and 'device_type_mobile' in feature_names:
        gsearch_idx = feature_names.index('utm_source_gsearch')
        mobile_idx = feature_names.index('device_type_mobile')
        gsearch_mobile = X_array[:, gsearch_idx] * X_array[:, mobile_idx]
        X_array = np.column_stack([X_array, gsearch_mobile])
        feature_names.append('gsearch_x_mobile')
    
    # landing_page × is_repeat (different pages for new vs returning)
    landing_pages = [f for f in feature_names if f.startswith('landing_page_')]
    if landing_pages and 'is_repeat_session' in feature_names:
        repeat_idx = feature_names.index('is_repeat_session')
        for lp in landing_pages[:3]:  # Top 3 most common landing pages
            lp_idx = feature_names.index(lp)
            lp_repeat = X_array[:, lp_idx] * X_array[:, repeat_idx]
            X_array = np.column_stack([X_array, lp_repeat])
            feature_names.append(f'{lp.split("/")[-1]}_x_repeat')
    
    print(f"      Added {X_array.shape[1] - X_df.shape[1]} interaction features")
    print(f"      Total features: {len(feature_names)}")
    
    return X_array, y, feature_names


# ============================================================
# Evaluate helper
# ============================================================
def evaluate(name, y_true, y_pred):
    """Return a dict of metrics."""
    metrics = {
        "model": name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "MAPE_%": mape_nonzero(y_true, y_pred),
    }
    return metrics


# ============================================================
# Train all from-scratch models with INCREMENTAL LEARNING
# ============================================================
def train_scratch_models_incremental(X_train, y_train, X_test, y_test, feature_names):
    """
    Train all 4 from-scratch models with warm-start support.
    
    - First run: Train from scratch with high epochs (5K-10K)
    - Subsequent runs: Load from MinIO, continue training (1K-2K incremental)
    """
    from src.models_scratch.base import BaseModel
    
    results = []
    
    # Model configurations with OPTIMIZED HYPERPARAMETERS
    configs = [
        {
            "name": "linear_regression",
            "class": LinearRegressionScratch,
            "label": "Linear (scratch)",
            "initial_epochs": 5000,      # First run: thorough training
            "incremental_epochs": 1000,  # Subsequent runs: fine-tuning
            "lr": 0.01,
            "params": {}
        },
        {
            "name": "multiple_regression",
            "class": MultipleRegressionScratch,
            "label": "Multiple (scratch)",
            "initial_epochs": 8000,      # More features = more iterations
            "incremental_epochs": 1500,
            "lr": 0.01,
            "params": {}
        },
        {
            "name": "polynomial_regression",
            "class": PolynomialRegressionScratch,
            "label": "Polynomial (scratch)",
            "initial_epochs": 10000,     # Many features = needs more training
            "incremental_epochs": 2000,
            "lr": 0.0005,                # Lower LR for stability
            "params": {"degree": 2}
        },
        {
            "name": "ridge_regression",
            "class": RidgeRegressionScratch,
            "label": "Ridge (scratch)",
            "initial_epochs": 8000,
            "incremental_epochs": 1500,
            "lr": 0.01,
            "params": {"alpha": 1.0}
        }
    ]
    
    for config in configs:
        print("\n" + "=" * 60)
        print(f"  [SCRATCH] {config['label']} (Incremental GD)")
        print("=" * 60)
        
        # Try to load existing model from MinIO
        existing_model = BaseModel.load_from_minio(config["name"])
        
        if existing_model is not None:
            # WARM START: Continue from existing weights
            model = existing_model
            epochs = config["incremental_epochs"]
            warm_start = True
            print(f"   📈 Continuing training (+{epochs} more epochs)")
        else:
            # COLD START: First-time training
            model = config["class"](**config["params"])
            epochs = config["initial_epochs"]
            warm_start = False
            print(f"   🆕 First training ({epochs} epochs from scratch)")
        
        # Train the model
        model.fit(
            X_train, y_train,
            epochs=epochs,
            lr=config["lr"],
            feature_names=feature_names,
            warm_start=warm_start
        )
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        result = evaluate(config["label"], y_test, y_pred)
        result["model_obj"] = model
        result["model_name"] = config["name"]
        result["epochs_trained"] = epochs
        results.append(result)
        
        # Save back to MinIO for next run
        model.save_to_minio(config["name"])
    
    return results


def train_builtin_model(X_train, y_train, X_test, y_test):
    """Train ONE strong sklearn model for comparison."""
    from sklearn.ensemble import RandomForestRegressor
    
    print("\n" + "=" * 60)
    print("  [SKLEARN] Random Forest Regressor")
    print("  (100 decision trees built iteratively)")
    print("=" * 60)
    
    model = RandomForestRegressor(
        n_estimators=200,  # Increased from 100 for better ensemble
        max_depth=20,      # Increased from 15 for more complex trees
        min_samples_split=2,  # Decreased from 5 for finer splits
        min_samples_leaf=1,   # Decreased from 2 for more detailed leaves
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"   Training Random Forest with {model.n_estimators} trees...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    result = evaluate("Random Forest (sklearn)", y_test, y_pred)
    result["model_obj"] = model
    result["n_estimators"] = 100
    
    print(f"   ✅ Training complete")
    return [result]


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  REGRESSION BENCHMARK — ALL MODELS")
    print("  From-Scratch (NumPy) + Library (sklearn)")
    print("=" * 60)

    # 1. Load data
    print("\n📥 Loading data (converting_only strategy)...")
    X, y, feature_names = load_data(strategy="converting_only")
    print(f"   Samples: {len(y):,}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Revenue — mean=${y.mean():.2f}, std=${y.std():.2f}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split_np(X, y)
    print(f"\n🔀 Split: Train={len(X_train):,}, Test={len(X_test):,}")

    # 3. Setup MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Regression_Models_Benchmark")

    # 4. Train all models
    all_results = []

    with mlflow.start_run(run_name="full_benchmark"):
        mlflow.log_param("strategy", "converting_only")
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", len(feature_names))

        print("\n" + "━" * 60)
        print("  PART 1: FROM-SCRATCH MODELS (Incremental Learning)")
        print("━" * 60)
        scratch_results = train_scratch_models_incremental(X_train, y_train, X_test, y_test, feature_names)
        all_results.extend(scratch_results)

        print("\n" + "━" * 60)
        print("  PART 2: BUILT-IN MODEL (sklearn Random Forest)")
        print("━" * 60)
        lib_results = train_builtin_model(X_train, y_train, X_test, y_test)
        all_results.extend(lib_results)

        # 5. Log all metrics to MLflow
        for r in all_results:
            name = r["model"].replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            mlflow.log_metric(f"{name}_RMSE", r["RMSE"])
            mlflow.log_metric(f"{name}_MAE", r["MAE"])
            mlflow.log_metric(f"{name}_R2", r["R2"])

    # 6. Print comparison table
    print("\n" + "=" * 60)
    print("  📊 RESULTS COMPARISON")
    print("=" * 60)

    # Build table
    table_data = []
    for r in all_results:
        table_data.append({
            "Model": r["model"],
            "RMSE ($)": f"{r['RMSE']:.2f}",
            "MAE ($)": f"{r['MAE']:.2f}",
            "R²": f"{r['R2']:.4f}",
            "MAPE (%)": f"{r['MAPE_%']:.1f}",
        })

    df_table = pd.DataFrame(table_data)
    print(df_table.to_string(index=False))

    # Save to CSV
    os.makedirs("reports", exist_ok=True)
    csv_path = os.path.join(PROJECT_ROOT, "reports", "regression_benchmark_results.csv")
    df_raw = pd.DataFrame([{k: v for k, v in r.items() if k != "model_obj"} for r in all_results])
    df_raw.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to {csv_path}")

    # Save predictions for comparison plots
    predictions = {}
    for r in all_results:
        model = r.get("model_obj")
        if model is not None:
            name = r["model"]
            try:
                predictions[name] = model.predict(X_test)
            except Exception:
                pass

    np.savez(
        os.path.join(PROJECT_ROOT, "reports", "regression_predictions.npz"),
        y_test=y_test,
        **{k.replace(" ", "_"): v for k, v in predictions.items()},
    )
    print("💾 Predictions saved to reports/regression_predictions.npz")

    # Find best model
    best = min(all_results, key=lambda r: r["RMSE"])
    print(f"\n🏆 Best model by RMSE: {best['model']} (RMSE=${best['RMSE']:.2f}, R²={best['R2']:.4f})")

    return all_results


if __name__ == "__main__":
    results = main()
