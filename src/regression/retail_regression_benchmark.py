"""
Retail Regression Benchmark
Compares 4 from-scratch regression models vs 1 sklearn Random Forest
Target: total_sales (predicting customer purchase amount)
"""
import os
import numpy as np
import pandas as pd
import mlflow
from io import BytesIO
from minio import Minio

# Import from-scratch models
from src.models_scratch.linear_regression import LinearRegressionScratch
from src.models_scratch.multiple_regression import MultipleRegressionScratch
from src.models_scratch.polynomial_regression import PolynomialRegressionScratch
from src.models_scratch.xgboost import XGBoostScratch

# Import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# ============================================================
# Utility Functions (Metrics + Train/Test Split)
# ============================================================
def train_test_split_np(X, y, test_size=0.2, random_state=42):
    """NumPy-based train/test split"""
    rng = np.random.RandomState(random_state)
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


print("=" * 60)
print("RETAIL REGRESSION BENCHMARK - Total Sales Prediction")
print("=" * 60)


# ============================================================
# Data Loader
# ============================================================
def load_retail_data(sample_size=None):
    """
    Load retail training data from MinIO.
    
    Args:
        sample_size: Number of rows to sample (default 50k to avoid OOM)
                     Set to None to use full dataset
    """
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minio_password")
    bucket = os.getenv("SOURCE_BUCKET", "processed-data")

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
    
    print(f"\n📥 Loading from s3://{bucket}/training_data.parquet...")
    response = client.get_object(bucket, "training_data.parquet")
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    response.release_conn()

    print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Sample data to reduce memory if needed
    if sample_size and len(df) > sample_size:
        print(f"   📊 Sampling {sample_size:,} rows to reduce memory usage...")
        df = df.sample(n=sample_size, random_state=42)
        print(f"   Using {len(df):,} rows for training")

    # Drop both targets (we'll use total_sales) AND leaked features
    drop_cols = ["total_sales", "churned", "clv_per_year"]
    X_df = df.drop(columns=drop_cols, errors='ignore')
    
    # CRITICAL: Only keep numeric columns (drop dates/strings that can't convert to float)
    X_df = X_df.select_dtypes(include=[np.number])
    
    y = df["total_sales"].values.astype(np.float64)
    feature_names = list(X_df.columns)
    
    # Show top correlated features (on sample, numeric only to avoid OOM and type errors)
    print(f"\n   📊 Top Feature Correlations with Total Sales:")
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    numeric_features = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in drop_cols]
    
    correlations = []
    for fname in numeric_features:
        corr = np.corrcoef(sample_df[fname].values, sample_df["total_sales"].values)[0, 1]
        if not np.isnan(corr):
            correlations.append((fname, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for fname, corr in correlations[:10]:
        print(f"      {fname:<45} {corr:>7.4f}")
    
    print(f"\n   Features: {len(feature_names)}")
    print(f"   Target (total_sales): mean=${y.mean():.2f}, std=${y.std():.2f}")
    
    return X_df.values.astype(np.float64), y, feature_names


# ============================================================
# Train From-Scratch Models
# ============================================================
def train_scratch_models(X_train, y_train, X_test, y_test, feature_names):
    """Train 4 from-scratch models with incremental learning"""
    print("\n" + "=" * 60)
    print("🔨 TRAINING FROM-SCRATCH MODELS (Incremental)")
    print("=" * 60)
    
    models = [
        ("Linear", LinearRegressionScratch(), 10000, 0.001),
        ("Multiple", MultipleRegressionScratch(), 8000, 0.0001),
        ("Polynomial", PolynomialRegressionScratch(degree=2), 50, 0.00001), # Reduced epochs for SGD
        ("XGBoost", XGBoostScratch(n_estimators=50, learning_rate=0.1, max_depth=3), 50, 0.1),
    ]
    
    results = []
    
    for name, model, epochs, lr in models:
        print(f"\n{'─' * 60}")
        print(f"📊 {name} Regression")
        print(f"{'─' * 60}")
        
        # Try to load from MinIO (warm start)
        # Try to load from MinIO (warm start)
        model_name = f"{name.lower()}_regression"
        try:
            # load_from_minio is a class method that returns a new instance
            loaded_model = model.load_from_minio(model_name)
            if loaded_model:
                model = loaded_model
                print(f"   ✅ Loaded existing model from MinIO: {model_name}")
                warm_start = True
                actual_epochs = 1500  # Fewer epochs for incremental
            else:
                print(f"   🆕 No existing model found: {model_name} (training from scratch)")
                warm_start = False
                actual_epochs = epochs
        except Exception as e:
            print(f"   🆕 Error loading model: {e} (training from scratch)")
            warm_start = False
            actual_epochs = epochs
        
        # Train
        # Train
        if name == "Polynomial":
            model.fit(X_train, y_train, epochs=actual_epochs, lr=lr, 
                      feature_names=feature_names, warm_start=warm_start, batch_size=4096)
        elif name == "XGBoost":
            # XGBoost doesn't use batch_size, maps epochs -> n_estimators
            model.fit(X_train, y_train, epochs=actual_epochs, lr=lr,
                      feature_names=feature_names, warm_start=warm_start)
        else:
            model.fit(X_train, y_train, epochs=actual_epochs, lr=lr, 
                      feature_names=feature_names, warm_start=warm_start)
        
        # Save back to MinIO
        model.save_to_minio(model_name)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        _rmse = rmse(y_test, y_pred)
        _mae = mae(y_test, y_pred)
        _r2 = r2(y_test, y_pred)
        _mape = mape_nonzero(y_test, y_pred)
        
        print(f"\n   📈 Results:")
        print(f"      RMSE:  ${_rmse:,.2f}")
        print(f"      MAE:   ${_mae:,.2f}")
        print(f"      R²:    {_r2:.4f}")
        print(f"      MAPE:  {_mape:.2f}%")
        
        results.append({
            "model": f"{name} (scratch)",
            "rmse": _rmse,
            "mae": _mae,
            "r2": _r2,
            "mape": _mape
        })
    
    return results


# ============================================================
# Train Sklearn Model
# ============================================================
def train_sklearn_model(X_train, y_train, X_test, y_test):
    """Train Random Forest as benchmark"""
    print("\n" + "=" * 60)
    print("🌲 TRAINING SKLEARN RANDOM FOREST")
    print("=" * 60)
    
    model = RandomForestRegressor(
        n_estimators=50,     # Reduced from 200
        max_depth=10,        # Reduced from 20
        min_samples_split=5, # Increased for simpler trees
        min_samples_leaf=2,  # Increased for simpler trees
        max_samples=0.5,     # Only train on 50% of data per tree (huge memory saver)
        random_state=42,
        n_jobs=2             # Limit parallelism to avoid thread overhead
    )
    
    print("   Training...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    _rmse = rmse(y_test, y_pred)
    _mae = mae(y_test, y_pred)
    _r2 = r2(y_test, y_pred)
    _mape = mape_nonzero(y_test, y_pred)
    
    print(f"\n   📈 Results:")
    print(f"      RMSE:  ${_rmse:,.2f}")
    print(f"      MAE:   ${_mae:,.2f}")
    print(f"      R²:    {_r2:.4f}")
    print(f"      MAPE:  {_mape:.2f}%")
    
    return [{
        "model": "Random Forest (sklearn)",
        "rmse": _rmse,
        "mae": _mae,
        "r2": _r2,
        "mape": _mape
    }]


# ============================================================
# Main
# ============================================================
def main():
    mlflow.set_experiment("retail_regression_benchmark")
    
    with mlflow.start_run(run_name="retail_total_sales_prediction"):
        # Load data
        X, y, feature_names = load_retail_data()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split_np(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n🔀 Train/Test Split:")
        print(f"   Train: {len(y_train):,} samples")
        print(f"   Test:  {len(y_test):,} samples")
        print(f"   Target (total_sales): mean=${y.mean():.2f}, std=${y.std():.2f}")
        
        # Train models
        scratch_results = train_scratch_models(X_train, y_train, X_test, y_test, feature_names)
        sklearn_results = train_sklearn_model(X_train, y_train, X_test, y_test)
        
        all_results = scratch_results + sklearn_results
        
        # Print comparison
        print("\n" + "=" * 60)
        print("📊 FINAL COMPARISON")
        print("=" * 60)
        df_results = pd.DataFrame(all_results)
        print(df_results.to_string(index=False))
        
        # Log to MLflow
        # Log to MLflow
        for result in all_results:
            # Sanitize model name for MLflow (replace spaces/parentheses with underscores)
            clean_name = result['model'].replace(" ", "_").replace("(", "").replace(")", "").lower()
            metric_prefix = f"{clean_name}"
            
            mlflow.log_metrics({
                f"{metric_prefix}_r2": result['r2'],
                f"{metric_prefix}_rmse": result['rmse'],
                f"{metric_prefix}_mae": result['mae'],
                f"{metric_prefix}_mape": result['mape']
            })
        
        print("\n✅ Benchmark complete!")
        print(f"   Best R²: {df_results['r2'].max():.4f} ({df_results.loc[df_results['r2'].idxmax(), 'model']})")


if __name__ == "__main__":
    main()
