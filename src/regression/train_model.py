"""
Revenue Prediction (Regression Only, ALL sessions)

Note requirement satisfied:
✅ Uses libraries (sklearn) for the model
✅ ALSO uses from-scratch functions:
   - train_test_split_np (data split)
   - rmse / mae / r2 / mape_nonzero (metrics)

Fix in this version:
✅ Correct sample_weight passing to TransformedTargetRegressor:
   model.fit(..., sample_weight=sample_weight)
"""

import pandas as pd
import numpy as np
from minio import Minio
from io import BytesIO
import mlflow
import mlflow.sklearn
import joblib

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
SOURCE_BUCKET = "processed-data"
MODEL_BUCKET = "models"
MLFLOW_URI = "http://localhost:5001"


# ----------------------------
# FROM SCRATCH (required)
# ----------------------------
def train_test_split_np(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(np.floor(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def rmse(y_true, y_pred):
    e = y_true - y_pred
    return float(np.sqrt(np.mean(e * e)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    y_true = y_true.astype(float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def mape_nonzero(y_true, y_pred):
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


# ----------------------------
# Trainer
# ----------------------------
class RegressionModelTrainer:
    def __init__(self, strategy="all"):
        self.strategy = strategy
        self.minio_client = Minio(
            MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
        )
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("Revenue_Prediction_Regression_v2")

    def load_data(self):
        print("📥 Loading training data...")
        response = self.minio_client.get_object(SOURCE_BUCKET, "training_data.parquet")
        df = pd.read_parquet(BytesIO(response.read()))
        response.close()
        response.release_conn()

        if self.strategy != "all":
            raise ValueError("This script is for strategy='all' (all sessions).")

        print("   Strategy: All sessions (single regressor, weighted)")

        # Separate features and target
        drop_cols = ["revenue"]
        if "is_ordered" in df.columns:
            drop_cols.append("is_ordered")  # avoid leakage

        X_df = df.drop(columns=drop_cols)
        y_s = df["revenue"].astype(float)

        print(f"✅ Loaded {len(df):,} samples")
        print(f"   Features: {X_df.shape[1]}")
        print(
            f"   Target stats: mean=${y_s.mean():.2f}, std=${y_s.std():.2f}, "
            f"max=${y_s.max():.2f}, nonzero={(y_s>0).mean()*100:.2f}%"
        )

        return X_df, y_s

    def prepare_data(self, X_df, y_s, test_size=0.2, random_state=42):
        print(f"\n🔀 Splitting data (test_size={test_size})...")

        X = X_df.to_numpy(dtype=np.float64)
        y = y_s.to_numpy(dtype=np.float64)

        X_train, X_test, y_train, y_test = train_test_split_np(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"   Train: {len(X_train):,} samples")
        print(f"   Test:  {len(X_test):,} samples")

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        print("\n🚀 Training ONE regression model (HistGBR + log1p target + weights)...")
        print(f"   Train samples: {len(X_train):,}")
        print(f"   Non-zero rate: {(y_train > 0).mean()*100:.2f}%")
        print(f"   Revenue mean: ${y_train.mean():.2f}, max=${y_train.max():.2f}")

        converter_weight = 30.0  # try 10, 30, 100
        sample_weight = np.where(y_train > 0, converter_weight, 1.0).astype(np.float64)

        base = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=1200,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )

        model = TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1,
        )

        # ✅ FIX: pass sample_weight directly (not regressor__sample_weight)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        mlflow.log_param("converter_weight", converter_weight)
        mlflow.log_param("model_type", "HistGradientBoostingRegressor")
        mlflow.log_param("target_transform", "log1p/expm1")

        return model

    def evaluate_model(self, model, X_test, y_test):
        print("\n📊 Evaluating model...")

        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0.0)

        _rmse = rmse(y_test, y_pred)
        _mae = mae(y_test, y_pred)
        _r2 = r2(y_test, y_pred)
        _mape = mape_nonzero(y_test, y_pred)

        print("\n🎯 Regression Metrics:")
        print(f"   RMSE:  ${_rmse:.2f}")
        print(f"   MAE:   ${_mae:.2f}")
        print(f"   R²:    {_r2:.4f}")
        print(f"   MAPE:  {_mape:.2f}%")

        return {"rmse": _rmse, "mae": _mae, "r2": _r2, "mape": _mape}

    def save_model(self, model, model_name="regression_model.pkl"):
        print("\n💾 Saving model to MinIO...")

        if not self.minio_client.bucket_exists(MODEL_BUCKET):
            self.minio_client.make_bucket(MODEL_BUCKET)

        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        self.minio_client.put_object(
            MODEL_BUCKET,
            model_name,
            buffer,
            length=buffer.getbuffer().nbytes,
            content_type="application/octet-stream",
        )

        print(f"✅ Model saved to s3://{MODEL_BUCKET}/{model_name}")

    def run(self):
        print("=" * 60)
        print("  REGRESSION MODEL TRAINING (ALL SESSIONS, REGRESSION ONLY)")
        print("=" * 60)

        with mlflow.start_run():
            X_df, y_s = self.load_data()
            X_train, X_test, y_train, y_test = self.prepare_data(X_df, y_s)

            mlflow.log_param("strategy", self.strategy)
            mlflow.log_param("total_samples", int(len(y_s)))
            mlflow.log_param("train_samples", int(len(y_train)))
            mlflow.log_param("test_samples", int(len(y_test)))
            mlflow.log_param("num_features", int(X_train.shape[1]))
            mlflow.log_param("target_mean", float(y_s.mean()))
            mlflow.log_param("target_std", float(y_s.std()))
            mlflow.log_param("target_nonzero_rate", float((y_s > 0).mean()))

            model = self.train_model(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            self.save_model(model)
            mlflow.log_param("model_saved_to", f"s3://{MODEL_BUCKET}/regression_model.pkl")

            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETE!")
            print(f"   RMSE: ${metrics['rmse']:.2f}")
            print(f"   MAE:  ${metrics['mae']:.2f}")
            print(f"   R²:   {metrics['r2']:.4f}")
            print("=" * 60)


if __name__ == "__main__":
    trainer = RegressionModelTrainer(strategy="all")
    trainer.run()
