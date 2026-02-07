"""
Regression Model Template for ML Engineers

This is a TEMPLATE. ML Engineers should:
1. Implement the train_model() function with their chosen algorithm
2. Decide on regression strategy (all sessions vs converting-only)
3. Handle long-tail distribution (log transform, robust scaler, etc.)
4. Use RMSE, MAE, or R² for evaluation

Infrastructure (already provided):
- Data loading from MinIO
- Train/test split
- MLflow integration
- Model saving to MinIO
"""

import pandas as pd
import numpy as np
from minio import Minio
from io import BytesIO
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
SOURCE_BUCKET = "processed-data"
MODEL_BUCKET = "models"
MLFLOW_URI = "http://localhost:5001"

class RegressionModelTrainer:
    """Template for regression model training"""
    
    def __init__(self, strategy='all'):
        """
        Args:
            strategy: 'all' (predict revenue for all sessions)
                     'converting_only' (predict only for sessions with orders)
        """
        self.strategy = strategy
        self.minio_client = Minio(
            MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
        )
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("Revenue_Prediction_Regression")
    
    def load_data(self):
        """Load training data from MinIO"""
        print("📥 Loading training data...")
        response = self.minio_client.get_object(SOURCE_BUCKET, "training_data.parquet")
        df = pd.read_parquet(BytesIO(response.read()))
        response.close()
        response.release_conn()
        
        # Filter based on strategy
        if self.strategy == 'converting_only':
            df = df[df['is_ordered'] == 1]
            print(f"   Strategy: Converting sessions only")
        else:
            print(f"   Strategy: All sessions")
        
        # Separate features and target
        X = df.drop(columns=['is_ordered', 'revenue'])
        y = df['revenue']
        
        print(f"✅ Loaded {len(df):,} samples")
        print(f"   Features: {X.shape[1]}")
        print(f"   Target stats: mean=${y.mean():.2f}, std=${y.std():.2f}, max=${y.max():.2f}")
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train/test sets"""
        print(f"\n🔀 Splitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"   Train: {len(X_train):,} samples")
        print(f"   Test:  {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        🚨 ML ENGINEERS: IMPLEMENT YOUR MODEL HERE
        
        Requirements:
        1. Choose a regression strategy:
           Option A: Two-stage model
             - First predict if user will convert (classification)
             - Then predict revenue only for predicted converters
           
           Option B: Direct regression on all sessions
             - Predict revenue for all (most will be 0)
             - May need to handle zero-inflation
           
           Option C: Regression on converting sessions only
             - Simplest approach
             - Only predict order value given that user converted
        
        2. Handle distribution issues:
           - Long-tail: Consider log transformation of target
           - Zeros: If predicting all sessions, handle zero-inflation
        
        3. Choose an algorithm:
           - Linear Regression (baseline)
           - Random Forest Regressor (robust to outliers)
           - XGBoost Regressor (best performance)
           - LightGBM Regressor (fast training)
        
        Example Implementation:
        ```python
        from sklearn.ensemble import RandomForestRegressor
        
        # Option: Log transform target for better distribution
        # y_log = np.log1p(y_train)  # log(1 + y) to handle zeros
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # If log transformed, remember to inverse transform predictions:
        # y_pred = np.expm1(model.predict(X_test))
        
        return model
        ```
        """
        
        # PLACEHOLDER - Replace with your implementation
        print("\n⚠️  PLACEHOLDER MODEL - Replace with actual implementation")
        from sklearn.dummy import DummyRegressor
        model = DummyRegressor(strategy='mean')
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 Evaluating model...")
        
        y_pred = model.predict(X_test)
        
        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 0)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Additional metric: MAPE (Mean Absolute Percentage Error) for non-zero values
        non_zero_mask = y_test > 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = None
        
        print(f"\n🎯 Regression Metrics:")
        print(f"   RMSE:  ${rmse:.2f}")
        print(f"   MAE:   ${mae:.2f}")
        print(f"   R²:    {r2:.4f}")
        if mape:
            print(f"   MAPE:  {mape:.2f}%")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape if mape else 0.0,
            'predictions': y_pred
        }
    
    def save_model(self, model, model_name="regression_model.pkl"):
        """Save model to MinIO"""
        print(f"\n💾 Saving model to MinIO...")
        
        # Ensure bucket exists
        if not self.minio_client.bucket_exists(MODEL_BUCKET):
            self.minio_client.make_bucket(MODEL_BUCKET)
        
        # Serialize model
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        
        # Upload
        self.minio_client.put_object(
            MODEL_BUCKET,
            model_name,
            buffer,
            length=buffer.getbuffer().nbytes,
            content_type="application/octet-stream"
        )
        
        print(f"✅ Model saved to s3://{MODEL_BUCKET}/{model_name}")
    
    def run(self):
        """Main training pipeline"""
        print("=" * 60)
        print("  REGRESSION MODEL TRAINING")
        print("  Task: Revenue Prediction")
        print("=" * 60)
        
        with mlflow.start_run():
            # 1. Load data
            X, y = self.load_data()
            
            # 2. Split data
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)
            
            # Log data info
            mlflow.log_param("total_samples", len(X))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("num_features", X.shape[1])
            mlflow.log_param("strategy", self.strategy)
            mlflow.log_param("target_mean", float(y.mean()))
            mlflow.log_param("target_std", float(y.std()))
            
            # 3. Train model (ML Engineers implement this)
            model = self.train_model(X_train, y_train)
            
            # 4. Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metric("rmse", metrics['rmse'])
            mlflow.log_metric("mae", metrics['mae'])
            mlflow.log_metric("r2", metrics['r2'])
            mlflow.log_metric("mape", metrics['mape'])
            
            # 5. Save to MinIO
            self.save_model(model)
            
            # 6. Log model path to MLflow
            mlflow.log_param("model_saved_to", f"s3://{MODEL_BUCKET}/regression_model.pkl")
            
            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETE!")
            print(f"   MLflow Run ID: {mlflow.active_run().info.run_id}")
            print(f"   RMSE: ${metrics['rmse']:.2f}")
            print(f"   MAE:  ${metrics['mae']:.2f}")
            print(f"   R²:   {metrics['r2']:.4f}")
            print("=" * 60)

if __name__ == "__main__":
    # You can change strategy here
    trainer = RegressionModelTrainer(strategy='converting_only')
    trainer.run()
