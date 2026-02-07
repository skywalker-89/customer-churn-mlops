"""
Classification Model Template for ML Engineers

This is a TEMPLATE. ML Engineers should:
1. Implement the train_model() function with their chosen algorithm
2. Handle class imbalance (SMOTE, class weights, etc.)
3. Use F1-score or AUC-ROC for evaluation (NOT accuracy)
4. Log all experiments to MLflow

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
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
SOURCE_BUCKET = "processed-data"
MODEL_BUCKET = "models"
MLFLOW_URI = "http://localhost:5001"

class ClassificationModelTrainer:
    """Template for classification model training"""
    
    def __init__(self):
        self.minio_client = Minio(
            MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
        )
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("Conversion_Prediction_Classification")
    
    def load_data(self):
        """Load training data from MinIO"""
        print("📥 Loading training data...")
        response = self.minio_client.get_object(SOURCE_BUCKET, "training_data.parquet")
        df = pd.read_parquet(BytesIO(response.read()))
        response.close()
        response.release_conn()
        
        # Separate features and target
        X = df.drop(columns=['is_ordered', 'revenue'])
        y = df['is_ordered']
        
        print(f"✅ Loaded {len(df):,} samples")
        print(f"   Features: {X.shape[1]}")
        print(f"   Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train/test sets"""
        print(f"\n🔀 Splitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   Train: {len(X_train):,} samples")
        print(f"   Test:  {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        🚨 ML ENGINEERS: IMPLEMENT YOUR MODEL HERE
        
        Requirements:
        1. Handle class imbalance (6.8% positive class)
           - Use SMOTE: from imblearn.over_sampling import SMOTE
           - Or class_weight='balanced'
           - Or threshold adjustment
        
        2. Choose an algorithm:
           - Logistic Regression (baseline)
           - Random Forest (recommended)
           - XGBoost (best performance)
           - LightGBM (fast training)
        
        3. Tune hyperparameters (GridSearchCV or RandomizedSearchCV)
        
        Example Implementation:
        ```python
        from sklearn.ensemble import RandomForestClassifier
        from imblearn.over_sampling import SMOTE
        
        # Handle imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_resampled, y_resampled)
        
        return model
        ```
        """
        
        # PLACEHOLDER - Replace with your implementation
        print("\n⚠️  PLACEHOLDER MODEL - Replace with actual implementation")
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='most_frequent')
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 Evaluating model...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        f1 = f1_score(y_test, y_pred)
        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None
        
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n🎯 Key Metrics:")
        print(f"   F1-Score: {f1:.4f}")
        if auc:
            print(f"   AUC-ROC:  {auc:.4f}")
        
        return {
            'f1_score': f1,
            'auc_roc': auc if auc else 0.0,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_model(self, model, model_name="classification_model.pkl"):
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
        print("  CLASSIFICATION MODEL TRAINING")
        print("  Task: Conversion Prediction (is_ordered)")
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
            mlflow.log_param("class_imbalance_ratio", y.value_counts()[0] / y.value_counts()[1])
            
            # 3. Train model (ML Engineers implement this)
            model = self.train_model(X_train, y_train)
            
            # 4. Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metric("f1_score", metrics['f1_score'])
            
            # 5. Save to MinIO
            self.save_model(model)
            
            # 6. Log model path to MLflow
            mlflow.log_param("model_saved_to", f"s3://{MODEL_BUCKET}/classification_model.pkl")
            
            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETE!")
            print(f"   MLflow Run ID: {mlflow.active_run().info.run_id}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
            print("=" * 60)

if __name__ == "__main__":
    trainer = ClassificationModelTrainer()
    trainer.run()
