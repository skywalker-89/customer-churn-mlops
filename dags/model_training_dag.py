"""
Model Training DAG - Orchestrates Classification and Regression Model Training

This DAG:
1. Loads training_data.parquet from MinIO
2. Triggers classification model training
3. Triggers regression model training
4. Logs all experiments to MLflow
5. Saves trained models to MinIO

ML Engineers implement the actual models in:
- src/classification/train_model.py
- src/regression/train_model.py
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
from pathlib import Path

# Add src to path so we can import model trainers
project_root = Path("/opt/airflow")
sys.path.insert(0, str(project_root))

def train_classification_model():
    """Train classification model (conversion prediction)"""
    print("🎯 Starting Classification Model Training...")
    
    from src.classification.train_model import ClassificationModelTrainer
    
    trainer = ClassificationModelTrainer()
    trainer.run()
    
    print("✅ Classification model training complete")

def train_regression_model():
    """Train regression model (revenue prediction)"""
    print("💰 Starting Regression Model Training...")
    
    from src.regression.train_model import RegressionModelTrainer
    
    # You can change strategy here: 'all' or 'converting_only'
    trainer = RegressionModelTrainer(strategy='converting_only')
    trainer.run()
    
    print("✅ Regression model training complete")

def validate_training_data():
    """Validate that training data exists before training models"""
    from minio import Minio
    from io import BytesIO
    import pandas as pd
    
    print("🔍 Validating training data...")
    
    client = Minio(
        "minio:9000",  # Docker service name
        access_key="minio_admin",
        secret_key="minio_password",
        secure=False
    )
    
    # Check if training data exists
    try:
        response = client.get_object("processed-data", "training_data.parquet")
        df = pd.read_parquet(BytesIO(response.read()))
        response.close()
        response.release_conn()
        
        print(f"✅ Training data found: {len(df):,} rows, {len(df.columns)} columns")
        
        # Validate required columns
        required_cols = ['is_ordered', 'revenue']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print("✅ Data validation passed")
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        raise

# --- DAG Definition ---
default_args = {
    "owner": "lead_engineer",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    "model_training_pipeline",
    default_args=default_args,
    description="Train classification and regression models",
    schedule_interval="@weekly",  # Run weekly or trigger manually
    catchup=False,
    tags=['ml', 'training', 'models'],
) as dag:
    
    # Task 1: Validate data
    validate_task = PythonOperator(
        task_id="validate_training_data",
        python_callable=validate_training_data
    )
    
    # Task 2: Train classification model
    classification_task = PythonOperator(
        task_id="train_classification_model",
        python_callable=train_classification_model
    )
    
    # Task 3: Train regression model
    regression_task = PythonOperator(
        task_id="train_regression_model",
        python_callable=train_regression_model
    )
    
    # Task dependencies
    # Both models train in parallel after data validation
    validate_task >> [classification_task, regression_task]
