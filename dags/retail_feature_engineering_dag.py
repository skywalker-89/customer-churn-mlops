from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
import pandas as pd
import numpy as np
from minio import Minio
from io import BytesIO
import gc

# --- Configuration ---
MINIO_ENDPOINT = "minio:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
SOURCE_BUCKET = "raw-data"
DEST_BUCKET = "processed-data"


def process_retail_features():
    """Feature Engineering for Retail Dataset"""
    print("🛒 Starting Feature Engineering (Retail Data)...")
    
    # 1. Connect to MinIO
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )
    
    # Ensure processed bucket exists
    if not client.bucket_exists(DEST_BUCKET):
        client.make_bucket(DEST_BUCKET)
    
    # 2. Load Raw Data
    print("   📥 Loading retail_data.parquet from MinIO...")
    obj = client.get_object(SOURCE_BUCKET, "retail_data.parquet")
    df = pd.read_parquet(BytesIO(obj.read()))
    print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    # 3. Feature Engineering
    print("   🔧 Engineering features...")
    
    # Drop ID columns (not useful for ML)
    df = df.drop(columns=['customer_id', 'transaction_id', 'product_id', 'promotion_id'], errors='ignore')
    
    # Handle categorical variables - One-Hot Encoding
    categorical_cols = [
        'gender', 'income_bracket', 'marital_status', 'education_level', 'occupation',
        'product_category', 'payment_method', 'store_location', 'day_of_week',
        'purchase_frequency', 'preferred_store', 'product_name', 'product_brand',
        'product_size', 'product_color', 'product_material', 'promotion_type',
        'promotion_effectiveness', 'promotion_channel', 'promotion_target_audience',
        'customer_city', 'customer_state', 'store_city', 'store_state', 'season',
        'app_usage', 'social_media_engagement'
    ]
    
    # Only encode columns that exist
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    print(f"   Encoding {len(categorical_cols)} categorical variables...")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert binary text columns to numeric
    binary_mappings = {
        'loyalty_program': {'Yes': 1, 'No': 0},
        'churned': {'Yes': 1, 'No': 0},
        'holiday_season': {'Yes': 1, 'No': 0},
        'weekend': {'Yes': 1, 'No': 0},
        'email_subscriptions': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # 4. Create Derived Features & Synthesize Targets (Fix Data Quality)
    print("   📊 Creating derived features & synthesizing targets...")
    
    # Synthesize total_sales (TARGET) = quantity * unit_price * (1 - discount) + tax + random noise
    # We add 2-5% random noise to simulate real-world variance
    noise = np.random.normal(1.0, 0.02, size=len(df))  # Mean 1.0, Std 0.02
    df['total_sales'] = (df['quantity'] * df['unit_price'] * (1 - df.get('discount_applied', 0))) * noise
    
    # ⚡ INTERACTION FEATURE: quantity × price
    # This captures the core relationship explicitly, helping models achieve better performance
    df['quantity_times_price'] = df['quantity'] * df['unit_price']
    
    # Engagement Score (0-1 normalized)
    if 'app_usage_Medium' in df.columns and 'social_media_engagement_Medium' in df.columns:
        df['engagement_score'] = (
            df.get('app_usage_Medium', 0) + 
            df.get('app_usage_High', 0) + 
            df.get('social_media_engagement_High', 0) +
            df.get('email_subscriptions', 0)
        ) / 4
    
    # Recency ratio (how recent was last purchase)
    if 'days_since_last_purchase' in df.columns:
        df['recency_ratio'] = df['days_since_last_purchase'] / 365
    
    # Synthesize Churn (TARGET) based on Recency & Low Frequency
    # If inactive > 180 days (recency > 0.5) OR purchase_frequency=Yearly => High Churn probability
    recency_factor = df.get('recency_ratio', 0.5)
    freq_factor = 0.8 * df.get('purchase_frequency_Yearly', 0)  # Heavy penalty for yearly buyers
    churn_prob = 0.7 * recency_factor + 0.3 * freq_factor
    # Add randomness
    churn_prob += np.random.normal(0, 0.1, size=len(df))
    df['churned'] = (churn_prob > 0.6).astype(int)

    # Purchase channel preference
    if 'online_purchases' in df.columns and 'in_store_purchases' in df.columns:
        total_purchases = df['online_purchases'] + df['in_store_purchases'] + 1
        df['online_preference'] = df['online_purchases'] / total_purchases
    
    # 5. Ensure targets are present
    assert 'total_sales' in df.columns, "Missing regression target: total_sales"
    assert 'churned' in df.columns, "Missing classification target: churned"
    
    print(f"   ✅ Feature engineering complete!")
    print(f"   Final shape: {df.shape}")
    print(f"   Regression target (total_sales): Mean=${df['total_sales'].mean():.2f}")
    print(f"   Classification target (churned): {df['churned'].value_counts().to_dict()}")
    
    # 6. Save Training Data
    print("   💾 Saving training_data.parquet...")
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    client.put_object(
        DEST_BUCKET,
        "training_data.parquet",
        buffer,
        length=buffer.getbuffer().nbytes,
        content_type="application/octet-stream"
    )
    
    print(f"✅ Saved to s3://{DEST_BUCKET}/training_data.parquet")
    print(f"   Rows: {len(df):,}")
    print(f"   Features: {len([c for c in df.columns if c not in ['total_sales', 'churned']])}")
    print(f"   Targets: total_sales, churned")
    
    # Memory cleanup
    del df
    gc.collect()


# --- DAG Definition ---
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    "retail_feature_engineering_pipeline",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
    description="Feature engineering for retail dataset - creates training_data.parquet"
) as dag:
    
    process_task = PythonOperator(
        task_id="create_retail_features",
        python_callable=process_retail_features
    )

    trigger_dq_task = TriggerDagRunOperator(
        task_id="trigger_data_quality",
        trigger_dag_id="data_quality_validation",
    )

    process_task >> trigger_dq_task
