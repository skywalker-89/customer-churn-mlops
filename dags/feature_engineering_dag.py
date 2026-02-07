from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from minio import Minio
from io import BytesIO
import gc

# --- Configuration ---
MINIO_ENDPOINT = "minio:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
SOURCE_BUCKET = "raw-data"
DEST_BUCKET = "processed-data"


def process_features():
    """Feature Engineering for Maven Fuzzy Factory Dataset"""
    print("🧪 Starting Feature Engineering (Lead DS)...")
    
    # 1. Connect to MinIO
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )
    
    # Ensure processed bucket exists
    if not client.bucket_exists(DEST_BUCKET):
        client.make_bucket(DEST_BUCKET)
    
    # 2. Load Raw Data (Parquet from Ingestion)
    def read_minio_parquet(filename):
        print(f"   Loading {filename}...")
        obj = client.get_object(SOURCE_BUCKET, filename)
        return pd.read_parquet(BytesIO(obj.read()))
    
    try:
        df_sessions = read_minio_parquet("website_sessions.parquet")
        df_orders = read_minio_parquet("orders.parquet")
        df_pageviews = read_minio_parquet("website_pageviews.parquet")
    except Exception as e:
        print(f"❌ Could not load data from MinIO: {e}")
        raise e
    
    # 3. FEATURE 1 & 2: Traffic Source & Device (Already in sessions)
    print("   Feature 1 (Source) & 2 (Device): Ready.")
    
    # 4. FEATURE 3: Time of Day (Transform)
    print("   Feature 3: Extracting Time of Day...")
    df_sessions['created_at'] = pd.to_datetime(df_sessions['created_at'])
    df_sessions['hour_of_day'] = df_sessions['created_at'].dt.hour
    df_sessions['is_weekend'] = df_sessions['created_at'].dt.weekday >= 5
    
    # 5. FEATURE 4: Landing Page (First page viewed in session)
    print("   Feature 4: Computing Landing Page...")
    df_pageviews['created_at'] = pd.to_datetime(df_pageviews['created_at'])
    landing_pages = df_pageviews.sort_values('created_at').groupby('website_session_id').first()['pageview_url'].reset_index()
    landing_pages.columns = ['website_session_id', 'landing_page']
    df_sessions = pd.merge(df_sessions, landing_pages, on='website_session_id', how='left')
    
    # 6. FEATURE 5: Engagement Depth (Number of pages viewed)
    print("   Feature 5: Computing Engagement Depth...")
    engagement = df_pageviews.groupby('website_session_id').size().reset_index(name='engagement_depth')
    df_sessions = pd.merge(df_sessions, engagement, on='website_session_id', how='left')
    df_sessions['engagement_depth'] = df_sessions['engagement_depth'].fillna(0).astype(int)
    
    # 7. MERGE TARGETS (Orders)
    print("   Merging Targets...")
    df_final = pd.merge(
        df_sessions,
        df_orders[['website_session_id', 'order_id', 'price_usd']],
        on='website_session_id',
        how='left'
    )
    
    # Targets
    df_final['is_ordered'] = df_final['order_id'].notnull().astype(int)
    df_final['revenue'] = df_final['price_usd'].fillna(0.0)
    
    # 8. Select Columns for ML
    features = [
        # Raw features from sessions
        'utm_source', 'device_type', 'is_repeat_session',
        # Engineered features
        'hour_of_day', 'is_weekend', 'landing_page', 'engagement_depth',
        # Targets
        'is_ordered', 'revenue'
    ]
    
    training_data = df_final[features].copy()
    
    # 9. One-Hot Encoding (Preprocessing)
    print("   Encoding Categoricals...")
    training_data = pd.get_dummies(
        training_data, 
        columns=['utm_source', 'device_type', 'landing_page'], 
        drop_first=True
    )
    
    # 10. Save Final Matrix
    print("   Saving Training Matrix...")
    buffer = BytesIO()
    training_data.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    client.put_object(
        DEST_BUCKET,
        "training_data.parquet",
        buffer,
        length=buffer.getbuffer().nbytes,
        content_type="application/octet-stream"
    )
    
    print("✅ Feature Engineering Complete!")
    print(f"   Saved to {DEST_BUCKET}/training_data.parquet")
    print(f"   Rows: {len(training_data)}")
    print(f"   Columns: {training_data.columns.tolist()}")
    
    # Memory cleanup
    del df_sessions, df_orders, df_pageviews, df_final, training_data
    gc.collect()



# --- DAG Definition ---
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    "feature_engineering_pipeline",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
) as dag:

    process_task = PythonOperator(
        task_id="create_rl_features", python_callable=process_features
    )
