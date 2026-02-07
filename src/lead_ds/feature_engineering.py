import pandas as pd
from minio import Minio
from io import BytesIO

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
BUCKET_NAME = "raw-data"
PROCESSED_BUCKET = "processed-data"

def run_feature_engineering():
    print("🧪 Starting Feature Engineering (Lead DS)...")
    
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )
    
    # Ensure processed bucket exists
    if not client.bucket_exists(PROCESSED_BUCKET):
        client.make_bucket(PROCESSED_BUCKET)

    # 1. Load Raw Data (Parquet from Ingestion)
    # Note: In real production, we'd read specific files. For this demo, we assume the previous step ran.
    # We will read the RAW CSVs from local for simplicity if MinIO isn't fully piped yet, 
    # OR read the artifacts we just made. Let's try reading the converted parquets from MinIO.
    
    def read_minio_parquet(filename):
        print(f"   Loading {filename}...")
        obj = client.get_object(BUCKET_NAME, filename)
        return pd.read_parquet(BytesIO(obj.read()))

    # Assuming ingestion created these. If not, we might need to rely on the DAG having run.
    # For safety/presentation, if MinIO load fails, we fallback to local CSVs?
    # Let's stick to the "Correct" path: Read from MinIO.
    
    try:
        df_sessions = read_minio_parquet("website_sessions.parquet")
        df_orders = read_minio_parquet("orders.parquet")
        # df_pageviews = read_minio_parquet("website_pageviews.parquet") # Optional for advanced
    except Exception as e:
        print(f"❌ Could not verify data in MinIO. Have you run the Airflow DAG yet?\nError: {e}")
        return

    # 2. FEATURE 1 & 2: Traffic Source & Device (Already in sessions)
    print("   Feature 1 (Source) & 2 (Device): Ready.")
    
    # 3. FEATURE 5: Time of Day (Transform)
    print("   Feature 5: Extracting Time of Day...")
    df_sessions['created_at'] = pd.to_datetime(df_sessions['created_at'])
    df_sessions['hour_of_day'] = df_sessions['created_at'].dt.hour
    df_sessions['is_weekend'] = df_sessions['created_at'].dt.weekday >= 5
    
    # 4. MERGE TARGETS (Orders)
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
    
    # Select Columns for ML
    features = [
        'utm_source', 'device_type', 'hour_of_day', 'is_weekend', # Features
        'is_ordered', 'revenue' # Targets
    ]
    # Note: 'landing_page' and 'engagement_depth' require joining pageviews.
    # We start with this MVP set to get 5 features (Source, Device, Hour, Weekend, +1 more?)
    # Let's add 'is_repeat'
    features.append('is_repeat_session') 
    
    training_data = df_final[features].copy()
    
    # 5. One-Hot Encoding (Preprocessing)
    # ML models usually need numbers, not strings.
    print("   Encoding Categoricals...")
    training_data = pd.get_dummies(training_data, columns=['utm_source', 'device_type'], drop_first=True)
    
    # 6. Save Final Matrix
    print("   Saving Training Matrix...")
    buffer = BytesIO()
    training_data.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    client.put_object(
        PROCESSED_BUCKET,
        "training_data.parquet",
        buffer,
        length=buffer.getbuffer().nbytes,
        content_type="application/octet-stream"
    )
    
    print("✅ Feature Engineering Complete!")
    print(f"   Saved to s3://{PROCESSED_BUCKET}/training_data.parquet")
    print("   Rows:", len(training_data))
    print("   Columns:", training_data.columns.tolist())

if __name__ == "__main__":
    run_feature_engineering()
