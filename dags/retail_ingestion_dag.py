from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
from minio import Minio
from io import BytesIO

# --- Configurations ---
MINIO_ENDPOINT = "minio:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
BUCKET_NAME = "raw-data"

# PATHS inside Docker
RAW_DATA_FILE = "/opt/airflow/data/raw/retail_data.csv"

def ingest_retail_data():
    print("🛒 Starting Ingestion of Retail Dataset...") 
    
    # 1. Connect to MinIO
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

    # 2. Check file exists
    if not os.path.exists(RAW_DATA_FILE):
        raise FileNotFoundError(f"Retail data file not found: {RAW_DATA_FILE}")
        
    print(f"📥 Reading {RAW_DATA_FILE}...")
    
    try:
        # Read CSV
        df = pd.read_csv(RAW_DATA_FILE)
        print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Convert to Parquet for storage efficiency
        parquet_name = "retail_data.parquet"
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        # Upload to MinIO
        client.put_object(
            BUCKET_NAME,
            parquet_name,
            buffer,
            length=buffer.getbuffer().nbytes,
            content_type="application/octet-stream",
        )
        print(f"✅ Uploaded {parquet_name} to s3://{BUCKET_NAME}/{parquet_name}")
        print(f"   Columns: {list(df.columns[:10])}... (showing first 10)")
        
    except Exception as e:
        print(f"❌ Failed to ingest retail data: {e}")
        raise e

    print("🎉 Retail data ingested successfully!")

# --- DAG Definition ---
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    "retail_data_ingestion_pipeline",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
    description="Ingest retail dataset (1M rows, 47 features) into MinIO"
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_retail_data",
        python_callable=ingest_retail_data
    )
