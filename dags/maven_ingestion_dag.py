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
RAW_DATA_DIR = "/opt/airflow/data/raw/Toy_Store"

def ingest_maven_data():
    print("🧸 Starting Full Ingestion of Maven Fuzzy Factory...")
    
    # 1. Connect to MinIO
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

    # 2. List all CSV files
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Directory not found: {RAW_DATA_DIR}")
        
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    print(f"Found {len(files)} CSV files to ingest: {files}")

    # 3. Loop and Upload
    for file in files:
        file_path = os.path.join(RAW_DATA_DIR, file)
        
        try:
            print(f"Reading {file}...")
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Convert to Parquet for storage efficiency
            parquet_name = file.replace(".csv", ".parquet")
            buffer = BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            
            # Upload
            client.put_object(
                BUCKET_NAME,
                parquet_name,
                buffer,
                length=buffer.getbuffer().nbytes,
                content_type="application/octet-stream",
            )
            print(f"✅ Uploaded {parquet_name} ({len(df)} rows)")
            
        except Exception as e:
            print(f"❌ Failed to ingest {file}: {e}")
            raise e

    print("🎉 All files ingested successfully!")

# --- DAG Definition ---
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    "maven_ingestion_pipeline",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_toy_store_data",
        python_callable=ingest_maven_data
    )
