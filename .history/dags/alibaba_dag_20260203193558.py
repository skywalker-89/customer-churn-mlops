from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from minio import Minio
from io import BytesIO
import os

# --- Configurations ---
MINIO_ENDPOINT = "minio:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "raw-data"
SOURCE_FILE = "/opt/airflow/data/raw/alibaba_behavior.txt"


def ingest_data_logic():
    print("Starting ingestion of Alibaba Data...")

    # 1. Connect to MinIO
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )

    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

    # 2. Process Data
    chunk_size = 50000
    chunk_counter = 0

    # Read using semi-colon separator
    try:
        # Check if file exists first
        if not os.path.exists(SOURCE_FILE):
            print(f"ERROR: File not found at {SOURCE_FILE}")
            return

        for chunk in pd.read_csv(
            SOURCE_FILE,
            sep=";",
            header=None,
            skiprows=1,
            chunksize=chunk_size,
            on_bad_lines="skip",
        ):

            df_clean = pd.DataFrame()

            # Map columns
            df_clean["page_id"] = chunk[0]
            df_clean["hour"] = chunk[1]
            df_clean["purchase_power"] = chunk[4]
            df_clean["is_churn"] = chunk[14]

            # Fix Lists
            def sum_list_string(val):
                try:
                    return sum([float(i) for i in str(val).split(",") if i.strip()])
                except:
                    return 0.0

            df_clean["page_value"] = chunk[12].apply(sum_list_string)
            df_clean["total_clicks"] = chunk[10].apply(sum_list_string)

            # Upload to MinIO
            file_name = f"alibaba_chunk_{chunk_counter}.parquet"
            parquet_buffer = BytesIO()
            df_clean.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)

            client.put_object(
                BUCKET_NAME,
                file_name,
                parquet_buffer,
                length=parquet_buffer.getbuffer().nbytes,
                content_type="application/octet-stream",
            )

            print(f"Uploaded {file_name} with {len(df_clean)} rows.")
            chunk_counter += 1

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise e


# --- The DAG Definition (This makes it show up in UI) ---
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    "alibaba_ingestion_pipeline",  # <--- This name will show in the UI
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_alibaba_data", python_callable=ingest_data_logic
    )
