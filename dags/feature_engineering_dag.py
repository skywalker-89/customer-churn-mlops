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
    print("Starting Feature Engineering...")

    # 1. Connect to MinIO
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )

    # 2. Get list of all raw files
    objects = client.list_objects(SOURCE_BUCKET)
    file_list = [
        obj.object_name for obj in objects if obj.object_name.endswith(".parquet")
    ]
    print(f"Found {len(file_list)} files to process.")

    # 3. Process each file one by one
    for file_name in file_list:
        print(f"Processing {file_name}...")

        try:
            # Read from MinIO
            response = client.get_object(SOURCE_BUCKET, file_name)
            df = pd.read_parquet(BytesIO(response.read()))
            response.close()
            response.release_conn()

            # --- THE MATH (RL Logic) ---

            # A. Calculate REWARD (The Goal)
            # Logic: If they bought something, high reward. If they clicked, small reward.
            # Reward = (Purchase Amount * 1.0) + (Clicks * 0.5)
            df["reward"] = (df["page_value"] * 1.0) + (df["total_clicks"] * 0.5)

            # B. Define STATE ( The Context)
            # We normalize 'purchase_power' to be between 0 and 1
            df["state_purchase_power"] = (
                df["purchase_power"] / 10.0
            )  # Assuming max level is ~10

            # C. Define ACTION (What happened)
            # In this dataset, the 'Action' was the page they were shown (page_id)
            df["action_page_id"] = df["page_id"]

            # D. Define NEXT STATE / TERMINAL
            # If is_churn = 1, the episode ends (Terminal = True)
            df["done"] = df["is_churn"].apply(lambda x: 1 if x == 1 else 0)

            # --- Save to Processed Bucket ---
            # Select only the columns the AI needs
            training_data = df[
                ["state_purchase_power", "action_page_id", "reward", "done"]
            ]

            save_name = f"train_{file_name}"
            parquet_buffer = BytesIO()
            training_data.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)

            client.put_object(
                DEST_BUCKET,
                save_name,
                parquet_buffer,
                length=parquet_buffer.getbuffer().nbytes,
                content_type="application/octet-stream",
            )
            print(f"Saved {save_name} to {DEST_BUCKET}")

            # Memory Cleanup (Crucial)
            del df
            del training_data
            del parquet_buffer
            gc.collect()

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


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
