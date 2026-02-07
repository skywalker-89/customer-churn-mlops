import pandas as pd
from minio import Minio
from io import BytesIO
import os
import gc

# --- Configuration ---
# Localhost for running outside Docker
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
BUCKET_NAME = "raw-data"
SOURCE_FILE = "data/raw/alibaba_behavior.txt" # Relative to project root

def ingest_data_manual():
    print("🚀 Starting MANUAL ingestion of Alibaba Data...")

    # 1. Connect to MinIO
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )

    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        print(f"Created bucket {BUCKET_NAME}")

    # 2. Process Data
    chunk_size = 5000
    chunk_counter = 0

    if not os.path.exists(SOURCE_FILE):
        print(f"❌ ERROR: File not found at {SOURCE_FILE}")
        return

    print(f"Reading from {SOURCE_FILE}...")

    try:
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

            # Fix Lists - Extract Interactions
            def sum_list_string(val):
                try:
                    return sum([float(i) for i in str(val).split(",") if i.strip()])
                except:
                    return 0.0

            df_clean["page_value"] = chunk[12].apply(sum_list_string)
            df_clean["click_count"] = chunk[9].apply(sum_list_string) # Col 10: IsClick
            df_clean["cart_count"] = chunk[10].apply(sum_list_string) # Col 11: IsCart
            df_clean["fav_count"] = chunk[11].apply(sum_list_string)  # Col 12: IsFav

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

            print(f"✅ Uploaded {file_name} with {len(df_clean)} rows.")
            chunk_counter += 1
            
            # Limit for demo/dev speed - Stop after 20 chunks (100k rows)
            # Remove this break to process full file
            if chunk_counter >= 20:
                print("🛑 Stopping after 20 chunks for development speed.")
                break

            # Cleanup
            del df_clean
            del parquet_buffer
            gc.collect()

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    ingest_data_manual()
