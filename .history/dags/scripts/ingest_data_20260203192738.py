import os
import pandas as pd
from minio import Minio
from io import BytesIO

# --- Configuration ---
# These settings match your Docker setup
MINIO_ENDPOINT = "minio:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "raw-data"
# This path is where Docker will see the file you just renamed
SOURCE_FILE = "/opt/airflow/data/raw/alibaba_behavior.txt"


def parse_and_upload():
    print("Starting ingestion of Alibaba Data...")

    # Connect to MinIO (Storage)
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )

    # Create the bucket if it doesn't exist
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

    # --- Processing Logic ---
    # We process 50,000 rows at a time to handle the 6GB size safely
    chunk_size = 50000
    chunk_counter = 0

    try:
        # Read the file using semicolon ';' separator and skip the first garbage row
        for chunk in pd.read_csv(
            SOURCE_FILE,
            sep=";",
            header=None,
            skiprows=1,
            chunksize=chunk_size,
            on_bad_lines="skip",
        ):

            df_clean = pd.DataFrame()

            # Extract main columns
            df_clean["page_id"] = chunk[0]
            df_clean["hour"] = chunk[1]
            df_clean["purchase_power"] = chunk[4]
            df_clean["is_churn"] = chunk[14]

            # --- Fix the "List" Columns ---

            # Function to sum up the list of numbers in a cell
            def sum_list_string(val):
                try:
                    return sum([float(i) for i in str(val).split(",") if i.strip()])
                except:
                    return 0.0

            # Calculate Total Value (from Column 12)
            df_clean["page_value"] = chunk[12].apply(sum_list_string)

            # Calculate Total Clicks (from Column 10)
            df_clean["total_clicks"] = chunk[10].apply(sum_list_string)

            # --- Save to MinIO ---
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
        print(f"Error processing file: {e}")

    print("Ingestion Complete!")


if __name__ == "__main__":
    parse_and_upload()
