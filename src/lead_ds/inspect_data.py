import pandas as pd
from minio import Minio
from io import BytesIO

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
BUCKET_NAME = "raw-data"  # <--- Looking at RAW data now

def inspect_bucket():
    print(f"🔍 Connecting to MinIO bucket: {BUCKET_NAME}...")

    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )

    # 1. Check if files exist
    try:
        if not client.bucket_exists(BUCKET_NAME):
            print(f"❌ ERROR: Bucket '{BUCKET_NAME}' does not exist!")
            return
            
        objects = client.list_objects(BUCKET_NAME)
        files = [obj.object_name for obj in objects]

        if not files:
            print("❌ ERROR: Bucket is empty!")
            return

        print(f"✅ Found {len(files)} files. Inspecting the first one: {files[0]}...\n")

        # 2. Download and Read the First File
        response = client.get_object(BUCKET_NAME, files[0])
        # Read parquet file
        df = pd.read_parquet(BytesIO(response.read()))
        response.close()
        response.release_conn()

        # 3. Show the Data
        print("--- 📊 DATA PREVIEW (First 5 Rows) ---")
        pd.set_option('display.max_columns', None) # Show all columns
        print(df.head())
        print("\n")

        print("--- ℹ️ COLUMN INFO ---")
        print(df.info())
        print("\n")

        print("--- 📈 STATISTICS ---")
        print(df.describe())

    except Exception as e:
        print(f"❌ CRITICAL ERROR READING FILE: {e}")

if __name__ == "__main__":
    inspect_bucket()
