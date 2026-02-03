import pandas as pd
from minio import Minio
from io import BytesIO

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"  # Localhost because we are running from terminal
ACCESS_KEY = "minio_admin"  # <--- PUT YOUR REAL PASSWORD HERE
SECRET_KEY = "minioadmin"  # <--- PUT YOUR REAL PASSWORD HERE
BUCKET_NAME = "processed-data"


def inspect_bucket():
    print(f"🔍 Connecting to MinIO bucket: {BUCKET_NAME}...")

    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )

    # 1. Check if files exist
    objects = client.list_objects(BUCKET_NAME)
    files = [obj.object_name for obj in objects]

    if not files:
        print("❌ ERROR: Bucket is empty!")
        return

    print(f"✅ Found {len(files)} files. Inspecting the first one: {files[0]}...\n")

    # 2. Download and Read the First File
    try:
        response = client.get_object(BUCKET_NAME, files[0])
        df = pd.read_parquet(BytesIO(response.read()))
        response.close()
        response.release_conn()

        # 3. Show the Data
        print("--- 📊 DATA PREVIEW (First 5 Rows) ---")
        print(df.head())
        print("\n")

        print("--- ℹ️ COLUMN INFO ---")
        print(df.info())
        print("\n")

        print("--- 📈 STATISTICS ---")
        print(df.describe())

        # 4. Sanity Checks
        print("\n--- 🧠 SANITY CHECK ---")
        if df["reward"].isnull().sum() > 0:
            print("⚠️ WARNING: Found NULL values in 'reward' column!")
        else:
            print("✅ 'reward' column looks clean.")

        if df["state_purchase_power"].max() > 1.0:
            print(
                "⚠️ WARNING: 'state_purchase_power' is not normalized (values > 1.0 found)."
            )
        else:
            print("✅ 'state_purchase_power' is properly normalized (0.0 - 1.0).")

    except Exception as e:
        print(f"❌ CRITICAL ERROR READING FILE: {e}")


if __name__ == "__main__":
    inspect_bucket()
