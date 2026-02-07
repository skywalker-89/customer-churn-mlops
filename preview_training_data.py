
import pandas as pd
from minio import Minio
from io import BytesIO

MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
BUCKET_NAME = "processed-data"
FILE_NAME = "training_data.parquet"

print(f"Reading {FILE_NAME} from bucket {BUCKET_NAME}...")

client = Minio(
    MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
)

try:
    response = client.get_object(BUCKET_NAME, FILE_NAME)
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    
    print("\n--- DATA SHAPE ---")
    print(df.shape)
    
    print("\n--- COLUMNS ---")
    print(df.columns.tolist())
    
    print("\n--- HEAD (5 rows) ---")
    print(df.head())
    
    print("\n--- SAMPLE (5 random rows) ---")
    print(df.sample(5))

except Exception as e:
    print(f"Error reading file: {e}")
