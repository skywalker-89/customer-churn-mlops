
from minio import Minio

MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
BUCKET_NAME = "processed-data"

client = Minio(
    MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
)

if not client.bucket_exists(BUCKET_NAME):
    print(f"Bucket {BUCKET_NAME} does not exist.")
else:
    objects = client.list_objects(BUCKET_NAME)
    print(f"Files in {BUCKET_NAME}:")
    for obj in objects:
        print(f" - {obj.object_name}")
