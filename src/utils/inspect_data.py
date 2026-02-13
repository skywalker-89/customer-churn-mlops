import pandas as pd
import numpy as np
from minio import Minio
from io import BytesIO
import os

def inspect():
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minio_password")
    
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
    
    print("Downloading training_data.parquet...")
    obj = client.get_object("processed-data", "training_data.parquet")
    df = pd.read_parquet(BytesIO(obj.read()))
    
    print(f"Shape: {df.shape}")
    print("\nColumns:", list(df.columns))
    
    print("\nSample Data (First 5 rows):")
    print(df[['total_sales', 'avg_purchase_value', 'total_returned_items']].head())
    
    print("\nFeature Statistics:")
    print(df[['total_sales', 'avg_purchase_value', 'total_returned_items']].describe())
    
    print("\nCorrelations:")
    cols_to_check = ['total_sales', 'avg_purchase_value', 'quantity', 'unit_price']
    print(df[cols_to_check].corr())
    
    print("\nSanity Check: total_sales vs (total_transactions * avg_transaction_value)")
    df['calculated_clv'] = df['total_transactions'] * df['avg_transaction_value']
    print(df[['total_sales', 'calculated_clv']].head())
    print("\nCorrelation with calculated CLV:", df['total_sales'].corr(df['calculated_clv']))
    
    print("\nSanity Check 2: total_sales vs total_transactions")
    print("Correlation:", df['total_sales'].corr(df['total_transactions']))

if __name__ == "__main__":
    inspect()
