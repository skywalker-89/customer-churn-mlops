
import pandas as pd
from minio import Minio
from io import BytesIO

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
BUCKET_NAME = "processed-data"
TRAINING_FILE = "training_data.parquet"

def analyze_assumptions():
    print("📊 Connecting to MinIO...")
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )
    
    try:
        # Load Data
        print(f"   Loading {TRAINING_FILE}...")
        obj = client.get_object(BUCKET_NAME, TRAINING_FILE)
        df = pd.read_parquet(BytesIO(obj.read()))
        
        # 1. Traffic Source Analysis (Paid vs Organic Hypothesis)
        print("\n--- 1. Traffic Source Analysis ---")
        # We need to see if we have source columns.
        # Based on previous `preview_training_data.py`, we have:
        # 'utm_source_gsearch', 'utm_source_socialbook' 
        # But we lost the raw 'utm_source' string column in the encoding step.
        # However, we can reconstruct it or just analyze the One-Hot columns.
        
        # Let's see conversion rate for gsearch vs socialbook
        # We can group by the presence of these flags
        
        # Gsearch Conversion Rate
        gsearch_conversions = df[df['utm_source_gsearch'] == True]['is_ordered'].mean()
        print(f"   GSearch Conversion Rate: {gsearch_conversions:.2%}")
        
        # Socialbook Conversion Rate
        socialbook_conversions = df[df['utm_source_socialbook'] == True]['is_ordered'].mean()
        print(f"   Socialbook Conversion Rate: {socialbook_conversions:.2%}")
        
        # 'Other' sources (where both are False? Need to check if there are others)
        other_conversions = df[(df['utm_source_gsearch'] == False) & (df['utm_source_socialbook'] == False)]['is_ordered'].mean()
        print(f"   Other Sources Conversion Rate: {other_conversions:.2%}")
        
        # 2. Class Imbalance
        print("\n--- 2. Class Imbalance ---")
        class_counts = df['is_ordered'].value_counts()
        print(class_counts)
        print(f"   Ratio: 1:{class_counts[0]/class_counts[1]:.1f} (Neg:Pos)")
        
        # 3. Revenue Target (Session vs CLV)
        print("\n--- 3. Revenue Target Definition ---")
        # We just need to check if revenue > 0 ONLY when is_ordered = 1
        # And if revenue aligns with single session 'price_usd' (we saw this in feature_engineering.py)
        # But let's check the distribution.
        
        revenue_stats = df[df['is_ordered'] == 1]['revenue'].describe()
        print(f"   Revenue Stats for Orders:\n{revenue_stats}")
        
        max_rev = df['revenue'].max()
        print(f"   Max Revenue in single row: ${max_rev:.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    analyze_assumptions()
