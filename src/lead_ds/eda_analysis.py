import pandas as pd
from minio import Minio
from io import BytesIO
import random

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
BUCKET_NAME = "raw-data"
MAX_FILES = 200 # Scan up to 200 files
TARGET_CHURNS = 1000 # Stop if we find this many churns

def run_eda():
    print(f"📊 Starting Deep Search for Churns on bucket: {BUCKET_NAME}...")

    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )

    objects = client.list_objects(BUCKET_NAME)
    files = [obj.object_name for obj in objects]
    
    # Shuffle files to avoid biases in sequential chunks
    random.shuffle(files)
    
    print(f"Scanning up to {MAX_FILES} files from {len(files)} total files...")
    
    df_list = []
    total_churns = 0
    files_scanned = 0
    
    for f in files:
        if files_scanned >= MAX_FILES:
            break
            
        try:
            response = client.get_object(BUCKET_NAME, f)
            df_chunk = pd.read_parquet(BytesIO(response.read()))
            response.close()
            response.release_conn()
            
            # Check for churns
            churns_in_chunk = df_chunk['is_churn'].sum()
            
            if churns_in_chunk > 0 or len(df_list) < 5: # Keep some data anyway, but prioritize churns
                df_list.append(df_chunk)
                total_churns += churns_in_chunk
                print(f"Found {churns_in_chunk} churns in {f}")
            
            files_scanned += 1
            
            if total_churns >= TARGET_CHURNS:
                print(f"✅ Reached target churn count: {total_churns}")
                break
                
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        print("No data loaded.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Loaded {len(full_df)} rows with {total_churns} churns.")

    if total_churns == 0:
        print("❌ Still no churns found! Check dataset generation.")
        return

    # --- CORRELATION ANALYSIS ---
    print("\n--- 🔗 CORRELATION WITH CHURN (Target: is_churn) ---")
    # Ensure numeric types
    for col in full_df.columns:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
        
    correlation = full_df.corr()['is_churn'].sort_values(ascending=False)
    print(correlation)
    
    # --- FEATURE ANALYSIS ---
    print("\n--- 🕵️ FEATURE DISTRIBUTIONS ---")
    cols_to_check = ['purchase_power', 'hour', 'page_value', 'click_count', 'cart_count', 'fav_count', 'page_id']
    existing_cols = [c for c in cols_to_check if c in full_df.columns]
    
    print(full_df.groupby('is_churn')[existing_cols].mean())

    # --- DERIVED FEATURE IDEAS ---
    if 'click_count' in full_df.columns:
        full_df['power_click_interaction'] = full_df['purchase_power'] * full_df['click_count']
        full_df['intent_sum'] = full_df['cart_count'] + full_df['fav_count']
        
        print("\n--- 🧪 DERIVED FEATURE CORRELATION ---")
        derived_corr = full_df[['power_click_interaction', 'intent_sum', 'is_churn']].corr()['is_churn']
        print(derived_corr)

    return correlation

if __name__ == "__main__":
    run_eda()
