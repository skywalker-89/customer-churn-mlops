import pandas as pd
import os

# Local Paths for testing
SESSION_FILE = "data/raw/Toy_Store/website_sessions.csv"
ORDER_FILE = "data/raw/Toy_Store/orders.csv"

def test_ingestion_logic():
    print("🧸 DRY RUN: Maven Fuzzy Factory Ingestion...")
    
    if not os.path.exists(SESSION_FILE):
        print(f"❌ File not found: {SESSION_FILE}")
        return

    # Load
    print("Reading CSVs...")
    df_sessions = pd.read_csv(SESSION_FILE)
    df_orders = pd.read_csv(ORDER_FILE)
    
    print(f"Sessions: {len(df_sessions)} rows")
    print(f"Orders: {len(df_orders)} rows")
    
    # Merge
    print("Merging...")
    df_merged = pd.merge(
        df_sessions, 
        df_orders[['website_session_id', 'order_id', 'price_usd']], 
        on='website_session_id', 
        how='left'
    )
    
    # Targets
    df_merged['is_ordered'] = df_merged['order_id'].notnull().astype(int)
    df_merged['revenue'] = df_merged['price_usd'].fillna(0.0)
    
    # Stats
    conversion_rate = df_merged['is_ordered'].mean()
    total_rev = df_merged['revenue'].sum()
    
    print("\n--- 📊 DATA PREVIEW ---")
    print(df_merged[['website_session_id', 'utm_source', 'device_type', 'is_ordered', 'revenue']].head(10))
    
    print("\n--- 📈 KEY METRICS ---")
    print(f"Total Sessions: {len(df_merged)}")
    print(f"Conversion Rate: {conversion_rate:.2%}")
    print(f"Total Revenue: ${total_rev:,.2f}")
    
    if conversion_rate > 0 and total_rev > 0:
        print("\n✅ SUCCESS: Classification and Regression Targets created successfully!")
    else:
        print("\n❌ FAILURE: Targets are empty.")

if __name__ == "__main__":
    test_ingestion_logic()
