import pandas as pd
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.core.minio_client import minio_client
except ImportError:
    # If backend module not found, try adding src
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    from backend.core.minio_client import minio_client


def check_data():
    print("Attempting to load data from MinIO (training_data.parquet)...")
    df = minio_client.get_data("training_data.parquet")

    if df is None:
        print("Failed to load data from MinIO.")
        return

    print(f"Loaded data with shape: {df.shape}")
    print("Columns:")
    for col in df.columns:
        print(f" - {col}")

    print("\nFeature importance check (correlation with churned):")
    if "churned" in df.columns:
        corrs = df.corr()["churned"].sort_values(ascending=False)
        print("Top 10 positive correlations:")
        print(corrs.head(10))
        print("\nTop 10 negative correlations:")
        print(corrs.tail(10))

        # Check specific features
        print("\nSpecific feature correlations:")
        specifics = [
            "engagement_score",
            "quantity_times_price",
            "recency_score",
            "online_preference",
        ]
        for s in specifics:
            if s in df.columns:
                print(f"{s}: {corrs.get(s, 'N/A')}")
            else:
                print(f"{s}: Not in dataframe")
    else:
        print("Column 'churned' not found.")


if __name__ == "__main__":
    check_data()
