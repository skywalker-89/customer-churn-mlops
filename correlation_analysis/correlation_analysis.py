import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

# Load the retail dataset
print("📊 Loading retail dataset...")
df = pd.read_csv("/Users/jul/Desktop/uni/customer-churn-mlops/data/raw/retail_data.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Convert categorical columns to numeric where needed
df_numeric = df.copy()

# Convert binary categorical columns
binary_cols = [
    "gender",
    "loyalty_program",
    "churned",
    "marital_status",
    "weekend",
    "holiday_season",
]
for col in binary_cols:
    if col in df_numeric.columns:
        if df_numeric[col].dtype == "object":
            # Map common binary values
            if col == "gender":
                df_numeric[col] = df_numeric[col].map(
                    {"Male": 1, "Female": 0, "Other": 2}
                )
            elif col == "loyalty_program":
                df_numeric[col] = df_numeric[col].map({"Yes": 1, "No": 0})
            elif col == "churned":
                df_numeric[col] = df_numeric[col].map({"Yes": 1, "No": 0})
            elif col == "marital_status":
                df_numeric[col] = df_numeric[col].map(
                    {"Married": 1, "Single": 0, "Divorced": 2}
                )
            elif col == "weekend" or col == "holiday_season":
                df_numeric[col] = df_numeric[col].map({"Yes": 1, "No": 0})

# Convert categorical columns with multiple values
categorical_cols = [
    "income_bracket",
    "education_level",
    "occupation",
    "product_category",
    "payment_method",
    "store_location",
    "purchase_frequency",
    "preferred_store",
    "day_of_week",
    "season",
    "promotion_type",
    "promotion_channel",
    "promotion_target_audience",
    "customer_city",
    "customer_state",
    "store_city",
    "store_state",
    "app_usage",
    "social_media_engagement",
]

# For correlation analysis, we'll focus on numeric and encoded categorical columns
print("\n🔢 Selecting numeric and encoded features for correlation analysis...")

# Select only numeric columns for correlation
numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns.tolist()

# Remove ID columns and dates
id_cols = ["customer_id", "transaction_id", "product_id", "promotion_id"]
date_cols = [
    "transaction_date",
    "last_purchase_date",
    "product_manufacture_date",
    "product_expiry_date",
    "promotion_start_date",
    "promotion_end_date",
]

# Filter out non-predictive columns
exclude_cols = id_cols + date_cols + ["customer_zip_code", "store_zip_code"]
analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"Selected {len(analysis_cols)} features for correlation analysis:")
for i, col in enumerate(analysis_cols, 1):
    print(f"{i:2d}. {col}")

# Create correlation matrix
correlation_matrix = df_numeric[analysis_cols].corr()

print(f"\n📈 Correlation matrix shape: {correlation_matrix.shape}")
print("✅ Correlation analysis preparation complete!")
