import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load the retail dataset
print("📊 Loading retail dataset...")
df = pd.read_csv('/Users/jul/Desktop/uni/customer-churn-mlops/data/raw/retail_data.csv')

# Convert categorical columns to numeric where needed
df_numeric = df.copy()

# Convert binary categorical columns
binary_cols = ['gender', 'loyalty_program', 'churned', 'marital_status', 'weekend', 'holiday_season']
for col in binary_cols:
    if col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            # Map common binary values
            if col == 'gender':
                df_numeric[col] = df_numeric[col].map({'Male': 1, 'Female': 0, 'Other': 2})
            elif col == 'loyalty_program':
                df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})
            elif col == 'churned':
                df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})
            elif col == 'marital_status':
                df_numeric[col] = df_numeric[col].map({'Married': 1, 'Single': 0, 'Divorced': 2})
            elif col == 'weekend' or col == 'holiday_season':
                df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})

# Filter out non-predictive columns
id_cols = ['customer_id', 'transaction_id', 'product_id', 'promotion_id']
date_cols = ['transaction_date', 'last_purchase_date', 'product_manufacture_date', 
             'product_expiry_date', 'promotion_start_date', 'promotion_end_date']
exclude_cols = id_cols + date_cols + ['customer_zip_code', 'store_zip_code']

# Select only numeric columns for correlation
numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns.tolist()
analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

# Create correlation matrix
correlation_matrix = df_numeric[analysis_cols].corr()

print("=" * 80)
print("🔥 REGRESSION ANALYSIS: Correlations with TOTAL_SALES")
print("=" * 80)

# Get correlations with total_sales
total_sales_corr = correlation_matrix['total_sales'].drop('total_sales').sort_values(key=abs, ascending=False)

print(f"\n📈 Top 15 Features Most Correlated with TOTAL_SALES:")
print("-" * 60)
for i, (feature, corr) in enumerate(total_sales_corr.head(15).items(), 1):
    strength = "🔥 Very Strong" if abs(corr) > 0.7 else "⚡ Strong" if abs(corr) > 0.5 else "📊 Moderate" if abs(corr) > 0.3 else "💤 Weak"
    direction = "📈 Positive" if corr > 0 else "📉 Negative"
    print(f"{i:2d}. {feature:<35} {corr:>7.4f} {direction:<12} {strength}")

print(f"\n📉 Bottom 10 Features (Least Correlated with Total Sales):")
print("-" * 60)
for i, (feature, corr) in enumerate(total_sales_corr.tail(10).items(), 1):
    direction = "📈 Positive" if corr > 0 else "📉 Negative"
    print(f"{i:2d}. {feature:<35} {corr:>7.4f} {direction}")

print("\n" + "=" * 80)
print("🎯 CLASSIFICATION ANALYSIS: Correlations with CHURNED")
print("=" * 80)

# Get correlations with churned
churned_corr = correlation_matrix['churned'].drop('churned').sort_values(key=abs, ascending=False)

print(f"\n📊 Top 15 Features Most Correlated with CHURNED:")
print("-" * 60)
for i, (feature, corr) in enumerate(churned_corr.head(15).items(), 1):
    strength = "🔥 Very Strong" if abs(corr) > 0.7 else "⚡ Strong" if abs(corr) > 0.5 else "📊 Moderate" if abs(corr) > 0.3 else "💤 Weak"
    direction = "🚨 Churn Risk" if corr > 0 else "✅ Retention"
    print(f"{i:2d}. {feature:<35} {corr:>7.4f} {direction:<15} {strength}")

print(f"\n🛡️ Bottom 10 Features (Least Correlated with Churn):")
print("-" * 60)
for i, (feature, corr) in enumerate(churned_corr.tail(10).items(), 1):
    direction = "🚨 Churn Risk" if corr > 0 else "✅ Retention"
    print(f"{i:2d}. {feature:<35} {corr:>7.4f} {direction}")

# Statistical significance testing for top correlations
print("\n" + "=" * 80)
print("📊 STATISTICAL SIGNIFICANCE TESTING")
print("=" * 80)

print(f"\n🔍 Testing Statistical Significance of Top Correlations:")
print("-" * 70)
print(f"{'Feature':<35} {'Correlation':<12} {'P-value':<12} {'Significance'}")
print("-" * 70)

# Test significance for top 10 correlations with total_sales
for feature in total_sales_corr.head(10).index:
    corr_coef, p_value = pearsonr(df_numeric['total_sales'].dropna(), 
                                  df_numeric[feature].dropna())
    significance = "⭐***" if p_value < 0.001 else "⭐**" if p_value < 0.01 else "⭐*" if p_value < 0.05 else "❌"
    print(f"{feature:<35} {corr_coef:>8.4f}    {p_value:>8.2e}    {significance}")

print("\n" + "=" * 80)
print("🔄 CROSS-TARGET ANALYSIS")
print("=" * 80)

# Compare how features correlate with both targets
comparison_df = pd.DataFrame({
    'Feature': analysis_cols,
    'Total_Sales_Corr': [correlation_matrix.loc[f, 'total_sales'] for f in analysis_cols],
    'Churned_Corr': [correlation_matrix.loc[f, 'churned'] for f in analysis_cols],
})

# Remove target variables from comparison
comparison_df = comparison_df[~comparison_df['Feature'].isin(['total_sales', 'churned'])]

# Find features that correlate strongly with both targets
comparison_df['Abs_Sales_Corr'] = comparison_df['Total_Sales_Corr'].abs()
comparison_df['Abs_Churned_Corr'] = comparison_df['Churned_Corr'].abs()
comparison_df['Avg_Correlation'] = (comparison_df['Abs_Sales_Corr'] + comparison_df['Abs_Churned_Corr']) / 2

print(f"\n🔗 Features with Strong Correlations to BOTH Targets:")
print("-" * 80)
print(f"{'Feature':<35} {'Sales Corr':<12} {'Churn Corr':<12} {'Avg Strength':<12}")
print("-" * 80)

# Sort by average correlation strength
top_dual_features = comparison_df.nlargest(10, 'Avg_Correlation')
for _, row in top_dual_features.iterrows():
    print(f"{row['Feature']:<35} {row['Total_Sales_Corr']:>8.4f}    {row['Churned_Corr']:>8.4f}    {row['Avg_Correlation']:>8.4f}")

print("\n✅ Correlation analysis complete!")
print(f"📊 Dataset: {df.shape[0]:,} customers with {len(analysis_cols)} features analyzed")
print(f"🎯 Targets: total_sales (regression) & churned (classification)")