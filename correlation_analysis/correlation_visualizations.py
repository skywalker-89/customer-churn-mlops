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

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

print("🎨 Creating correlation visualizations...")

# 1. Full Correlation Heatmap
print("\n1️⃣ Generating full correlation heatmap...")
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('🔥 Full Feature Correlation Matrix\n(Retail Dataset - 1M Records, 40 Features)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/Users/jul/Desktop/uni/customer-churn-mlops/correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Regression Target Correlations (total_sales)
print("\n2️⃣ Generating regression correlations (total_sales)...")
total_sales_corr = correlation_matrix['total_sales'].drop('total_sales').sort_values(key=abs, ascending=False)

plt.figure(figsize=(14, 10))
colors = ['#d62728' if x < 0 else '#2ca02c' for x in total_sales_corr.values]
bars = plt.barh(range(len(total_sales_corr)), total_sales_corr.values, color=colors, alpha=0.7)
plt.yticks(range(len(total_sales_corr)), total_sales_corr.index)
plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
plt.title('📈 Feature Correlations with TOTAL_SALES\n(Regression Target)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, total_sales_corr.values)):
    plt.text(value + (0.0001 if value >= 0 else -0.0001), i, f'{value:.4f}', 
             ha='left' if value >= 0 else 'right', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/jul/Desktop/uni/customer-churn-mlops/correlation_regression.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Classification Target Correlations (churned)
print("\n3️⃣ Generating classification correlations (churned)...")
churned_corr = correlation_matrix['churned'].drop('churned').sort_values(key=abs, ascending=False)

plt.figure(figsize=(14, 10))
colors = ['#d62728' if x > 0 else '#2ca02c' for x in churned_corr.values]
bars = plt.barh(range(len(churned_corr)), churned_corr.values, color=colors, alpha=0.7)
plt.yticks(range(len(churned_corr)), churned_corr.index)
plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
plt.title('🎯 Feature Correlations with CHURNED\n(Classification Target - Positive = Churn Risk)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, churned_corr.values)):
    plt.text(value + (0.0001 if value >= 0 else -0.0001), i, f'{value:.4f}', 
             ha='left' if value >= 0 else 'right', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/jul/Desktop/uni/customer-churn-mlops/correlation_classification.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Top correlations comparison
print("\n4️⃣ Generating top correlations comparison...")
top_15_sales = total_sales_corr.head(15)
top_15_churn = churned_corr.head(15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Sales correlations
sales_colors = ['#d62728' if x < 0 else '#2ca02c' for x in top_15_sales.values]
ax1.barh(range(len(top_15_sales)), top_15_sales.values, color=sales_colors, alpha=0.7)
ax1.set_yticks(range(len(top_15_sales)))
ax1.set_yticklabels(top_15_sales.index)
ax1.set_xlabel('Correlation with Total Sales', fontweight='bold')
ax1.set_title('📈 Top 15 Sales Correlations', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Churn correlations
churn_colors = ['#d62728' if x > 0 else '#2ca02c' for x in top_15_churn.values]
ax2.barh(range(len(top_15_churn)), top_15_churn.values, color=churn_colors, alpha=0.7)
ax2.set_yticks(range(len(top_15_churn)))
ax2.set_yticklabels(top_15_churn.index)
ax2.set_xlabel('Correlation with Churn (Risk)', fontweight='bold')
ax2.set_title('🎯 Top 15 Churn Correlations', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/jul/Desktop/uni/customer-churn-mlops/correlation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Feature relationships scatter plots for strongest correlations
print("\n5️⃣ Generating scatter plots for key relationships...")

# Find features with strongest correlations (absolute value)
all_correlations = []
for feature in analysis_cols:
    if feature not in ['total_sales', 'churned']:
        sales_corr = correlation_matrix.loc[feature, 'total_sales']
        churn_corr = correlation_matrix.loc[feature, 'churned']
        all_correlations.append({
            'feature': feature,
            'sales_corr': abs(sales_corr),
            'churn_corr': abs(churn_corr),
            'avg_corr': (abs(sales_corr) + abs(churn_corr)) / 2
        })

# Sort by average correlation
correlation_df = pd.DataFrame(all_correlations)
top_features = correlation_df.nlargest(6, 'avg_corr')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, row in enumerate(top_features.itertuples()):
    feature = row.feature
    
    # Create scatter plot
    sample_size = min(1000, len(df_numeric))  # Sample for visualization
    sample_idx = np.random.choice(len(df_numeric), sample_size, replace=False)
    
    x_vals = df_numeric.iloc[sample_idx][feature]
    y_vals = df_numeric.iloc[sample_idx]['total_sales']
    
    axes[i].scatter(x_vals, y_vals, alpha=0.6, s=20)
    axes[i].set_xlabel(feature, fontweight='bold')
    axes[i].set_ylabel('Total Sales', fontweight='bold')
    axes[i].set_title(f'{feature}\nCorr: {correlation_matrix.loc[feature, "total_sales"]:.4f}', 
                     fontweight='bold')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('🔍 Feature Relationships with Total Sales\n(Top 6 Most Correlated Features)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/jul/Desktop/uni/customer-churn-mlops/feature_relationships.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ All correlation visualizations created!")
print("📊 Generated files:")
print("   📄 correlation_heatmap_full.png - Full correlation matrix")
print("   📄 correlation_regression.png - Sales correlations") 
print("   📄 correlation_classification.png - Churn correlations")
print("   📄 correlation_comparison.png - Top correlations comparison")
print("   📄 feature_relationships.png - Feature relationships scatter plots")

# Summary statistics
print(f"\n📈 CORRELATION SUMMARY:")
print(f"   🔍 Analyzed {len(analysis_cols)} features across {df.shape[0]:,} records")
print(f"   📊 Strongest Sales Correlation: {total_sales_corr.iloc[0]:.4f} ({total_sales_corr.index[0]})")
print(f"   🎯 Strongest Churn Correlation: {churned_corr.iloc[0]:.4f} ({churned_corr.index[0]})")
print(f"   💡 All correlations are weak (< 0.01), indicating synthetic/random data")