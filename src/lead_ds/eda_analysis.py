import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from minio import Minio
from io import BytesIO
import os
from pathlib import Path

# --- CONFIG ---
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minio_admin"
SECRET_KEY = "minio_password"
BUCKET_NAME = "processed-data"
OUTPUT_DIR = "reports/figures"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def setup_output_dir():
    """Create output directory for plots"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")

def load_data():
    """Load training data from MinIO"""
    print("🔄 Loading data from MinIO...")
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )
    
    response = client.get_object(BUCKET_NAME, "training_data.parquet")
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    response.release_conn()
    
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def plot_1_conversion_overview(df):
    """Overall conversion rate visualization"""
    print("\n📊 1. Conversion Overview...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    conversion_counts = df['is_ordered'].value_counts()
    colors = ['#e74c3c', '#2ecc71']
    labels = ['No Purchase (93.2%)', 'Purchased (6.8%)']
    
    ax1.pie(conversion_counts, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    ax1.set_title('Conversion Rate Distribution\n(High Class Imbalance)', fontsize=14, weight='bold')
    
    # Bar chart with counts
    ax2.bar(['No Purchase', 'Purchased'], conversion_counts.values, color=colors, alpha=0.8)
    ax2.set_ylabel('Number of Sessions', fontsize=12)
    ax2.set_title('Session Counts by Conversion', fontsize=14, weight='bold')
    ax2.text(0, conversion_counts.iloc[0]/2, f'{conversion_counts.iloc[0]:,}', 
             ha='center', va='center', fontsize=12, weight='bold')
    ax2.text(1, conversion_counts.iloc[1]/2, f'{conversion_counts.iloc[1]:,}', 
             ha='center', va='center', fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_conversion_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 01_conversion_overview.png")

def plot_2_revenue_distribution(df):
    """Revenue distribution analysis"""
    print("\n📊 2. Revenue Distribution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total revenue stats
    total_revenue = df['revenue'].sum()
    avg_revenue = df['revenue'].mean()
    converting_sessions = df[df['is_ordered'] == 1]
    avg_order_value = converting_sessions['revenue'].mean()
    
    # 1. Revenue distribution (all sessions)
    axes[0, 0].hist(df['revenue'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Revenue (USD)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Revenue Distribution (All Sessions)', fontsize=12, weight='bold')
    axes[0, 0].axvline(avg_revenue, color='red', linestyle='--', linewidth=2, label=f'Mean: ${avg_revenue:.2f}')
    axes[0, 0].legend()
    
    # 2. Revenue distribution (only converting sessions)
    axes[0, 1].hist(converting_sessions['revenue'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Revenue (USD)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Revenue Distribution (Converted Sessions Only)', fontsize=12, weight='bold')
    axes[0, 1].axvline(avg_order_value, color='red', linestyle='--', linewidth=2, 
                       label=f'Avg Order: ${avg_order_value:.2f}')
    axes[0, 1].legend()
    
    # 3. Revenue summary stats
    axes[1, 0].axis('off')
    summary_text = f"""
    📈 REVENUE SUMMARY
    
    Total Revenue: ${total_revenue:,.2f}
    Total Sessions: {len(df):,}
    Converted Sessions: {len(converting_sessions):,}
    
    Average Revenue per Session: ${avg_revenue:.2f}
    Average Order Value (AOV): ${avg_order_value:.2f}
    
    Min Order: ${converting_sessions['revenue'].min():.2f}
    Max Order: ${converting_sessions['revenue'].max():.2f}
    Median Order: ${converting_sessions['revenue'].median():.2f}
    """
    axes[1, 0].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Top revenue percentiles
    percentiles = [50, 75, 90, 95, 99]
    values = [converting_sessions['revenue'].quantile(p/100) for p in percentiles]
    axes[1, 1].barh([f'{p}th' for p in percentiles], values, color='#9b59b6', alpha=0.8)
    axes[1, 1].set_xlabel('Revenue (USD)', fontsize=11)
    axes[1, 1].set_title('Revenue Percentiles (Converted Sessions)', fontsize=12, weight='bold')
    for i, v in enumerate(values):
        axes[1, 1].text(v, i, f' ${v:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_revenue_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 02_revenue_distribution.png")

def plot_3_time_analysis(df):
    """Time-based analysis"""
    print("\n📊 3. Time-Based Patterns...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Conversion by hour
    hourly_conv = df.groupby('hour_of_day').agg({
        'is_ordered': ['sum', 'mean', 'count']
    }).reset_index()
    hourly_conv.columns = ['hour', 'conversions', 'conv_rate', 'total_sessions']
    
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.bar(hourly_conv['hour'], hourly_conv['total_sessions'], alpha=0.3, color='gray', label='Sessions')
    ax1_twin.plot(hourly_conv['hour'], hourly_conv['conv_rate'] * 100, color='red', 
                  marker='o', linewidth=2, label='Conversion Rate %')
    ax1.set_xlabel('Hour of Day', fontsize=11)
    ax1.set_ylabel('Total Sessions', fontsize=11, color='gray')
    ax1_twin.set_ylabel('Conversion Rate (%)', fontsize=11, color='red')
    ax1.set_title('Hourly Traffic and Conversion Patterns', fontsize=12, weight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # 2. Weekend vs Weekday
    weekend_stats = df.groupby('is_weekend').agg({
        'is_ordered': ['mean', 'count'],
        'revenue': 'sum'
    }).reset_index()
    weekend_stats.columns = ['is_weekend', 'conv_rate', 'sessions', 'revenue']
    weekend_stats['label'] = weekend_stats['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    
    axes[0, 1].bar(weekend_stats['label'], weekend_stats['conv_rate'] * 100, 
                   color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[0, 1].set_ylabel('Conversion Rate (%)', fontsize=11)
    axes[0, 1].set_title('Weekend vs Weekday Conversion', fontsize=12, weight='bold')
    for i, row in weekend_stats.iterrows():
        axes[0, 1].text(i, row['conv_rate'] * 100, f"{row['conv_rate']*100:.2f}%", 
                       ha='center', va='bottom', fontsize=11, weight='bold')
    
    # 3. Revenue by hour
    hourly_revenue = df.groupby('hour_of_day')['revenue'].sum().reset_index()
    axes[1, 0].plot(hourly_revenue['hour_of_day'], hourly_revenue['revenue'], 
                    marker='o', linewidth=2, color='#2ecc71')
    axes[1, 0].fill_between(hourly_revenue['hour_of_day'], hourly_revenue['revenue'], 
                            alpha=0.3, color='#2ecc71')
    axes[1, 0].set_xlabel('Hour of Day', fontsize=11)
    axes[1, 0].set_ylabel('Total Revenue (USD)', fontsize=11)
    axes[1, 0].set_title('Revenue Generation by Hour', fontsize=12, weight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sessions by hour (area chart)
    hourly_sessions = df.groupby('hour_of_day').size().reset_index(name='sessions')
    axes[1, 1].fill_between(hourly_sessions['hour_of_day'], hourly_sessions['sessions'], 
                            alpha=0.5, color='#9b59b6')
    axes[1, 1].plot(hourly_sessions['hour_of_day'], hourly_sessions['sessions'], 
                    marker='o', linewidth=2, color='#8e44ad')
    axes[1, 1].set_xlabel('Hour of Day', fontsize=11)
    axes[1, 1].set_ylabel('Number of Sessions', fontsize=11)
    axes[1, 1].set_title('Traffic Volume by Hour', fontsize=12, weight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 03_time_analysis.png")

def plot_4_traffic_source_analysis(df):
    """Traffic source and device analysis"""
    print("\n📊 4. Traffic Source & Device Analysis...")
    
    # Decode one-hot encoding
    df_decoded = df.copy()
    
    # Traffic source
    if 'utm_source_gsearch' in df.columns:
        df_decoded['traffic_source'] = 'direct'
        df_decoded.loc[df['utm_source_gsearch'] == 1, 'traffic_source'] = 'gsearch'
        df_decoded.loc[df['utm_source_socialbook'] == 1, 'traffic_source'] = 'socialbook'
    
    # Device type
    if 'device_type_mobile' in df.columns:
        df_decoded['device'] = df['device_type_mobile'].map({1: 'mobile', 0: 'desktop'})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Traffic source distribution
    if 'traffic_source' in df_decoded.columns:
        source_counts = df_decoded['traffic_source'].value_counts()
        axes[0, 0].pie(source_counts, labels=source_counts.index, autopct='%1.1f%%',
                      colors=sns.color_palette('Set2', len(source_counts)), startangle=90)
        axes[0, 0].set_title('Traffic Source Distribution', fontsize=12, weight='bold')
    
    # 2. Conversion by traffic source
    if 'traffic_source' in df_decoded.columns:
        source_conv = df_decoded.groupby('traffic_source').agg({
            'is_ordered': ['mean', 'sum'],
            'revenue': 'sum'
        }).reset_index()
        source_conv.columns = ['source', 'conv_rate', 'conversions', 'revenue']
        
        axes[0, 1].bar(source_conv['source'], source_conv['conv_rate'] * 100, 
                      color=sns.color_palette('Set2', len(source_conv)), alpha=0.8)
        axes[0, 1].set_ylabel('Conversion Rate (%)', fontsize=11)
        axes[0, 1].set_title('Conversion Rate by Traffic Source', fontsize=12, weight='bold')
        for i, row in source_conv.iterrows():
            axes[0, 1].text(i, row['conv_rate'] * 100, f"{row['conv_rate']*100:.2f}%", 
                           ha='center', va='bottom', fontsize=10, weight='bold')
    
    # 3. Device distribution
    if 'device' in df_decoded.columns:
        device_counts = df_decoded['device'].value_counts()
        colors = ['#3498db', '#e74c3c']
        axes[1, 0].barh(device_counts.index, device_counts.values, color=colors, alpha=0.8)
        axes[1, 0].set_xlabel('Number of Sessions', fontsize=11)
        axes[1, 0].set_title('Device Type Distribution', fontsize=12, weight='bold')
        for i, v in enumerate(device_counts.values):
            axes[1, 0].text(v, i, f' {v:,} ({v/len(df)*100:.1f}%)', 
                           va='center', fontsize=10, weight='bold')
    
    # 4. Conversion by device
    if 'device' in df_decoded.columns:
        device_conv = df_decoded.groupby('device').agg({
            'is_ordered': 'mean',
            'revenue': 'mean'
        }).reset_index()
        device_conv.columns = ['device', 'conv_rate', 'avg_revenue']
        
        x = range(len(device_conv))
        width = 0.35
        axes[1, 1].bar([i - width/2 for i in x], device_conv['conv_rate'] * 100, 
                      width, label='Conv Rate %', color='#2ecc71', alpha=0.8)
        ax_twin = axes[1, 1].twinx()
        ax_twin.bar([i + width/2 for i in x], device_conv['avg_revenue'], 
                   width, label='Avg Revenue', color='#f39c12', alpha=0.8)
        
        axes[1, 1].set_xlabel('Device Type', fontsize=11)
        axes[1, 1].set_ylabel('Conversion Rate (%)', fontsize=11, color='#2ecc71')
        ax_twin.set_ylabel('Avg Revenue per Session ($)', fontsize=11, color='#f39c12')
        axes[1, 1].set_title('Device Performance Metrics', fontsize=12, weight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(device_conv['device'])
        axes[1, 1].legend(loc='upper left')
        ax_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_traffic_device_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 04_traffic_device_analysis.png")

def plot_5_engagement_analysis(df):
    """Engagement depth and repeat session analysis"""
    print("\n📊 5. Engagement & User Behavior...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Engagement depth distribution
    engagement_counts = df['engagement_depth'].value_counts().sort_index()
    axes[0, 0].bar(engagement_counts.index[:20], engagement_counts.values[:20], 
                  color='#3498db', alpha=0.8)
    axes[0, 0].set_xlabel('Number of Pages Viewed', fontsize=11)
    axes[0, 0].set_ylabel('Number of Sessions', fontsize=11)
    axes[0, 0].set_title('Engagement Depth Distribution (Top 20)', fontsize=12, weight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Conversion by engagement depth
    engagement_conv = df.groupby('engagement_depth').agg({
        'is_ordered': ['mean', 'count']
    }).reset_index()
    engagement_conv.columns = ['depth', 'conv_rate', 'count']
    engagement_conv = engagement_conv[engagement_conv['count'] >= 100]  # Filter for significance
    
    axes[0, 1].scatter(engagement_conv['depth'], engagement_conv['conv_rate'] * 100, 
                      s=engagement_conv['count']/10, alpha=0.6, color='#e74c3c')
    axes[0, 1].set_xlabel('Engagement Depth (Pages)', fontsize=11)
    axes[0, 1].set_ylabel('Conversion Rate (%)', fontsize=11)
    axes[0, 1].set_title('Conversion Rate vs Engagement\n(Bubble size = session count)', 
                        fontsize=12, weight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Repeat vs New users
    repeat_stats = df.groupby('is_repeat_session').agg({
        'is_ordered': ['mean', 'count'],
        'revenue': 'sum'
    }).reset_index()
    repeat_stats.columns = ['is_repeat', 'conv_rate', 'sessions', 'revenue']
    repeat_stats['label'] = repeat_stats['is_repeat'].map({0: 'New User', 1: 'Repeat User'})
    
    x = range(len(repeat_stats))
    width = 0.35
    axes[1, 0].bar([i - width/2 for i in x], repeat_stats['conv_rate'] * 100, 
                  width, label='Conv Rate %', color='#2ecc71', alpha=0.8)
    ax_twin = axes[1, 0].twinx()
    ax_twin.bar([i + width/2 for i in x], repeat_stats['sessions'], 
               width, label='Sessions', color='#9b59b6', alpha=0.8)
    
    axes[1, 0].set_ylabel('Conversion Rate (%)', fontsize=11, color='#2ecc71')
    ax_twin.set_ylabel('Number of Sessions', fontsize=11, color='#9b59b6')
    axes[1, 0].set_title('New vs Repeat User Performance', fontsize=12, weight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(repeat_stats['label'])
    axes[1, 0].legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    
    # 4. Engagement funnel
    funnel_data = {
        'Stage': ['All Sessions', '2+ Pages', '3+ Pages', '5+ Pages', 'Converted'],
        'Count': [
            len(df),
            len(df[df['engagement_depth'] >= 2]),
            len(df[df['engagement_depth'] >= 3]),
            len(df[df['engagement_depth'] >= 5]),
            len(df[df['is_ordered'] == 1])
        ]
    }
    funnel_df = pd.DataFrame(funnel_data)
    funnel_df['Percentage'] = (funnel_df['Count'] / len(df) * 100).round(1)
    
    colors_funnel = plt.cm.Blues(np.linspace(0.4, 0.9, len(funnel_df)))
    axes[1, 1].barh(funnel_df['Stage'], funnel_df['Count'], color=colors_funnel, alpha=0.8)
    axes[1, 1].set_xlabel('Number of Sessions', fontsize=11)
    axes[1, 1].set_title('Engagement Funnel', fontsize=12, weight='bold')
    for i, row in funnel_df.iterrows():
        axes[1, 1].text(row['Count'], i, f" {row['Count']:,} ({row['Percentage']}%)", 
                       va='center', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_engagement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 05_engagement_analysis.png")

def plot_6_landing_page_analysis(df):
    """Landing page performance analysis"""
    print("\n📊 6. Landing Page Analysis...")
    
    # Decode landing pages
    landing_page_cols = [col for col in df.columns if col.startswith('landing_page_')]
    
    if landing_page_cols:
        df_decoded = df.copy()
        df_decoded['landing_page'] = '/home'  # default
        for col in landing_page_cols:
            page_name = col.replace('landing_page_', '')
            df_decoded.loc[df[col] == 1, 'landing_page'] = page_name
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Landing page distribution
        lp_counts = df_decoded['landing_page'].value_counts()
        axes[0, 0].barh(lp_counts.index, lp_counts.values, color=sns.color_palette('viridis', len(lp_counts)), alpha=0.8)
        axes[0, 0].set_xlabel('Number of Sessions', fontsize=11)
        axes[0, 0].set_title('Landing Page Distribution', fontsize=12, weight='bold')
        for i, v in enumerate(lp_counts.values):
            axes[0, 0].text(v, i, f' {v:,}', va='center', fontsize=10)
        
        # 2. Conversion by landing page
        lp_conv = df_decoded.groupby('landing_page').agg({
            'is_ordered': ['mean', 'sum', 'count']
        }).reset_index()
        lp_conv.columns = ['landing_page', 'conv_rate', 'conversions', 'sessions']
        lp_conv = lp_conv.sort_values('conv_rate', ascending=True)
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(lp_conv)))
        axes[0, 1].barh(lp_conv['landing_page'], lp_conv['conv_rate'] * 100, color=colors, alpha=0.8)
        axes[0, 1].set_xlabel('Conversion Rate (%)', fontsize=11)
        axes[0, 1].set_title('Conversion Rate by Landing Page', fontsize=12, weight='bold')
        for i, row in lp_conv.iterrows():
            axes[0, 1].text(row['conv_rate'] * 100, i, f" {row['conv_rate']*100:.2f}%", 
                           va='center', fontsize=10, weight='bold')
        
        # 3. Revenue by landing page
        lp_revenue = df_decoded.groupby('landing_page')['revenue'].sum().sort_values(ascending=True)
        axes[1, 0].barh(lp_revenue.index, lp_revenue.values, color='#f39c12', alpha=0.8)
        axes[1, 0].set_xlabel('Total Revenue (USD)', fontsize=11)
        axes[1, 0].set_title('Revenue by Landing Page', fontsize=12, weight='bold')
        for i, v in enumerate(lp_revenue.values):
            axes[1, 0].text(v, i, f' ${v:,.0f}', va='center', fontsize=10)
        
        # 4. Landing page performance matrix
        lp_matrix = df_decoded.groupby('landing_page').agg({
            'is_ordered': 'mean',
            'revenue': 'mean',
            'engagement_depth': 'mean'
        }).reset_index()
        lp_matrix.columns = ['landing_page', 'conv_rate', 'avg_revenue', 'avg_engagement']
        
        axes[1, 1].scatter(lp_matrix['avg_engagement'], lp_matrix['conv_rate'] * 100, 
                          s=lp_matrix['avg_revenue']*20, alpha=0.6, 
                          c=range(len(lp_matrix)), cmap='viridis')
        axes[1, 1].set_xlabel('Avg Engagement Depth', fontsize=11)
        axes[1, 1].set_ylabel('Conversion Rate (%)', fontsize=11)
        axes[1, 1].set_title('Landing Page Performance\n(Bubble size = avg revenue)', 
                            fontsize=12, weight='bold')
        for i, row in lp_matrix.iterrows():
            axes[1, 1].annotate(row['landing_page'], 
                               (row['avg_engagement'], row['conv_rate'] * 100),
                               fontsize=8, alpha=0.7)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/06_landing_page_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 06_landing_page_analysis.png")
    else:
        print("   ⚠ No landing page columns found, skipping...")

def plot_7_correlation_matrix(df):
    """Feature correlation analysis"""
    print("\n📊 7. Feature Correlation Analysis...")
    
    # Select numerical features only
    numerical_cols = ['is_repeat_session', 'hour_of_day', 'is_weekend', 
                     'engagement_depth', 'is_ordered', 'revenue']
    
    # Add one-hot encoded features
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ['utm_source_', 'device_type_', 'landing_page_']):
            numerical_cols.append(col)
    
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    corr_matrix = df[numerical_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 07_correlation_matrix.png")

def generate_summary_report(df):
    """Generate text summary report"""
    print("\n📝 Generating Summary Report...")
    
    # Calculate key statistics
    total_sessions = len(df)
    total_conversions = df['is_ordered'].sum()
    conversion_rate = df['is_ordered'].mean() * 100
    total_revenue = df['revenue'].sum()
    avg_revenue_per_session = df['revenue'].mean()
    avg_order_value = df[df['is_ordered'] == 1]['revenue'].mean()
    
    # Decode categorical features
    df_decoded = df.copy()
    
    # Traffic source
    if 'utm_source_gsearch' in df.columns:
        df_decoded['traffic_source'] = 'direct'
        df_decoded.loc[df['utm_source_gsearch'] == 1, 'traffic_source'] = 'gsearch'
        df_decoded.loc[df['utm_source_socialbook'] == 1, 'traffic_source'] = 'socialbook'
        best_source = df_decoded.groupby('traffic_source')['is_ordered'].mean().idxmax()
        best_source_rate = df_decoded.groupby('traffic_source')['is_ordered'].mean().max() * 100
    else:
        best_source = "N/A"
        best_source_rate = 0
    
    # Device
    if 'device_type_mobile' in df.columns:
        df_decoded['device'] = df['device_type_mobile'].map({1: 'mobile', 0: 'desktop'})
        device_conv = df_decoded.groupby('device')['is_ordered'].mean()
        if len(device_conv) > 0:
            best_device = device_conv.idxmax()
            best_device_rate = device_conv.max() * 100
        else:
            best_device = "N/A"
            best_device_rate = 0
    else:
        best_device = "N/A"
        best_device_rate = 0
    
    # Time patterns
    best_hour = df.groupby('hour_of_day')['is_ordered'].mean().idxmax()
    best_hour_rate = df.groupby('hour_of_day')['is_ordered'].mean().max() * 100
    
    weekend_rate = df[df['is_weekend'] == 1]['is_ordered'].mean() * 100
    weekday_rate = df[df['is_weekend'] == 0]['is_ordered'].mean() * 100
    
    # Engagement patterns
    avg_engagement = df['engagement_depth'].mean()
    high_engagement_conv = df[df['engagement_depth'] >= 5]['is_ordered'].mean() * 100
    low_engagement_conv = df[df['engagement_depth'] <= 1]['is_ordered'].mean() * 100
    
    # Repeat users
    repeat_rate = df[df['is_repeat_session'] == 1]['is_ordered'].mean() * 100
    new_rate = df[df['is_repeat_session'] == 0]['is_ordered'].mean() * 100
    
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║         EXPLORATORY DATA ANALYSIS - SUMMARY REPORT            ║
║              Maven Fuzzy Factory - Toy Store                  ║
╚══════════════════════════════════════════════════════════════╝

📊 DATASET OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Sessions:              {total_sessions:,}
Total Conversions:           {total_conversions:,}
Overall Conversion Rate:     {conversion_rate:.2f}%

💰 REVENUE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Revenue:               ${total_revenue:,.2f}
Avg Revenue per Session:     ${avg_revenue_per_session:.2f}
Average Order Value (AOV):   ${avg_order_value:.2f}

🎯 KEY INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CLASS IMBALANCE CHALLENGE
   • 93.2% of sessions do NOT convert
   • Only 6.8% result in a purchase
   • Recommendation: Use stratified sampling, SMOTE, or class weights

2. BEST PERFORMING TRAFFIC SOURCE
   • Source: {best_source}
   • Conversion Rate: {best_source_rate:.2f}%

3. DEVICE PERFORMANCE
   • Best Device: {best_device}
   • Conversion Rate: {best_device_rate:.2f}%

4. OPTIMAL TIME WINDOWS
   • Best Hour: {best_hour}:00
   • Conversion Rate: {best_hour_rate:.2f}%
   • Weekend Conversion: {weekend_rate:.2f}%
   • Weekday Conversion: {weekday_rate:.2f}%

5. ENGAGEMENT IMPACT
   • Average Engagement: {avg_engagement:.1f} pages
   • High Engagement (5+ pages): {high_engagement_conv:.2f}% conversion
   • Low Engagement (1 page): {low_engagement_conv:.2f}% conversion
   • Insight: Engagement STRONGLY correlates with conversion

6. USER LOYALTY
   • New User Conversion: {new_rate:.2f}%
   • Repeat User Conversion: {repeat_rate:.2f}%

📈 RECOMMENDATIONS FOR MODELING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classification (Conversion Prediction):
  ✓ Handle severe class imbalance (93/7 split)
  ✓ Focus on F1-score or AUC-ROC, not just accuracy
  ✓ Consider ensemble methods (XGBoost, Random Forest)
  ✓ Feature importance: engagement_depth likely critical

Regression (Revenue Prediction):
  ✓ Consider two-stage model (convert first, then predict amount)
  ✓ Or model only converting sessions
  ✓ Long-tail distribution suggests log transformation

Feature Engineering V2:
  ✓ Session duration (time span of pageviews)
  ✓ Bounce rate indicator (engagement_depth == 1)
  ✓ Traffic source × device interaction terms
  ✓ UTM campaign granularity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Location: {OUTPUT_DIR}/eda_summary_report.txt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    # Save report
    with open(f'{OUTPUT_DIR}/eda_summary_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✓ Saved: eda_summary_report.txt")

def main():
    """Main EDA execution"""
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS - Maven Fuzzy Factory")
    print("  Role: Lead Data Scientist (The Storyteller)")
    print("=" * 60)
    
    # Setup
    setup_output_dir()
    
    # Load data
    df = load_data()
    
    # Generate all visualizations
    plot_1_conversion_overview(df)
    plot_2_revenue_distribution(df)
    plot_3_time_analysis(df)
    plot_4_traffic_source_analysis(df)
    plot_5_engagement_analysis(df)
    plot_6_landing_page_analysis(df)
    plot_7_correlation_matrix(df)
    
    # Generate summary report
    generate_summary_report(df)
    
    print("\n" + "=" * 60)
    print("✅ EDA COMPLETE!")
    print(f"📁 All visualizations saved to: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
