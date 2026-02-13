"""
Data Quality Validation DAG (Retail Dataset)

This DAG runs to ensure data quality for the retail dataset.

Validation checks:
1. Null value detection
2. Schema validation
3. Data freshness monitoring
4. Data range checks

If validation fails, the pipeline stops and sends alerts.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
project_root = Path("/opt/airflow")
sys.path.insert(0, str(project_root))

def validate_raw_retail_data():
    """Validate retail_data.parquet"""
    from src.lead_ds.data_quality_validator import DataQualityValidator
    
    print("🔍 Validating retail_data.parquet...")
    validator = DataQualityValidator("raw-data", "retail_data.parquet")
    results = validator.run_all_checks()
    validator.save_report()
    
    if not results['passed']:
        raise ValueError(f"Validation failed for retail_data with {len(results['errors'])} errors")
    
    return results

def validate_processed_data():
    """Validate training_data.parquet (runs after feature engineering)"""
    from src.lead_ds.data_quality_validator import DataQualityValidator
    
    print("🔍 Validating training_data.parquet...")
    validator = DataQualityValidator("processed-data", "training_data.parquet")
    results = validator.run_all_checks()
    validator.save_report()
    
    if not results['passed']:
        raise ValueError(f"Validation failed for training_data with {len(results['errors'])} errors")
    
    return results

def check_retail_schema():
    """Check that retail dataset has expected schema"""
    from minio import Minio
    import pandas as pd
    from io import BytesIO
    import os
    
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key="minio_admin",
        secret_key="minio_password",
        secure=False
    )
    
    print("🔍 Checking retail dataset schema...")
    
    # Load raw data
    response = client.get_object("raw-data", "retail_data.parquet")
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    response.release_conn()
    
    # Expected columns (key ones)
    expected_columns = [
        'age', 'gender', 'income_bracket', 'total_sales', 
        'churned', 'product_category', 'payment_method',
        'loyalty_program', 'membership_years'
    ]
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"❌ Missing expected columns: {missing_cols}")
    
    print(f"✅ Schema validation passed")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Total rows: {len(df):,}")
    
    # Check for required targets
    if 'total_sales' not in df.columns:
        raise ValueError("❌ Missing regression target: total_sales")
    if 'churned' not in df.columns:
        raise ValueError("❌ Missing classification target: churned")
    
    print(f"   ✅ Regression target (total_sales): ${df['total_sales'].mean():.2f} avg")
    print(f"   ✅ Classification target (churned): {df['churned'].value_counts().to_dict()}")
    
def check_data_freshness():
    """Check if data was recently updated"""
    from minio import Minio
    from datetime import datetime, timedelta
    import os
    
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key="minio_admin",
        secret_key="minio_password",
        secure=False
    )
    
    print("🔍 Checking data freshness...")
    
    files_to_check = [
        ("raw-data", "retail_data.parquet"),
    ]
    
    max_age_days = 30  # Alert if data is older than 30 days
    
    for bucket, filename in files_to_check:
        try:
            stat = client.stat_object(bucket, filename)
            age = datetime.now(stat.last_modified.tzinfo) - stat.last_modified
            age_days = age.days
            
            print(f"   {filename}: {age_days} days old")
            
            if age_days > max_age_days:
                print(f"   ⚠️  WARNING: {filename} is {age_days} days old (threshold: {max_age_days} days)")
            else:
                print(f"   ✅ Fresh data")
                
        except Exception as e:
            print(f"   ❌ Error checking {filename}: {e}")
            raise

def check_processed_data_quality():
    """Check quality metrics on processed training data"""
    from minio import Minio
    import pandas as pd
    from io import BytesIO
    import os
    
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key="minio_admin",
        secret_key="minio_password",
        secure=False
    )
    
    print("🔍 Checking processed data quality...")
    
    response = client.get_object("processed-data", "training_data.parquet")
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    response.release_conn()
    
    print(f"   Rows: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"   ⚠️  WARNING: Found {missing.sum()} missing values")
        print(missing[missing > 0])
    else:
        print(f"   ✅ No missing values")
    
    # Check target distribution
    if 'total_sales' in df.columns:
        print(f"   total_sales: mean=${df['total_sales'].mean():.2f}, std=${df['total_sales'].std():.2f}")
    
    if 'churned' in df.columns:
        churn_dist = df['churned'].value_counts(normalize=True)
        print(f"   churned distribution: {churn_dist.to_dict()}")
        
        # Check for class imbalance (should be close to 50/50 for retail)
        if churn_dist.min() < 0.3:
            print(f"   ⚠️  WARNING: Class imbalance detected in 'churned' target")

def generate_data_quality_report():
    """Generate summary report of all validations"""
    import json
    from pathlib import Path
    from datetime import datetime
    
    print("📊 Generating data quality summary report...")
    
    reports_dir = Path("/opt/airflow/reports/data_quality")
    
    if not reports_dir.exists():
        print("   ℹ️  No validation reports found")
        return
    
    # Get latest reports
    report_files = sorted(reports_dir.glob("validation_report_*.json"), reverse=True)
    
    if not report_files:
        print("   ℹ️  No validation reports found")
        return
    
    # Read latest reports (up to 10)
    latest_reports = []
    for report_file in report_files[:10]:
        try:
            with open(report_file, 'r') as f:
                latest_reports.append(json.load(f))
        except json.JSONDecodeError as e:
            print(f"   ⚠️  Skipping corrupted report file {report_file}: {e}")
        except Exception as e:
            print(f"   ⚠️  Error reading report file {report_file}: {e}")
    
    # Generate summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_reports': len(latest_reports),
        'passed': sum(1 for r in latest_reports if r['passed']),
        'failed': sum(1 for r in latest_reports if not r['passed']),
        'latest_validation': latest_reports[0] if latest_reports else None
    }
    
    # Save summary
    summary_file = reports_dir / 'quality_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary report saved to {summary_file}")
    print(f"   Total validations: {summary['total_reports']}")
    print(f"   Passed: {summary['passed']}")
    print(f"   Failed: {summary['failed']}")

# --- DAG Definition ---
default_args = {
    "owner": "lead_engineer",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "data_quality_validation",
    default_args=default_args,
    description="Validate retail dataset quality before and after feature engineering",
    schedule_interval="@daily",  # Run daily
    catchup=False,
    tags=['data-quality', 'validation', 'retail'],
) as dag:
    
    # Task 1: Check data freshness
    freshness_check = PythonOperator(
        task_id="check_data_freshness",
        python_callable=check_data_freshness
    )
    
    # Task 2: Validate raw retail data
    validate_raw = PythonOperator(
        task_id="validate_raw_retail_data",
        python_callable=validate_raw_retail_data
    )
    
    # Task 3: Check retail schema
    schema_check = PythonOperator(
        task_id="check_retail_schema",
        python_callable=check_retail_schema
    )
    
    # Task 4: Validate processed data
    validate_processed = PythonOperator(
        task_id="validate_processed_data",
        python_callable=validate_processed_data
    )
    
    # Task 5: Check processed data quality metrics
    quality_check = PythonOperator(
        task_id="check_processed_data_quality",
        python_callable=check_processed_data_quality
    )
    
    # Task 6: Generate summary report
    generate_report = PythonOperator(
        task_id="generate_summary_report",
        python_callable=generate_data_quality_report
    )
    
    # Task dependencies
    # Check freshness and schema in parallel, then validate raw data
    freshness_check >> validate_raw
    schema_check >> validate_raw
    
    # After raw validation, validate processed data and check quality
    validate_raw >> [validate_processed, quality_check]
    
    # Finally generate summary report
    [validate_processed, quality_check] >> generate_report


# Separate DAG for standalone raw data validation (before feature engineering)
with DAG(
    "data_quality_validation_standalone",
    default_args=default_args,
    description="Validate raw retail data quality only (manual trigger)",
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['data-quality', 'validation', 'manual', 'retail'],
) as standalone_dag:
    
    # Task 1: Check data freshness
    standalone_freshness = PythonOperator(
        task_id="check_data_freshness",
        python_callable=check_data_freshness
    )
    
    # Task 2: Check schema
    standalone_schema = PythonOperator(
        task_id="check_retail_schema",
        python_callable=check_retail_schema
    )
    
    # Task 3: Validate raw data
    standalone_raw = PythonOperator(
        task_id="validate_raw_retail_data",
        python_callable=validate_raw_retail_data
    )
    
    # Task 4: Generate summary
    standalone_report = PythonOperator(
        task_id="generate_summary_report",
        python_callable=generate_data_quality_report
    )
    
    # Dependencies
    [standalone_freshness, standalone_schema] >> standalone_raw >> standalone_report
