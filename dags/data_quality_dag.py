"""
Data Quality Validation DAG

This DAG runs BEFORE feature engineering to ensure data quality.

Validation checks:
1. Null value detection
2. Schema validation
3. Data drift monitoring
4. Data range checks

If validation fails, the pipeline stops and sends alerts.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
project_root = Path("/opt/airflow")
sys.path.insert(0, str(project_root))

def validate_raw_sessions():
    """Validate website_sessions.parquet"""
    from src.lead_ds.data_quality_validator import DataQualityValidator
    
    print("🔍 Validating website_sessions.parquet...")
    validator = DataQualityValidator("raw-data", "website_sessions.parquet")
    results = validator.run_all_checks()
    validator.save_report()
    
    if not results['passed']:
        raise ValueError(f"Validation failed for website_sessions with {len(results['errors'])} errors")
    
    return results

def validate_raw_orders():
    """Validate orders.parquet"""
    from src.lead_ds.data_quality_validator import DataQualityValidator
    
    print("🔍 Validating orders.parquet...")
    validator = DataQualityValidator("raw-data", "orders.parquet")
    results = validator.run_all_checks()
    validator.save_report()
    
    if not results['passed']:
        raise ValueError(f"Validation failed for orders with {len(results['errors'])} errors")
    
    return results

def validate_raw_pageviews():
    """Validate website_pageviews.parquet"""
    from src.lead_ds.data_quality_validator import DataQualityValidator
    
    print("🔍 Validating website_pageviews.parquet...")
    validator = DataQualityValidator("raw-data", "website_pageviews.parquet")
    results = validator.run_all_checks()
    validator.save_report()
    
    if not results['passed']:
        raise ValueError(f"Validation failed for pageviews with {len(results['errors'])} errors")
    
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

def check_data_freshness():
    """Check if data was recently updated"""
    from minio import Minio
    from datetime import datetime, timedelta
    
    client = Minio(
        "minio:9000",
        access_key="minio_admin",
        secret_key="minio_password",
        secure=False
    )
    
    print("🔍 Checking data freshness...")
    
    files_to_check = [
        ("raw-data", "website_sessions.parquet"),
        ("raw-data", "orders.parquet"),
        ("raw-data", "website_pageviews.parquet")
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
        with open(report_file, 'r') as f:
            latest_reports.append(json.load(f))
    
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
    description="Validate data quality before feature engineering",
    schedule_interval="@daily",  # Run daily
    catchup=False,
    tags=['data-quality', 'validation', 'monitoring'],
) as dag:
    
    # Task 1: Check data freshness
    freshness_check = PythonOperator(
        task_id="check_data_freshness",
        python_callable=check_data_freshness
    )
    
    # Task 2-4: Validate raw data files (in parallel)
    validate_sessions = PythonOperator(
        task_id="validate_raw_sessions",
        python_callable=validate_raw_sessions
    )
    
    validate_orders = PythonOperator(
        task_id="validate_raw_orders",
        python_callable=validate_raw_orders
    )
    
    validate_pageviews = PythonOperator(
        task_id="validate_raw_pageviews",
        python_callable=validate_raw_pageviews
    )
    
    # Task 5: Wait for feature engineering (if it runs)
    # This is optional - validates processed data after feature engineering
    wait_for_feature_engineering = ExternalTaskSensor(
        task_id="wait_for_feature_engineering",
        external_dag_id="feature_engineering_pipeline",
        external_task_id="process_features",
        mode="reschedule",
        timeout=3600,  # 1 hour timeout
        poke_interval=60,  # Check every minute
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        execution_delta=timedelta(hours=0),  # Same execution date
    )
    
    # Task 6: Validate processed data
    validate_processed = PythonOperator(
        task_id="validate_processed_data",
        python_callable=validate_processed_data
    )
    
    # Task 7: Generate summary report
    generate_report = PythonOperator(
        task_id="generate_summary_report",
        python_callable=generate_data_quality_report
    )
    
    # Task dependencies
    # First check freshness, then validate all raw files in parallel
    freshness_check >> [validate_sessions, validate_orders, validate_pageviews]
    
    # After raw validation, wait for feature engineering, then validate processed data
    [validate_sessions, validate_orders, validate_pageviews] >> wait_for_feature_engineering >> validate_processed
    
    # Finally generate summary report
    validate_processed >> generate_report


# Separate DAG for standalone validation (without waiting for feature engineering)
with DAG(
    "data_quality_validation_standalone",
    default_args=default_args,
    description="Validate raw data quality only (no feature engineering dependency)",
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['data-quality', 'validation', 'manual'],
) as standalone_dag:
    
    # Task 1: Check data freshness
    standalone_freshness = PythonOperator(
        task_id="check_data_freshness",
        python_callable=check_data_freshness
    )
    
    # Task 2-4: Validate raw data files
    standalone_sessions = PythonOperator(
        task_id="validate_raw_sessions",
        python_callable=validate_raw_sessions
    )
    
    standalone_orders = PythonOperator(
        task_id="validate_raw_orders",
        python_callable=validate_raw_orders
    )
    
    standalone_pageviews = PythonOperator(
        task_id="validate_raw_pageviews",
        python_callable=validate_raw_pageviews
    )
    
    # Task 5: Generate summary
    standalone_report = PythonOperator(
        task_id="generate_summary_report",
        python_callable=generate_data_quality_report
    )
    
    # Dependencies
    standalone_freshness >> [standalone_sessions, standalone_orders, standalone_pageviews] >> standalone_report
