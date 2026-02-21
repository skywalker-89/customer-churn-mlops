"""
Comprehensive Model Benchmark DAG

This DAG runs a complete benchmark of ALL classification and regression models:
1. Validates training data exists
2. Runs classification benchmark (all scratch models vs sklearn)
3. Runs regression benchmark (all scratch models vs sklearn)
4. Generates comparison summary
5. Creates benchmark report

Models Tested:
- Classification: Logistic Regression, Decision Tree, Random Forest, SVM, 
                  Random Forest + PCA, SVM + PCA, K-Means, Agglomerative Clustering,
                  Perceptron, MLP, Custom Model
- Regression: Linear Regression, Multiple Regression, Polynomial Regression, XGBoost
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
project_root = Path("/opt/airflow")
sys.path.insert(0, str(project_root))

def validate_training_data():
    """Validate that training data exists and has required columns"""
    from minio import Minio
    from io import BytesIO
    import pandas as pd
    import os
    
    print("🔍 Validating training data for comprehensive benchmark...")
    
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key="minio_admin",
        secret_key="minio_password",
        secure=False
    )
    
    try:
        # Check if training data exists
        response = client.get_object("processed-data", "training_data.parquet")
        df = pd.read_parquet(BytesIO(response.read()))
        response.close()
        response.release_conn()
        
        print(f"✅ Training data loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # Validate required columns for both tasks
        required_cols = ['total_sales', 'churned']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check data quality metrics
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"⚠️  Found {null_counts.sum()} null values")
            print(null_counts[null_counts > 0])
        
        # Check target distributions
        print(f"📊 Target distributions:")
        print(f"   - total_sales: mean=${df['total_sales'].mean():.2f}, std=${df['total_sales'].std():.2f}")
        print(f"   - churned: {df['churned'].value_counts().to_dict()}")
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'null_count': null_counts.sum(),
            'targets_valid': True
        }
        
    except Exception as e:
        print(f"❌ Training data validation failed: {e}")
        raise

def run_classification_benchmark():
    """Run comprehensive classification benchmark with all scratch models"""
    import os
    import time
    
    print("🎯 Starting Comprehensive Classification Benchmark...")
    print("=" * 60)
    
    # Set environment variables
    os.environ["MINIO_ENDPOINT"] = "minio:9000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
    
    start_time = time.time()
    
    try:
        from src.classification.retail_classification_benchmark import main
        
        print("📊 Running classification models:")
        print("   - Logistic Regression (Scratch)")
        print("   - Decision Tree (Scratch)")
        print("   - Random Forest (Scratch)")
        print("   - SVM (Scratch)")
        print("   - Random Forest + PCA (Scratch)")
        print("   - SVM + PCA (Scratch)")
        print("   - K-Means Clustering (Scratch)")
        print("   - Agglomerative Clustering (Scratch)")
        print("   - Perceptron (Scratch)")
        print("   - Multi-Layer Perceptron (Scratch)")
        print("   - Custom Model (Scratch)")
        print("   - Sklearn Baseline (Comparison)")
        
        # Run the benchmark
        results = main()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Classification Benchmark Complete!")
        print(f"   Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        
        return {
            'status': 'success',
            'duration': duration,
            'models_tested': 12  # 11 scratch + 1 sklearn
        }
        
    except Exception as e:
        print(f"❌ Classification benchmark failed: {e}")
        raise

def run_regression_benchmark():
    """Run comprehensive regression benchmark with all scratch models"""
    import os
    import time
    
    print("📈 Starting Comprehensive Regression Benchmark...")
    print("=" * 60)
    
    # Set environment variables
    os.environ["MINIO_ENDPOINT"] = "minio:9000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
    
    start_time = time.time()
    
    try:
        from src.regression.retail_regression_benchmark import main
        
        print("📊 Running regression models:")
        print("   - Linear Regression (Scratch)")
        print("   - Multiple Linear Regression (Scratch)")
        print("   - Polynomial Regression (Scratch)")
        print("   - XGBoost (Scratch)")
        print("   - Sklearn XGBoost (Comparison)")
        
        # Run the benchmark
        results = main()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Regression Benchmark Complete!")
        print(f"   Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        
        return {
            'status': 'success',
            'duration': duration,
            'models_tested': 5  # 4 scratch + 1 sklearn
        }
        
    except Exception as e:
        print(f"❌ Regression benchmark failed: {e}")
        raise

def generate_benchmark_summary(**context):
    """Generate comprehensive summary of all benchmark results"""
    import json
    from datetime import datetime
    from pathlib import Path
    
    print("📋 Generating Comprehensive Benchmark Summary...")
    
    # Get task results from context
    validation_results = context['task_instance'].xcom_pull(task_ids='validate_training_data')
    classification_results = context['task_instance'].xcom_pull(task_ids='run_classification_benchmark')
    regression_results = context['task_instance'].xcom_pull(task_ids='run_regression_benchmark')
    
    # Create comprehensive summary
    summary = {
        'benchmark_timestamp': datetime.now().isoformat(),
        'data_validation': validation_results or {},
        'classification_benchmark': classification_results or {},
        'regression_benchmark': regression_results or {},
        'total_models_tested': (
            (classification_results.get('models_tested', 0) if classification_results else 0) +
            (regression_results.get('models_tested', 0) if regression_results else 0)
        ),
        'total_benchmark_duration': (
            (classification_results.get('duration', 0) if classification_results else 0) +
            (regression_results.get('duration', 0) if regression_results else 0)
        )
    }
    
    # Save summary to reports directory
    reports_dir = Path("/opt/airflow/reports/benchmarks")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = reports_dir / f"comprehensive_benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Benchmark summary saved to: {summary_file}")
    print(f"📊 Summary Details:")
    print(f"   - Data validation: {'✅' if validation_results else '❌'}")
    print(f"   - Classification models: {classification_results.get('models_tested', 0) if classification_results else 0}")
    print(f"   - Regression models: {regression_results.get('models_tested', 0) if regression_results else 0}")
    print(f"   - Total duration: {summary['total_benchmark_duration']/60:.1f} minutes")
    
    return str(summary_file)

def cleanup_old_benchmarks():
    """Clean up old benchmark files to save space"""
    from pathlib import Path
    import os
    from datetime import datetime, timedelta
    
    print("🧹 Cleaning up old benchmark files...")
    
    reports_dir = Path("/opt/airflow/reports/benchmarks")
    if not reports_dir.exists():
        print("   No benchmark reports directory found")
        return
    
    # Keep only last 5 benchmark summaries
    summary_files = sorted(reports_dir.glob("comprehensive_benchmark_summary_*.json"), reverse=True)
    
    files_removed = 0
    for old_file in summary_files[5:]:  # Keep first 5, remove rest
        try:
            old_file.unlink()
            files_removed += 1
        except Exception as e:
            print(f"   ⚠️  Could not remove {old_file}: {e}")
    
    print(f"   Removed {files_removed} old benchmark summary files")

# --- DAG Definition ---
default_args = {
    "owner": "ml_engineer",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    "comprehensive_model_benchmark",
    default_args=default_args,
    description="Run comprehensive benchmark of all classification and regression models",
    schedule_interval="@weekly",  # Run weekly or trigger manually
    catchup=False,
    tags=['benchmark', 'classification', 'regression', 'comprehensive', 'models'],
    max_active_runs=1,  # Prevent concurrent runs
) as dag:
    
    # Task 1: Validate training data
    validate_task = PythonOperator(
        task_id="validate_training_data",
        python_callable=validate_training_data
    )
    
    # Task 2: Run classification benchmark (can run in parallel with regression)
    classification_task = PythonOperator(
        task_id="run_classification_benchmark",
        python_callable=run_classification_benchmark
    )
    
    # Task 3: Run regression benchmark (can run in parallel with classification)
    regression_task = PythonOperator(
        task_id="run_regression_benchmark",
        python_callable=run_regression_benchmark
    )
    
    # Task 4: Generate comprehensive summary (runs after both benchmarks complete)
    summary_task = PythonOperator(
        task_id="generate_benchmark_summary",
        python_callable=generate_benchmark_summary,
        provide_context=True
    )
    
    # Task 5: Cleanup old benchmark files (runs after summary generation)
    cleanup_task = PythonOperator(
        task_id="cleanup_old_benchmarks",
        python_callable=cleanup_old_benchmarks
    )
    
    # Task dependencies
    # Validate data first, then run benchmarks in parallel, then generate summary and cleanup
    validate_task >> [classification_task, regression_task] >> summary_task >> cleanup_task