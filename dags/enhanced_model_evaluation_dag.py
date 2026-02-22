"""
Enhanced Model Evaluation DAG with Visualizations

This DAG extends the original model evaluation DAG by adding comprehensive
visualization capabilities for model performance analysis.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
project_root = Path("/opt/airflow")
sys.path.insert(0, str(project_root))

# Import the original evaluation functions
from model_evaluation_dag import (
    validate_test_data,
    evaluate_classification_models,
    evaluate_regression_models,
    generate_evaluation_comparison,
)

# Import visualization functions
from model_visualization import (
    create_classification_visualizations,
    create_regression_visualizations,
)


def create_comprehensive_visualizations(**context):
    """Create comprehensive visualizations after evaluation"""
    import os
    from pathlib import Path

    print("🎨 Creating comprehensive model performance visualizations...")

    # Create classification visualizations
    print("📊 Creating classification visualizations...")
    clf_viz_dir = create_classification_visualizations()

    # Create regression visualizations
    print("📈 Creating regression visualizations...")
    reg_viz_dir = create_regression_visualizations()

    # Create summary report in the same directory as visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(os.getenv("VIZ_OUTPUT_DIR", "./visualization_output"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / f"summary_{timestamp}.txt"

    with open(str(summary_path), "w") as f:
        f.write("MODEL PERFORMANCE VISUALIZATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("CLASSIFICATION VISUALIZATIONS:\n")
        f.write(f"  Location: {clf_viz_dir}\n")
        f.write("  Contents:\n")
        f.write("    - Confusion matrices for all models\n")
        f.write("    - ROC curves comparison\n")
        f.write("    - Precision-Recall curves\n")
        f.write("    - Metrics comparison charts\n\n")
        f.write("REGRESSION VISUALIZATIONS:\n")
        f.write(f"  Location: {reg_viz_dir}\n")
        f.write("  Contents:\n")
        f.write("    - Predicted vs Actual scatter plots\n")
        f.write("    - Residual analysis plots\n")
        f.write("    - Error distribution histograms\n")
        f.write("    - Metrics comparison charts\n")

    print(f"✅ Visualization summary saved to: {str(summary_path)}")
    print(f"📊 Classification visualizations: {clf_viz_dir}")
    print(f"📈 Regression visualizations: {reg_viz_dir}")

    return {
        "classification_viz_dir": str(clf_viz_dir),
        "regression_viz_dir": str(reg_viz_dir),
        "summary_path": str(summary_path),
    }


# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    "enhanced_model_evaluation",
    default_args=default_args,
    description="Enhanced Model Evaluation DAG with Visualizations",
    schedule_interval=None,
    catchup=False,
    tags=["ml", "evaluation", "visualization"],
)

# Task 1: Validate test data
validate_data_task = PythonOperator(
    task_id="validate_test_data",
    python_callable=validate_test_data,
    dag=dag,
)

# Task 2: Evaluate classification models
evaluate_clf_task = PythonOperator(
    task_id="evaluate_classification_models",
    python_callable=evaluate_classification_models,
    dag=dag,
)

# Task 3: Evaluate regression models
evaluate_reg_task = PythonOperator(
    task_id="evaluate_regression_models",
    python_callable=evaluate_regression_models,
    dag=dag,
)

# Task 4: Generate comparison report
generate_report_task = PythonOperator(
    task_id="generate_evaluation_comparison",
    python_callable=generate_evaluation_comparison,
    provide_context=True,
    dag=dag,
)

# Task 5: Create visualizations (NEW!)
create_visualizations_task = PythonOperator(
    task_id="create_comprehensive_visualizations",
    python_callable=create_comprehensive_visualizations,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
validate_data_task >> [evaluate_clf_task, evaluate_reg_task]
[evaluate_clf_task, evaluate_reg_task] >> generate_report_task
generate_report_task >> create_visualizations_task
